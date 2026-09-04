import torch
import torch.nn as nn
from tqdm import tqdm
from models.det_motor2a import save_motor2a_model
from ...input_helpers import (INPUT_MODES, build_inputs)
import os, json

def save_metrics(metrics, save_path, epoch, filename):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, filename), "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "metrics": metrics}, f, indent=1)


#windows are B x seq_len=1 x nchns x F, labels as B x seq_len=1 x classes
#coefficients computed with no_grad() internally, clipping applied before the solver
def prepare_batch(windows, labels, device, input_mode, det_solver, noise_shape, clip):
    window = windows.to(device)
    labels = labels.to(device)

    inputs, _ = build_inputs(mode=input_mode, windows=window, solver=det_solver,
                             residual_operator=None, noise_shape=noise_shape, clip=clip)

    return inputs.squeeze(1), labels.squeeze(1) #B x input_dim x F, B x classes


#a window is task-active when any task head has a positive label, all-zeros is the implicit rest label
def active_mask(label):
    return label.max(dim=-1, keepdim=True).values #B x 1

#stage one, task-active vs idle/rest over every window in the batch
def stage_1_step(model, feat_vec, label, loss_fn):
    latent_1, _ = model.attend_gate(x=feat_vec)
    logits_1 = model.gate_logits(latent_1=latent_1) #B x out1_dim

    return loss_fn(logits_1, active_mask(label))

#stage two, task-specific over the active windows only
#rest samples in the batch are masked out of the reduction rather than dropped,
#so its a zero-gradient step instead of an empty forward
def stage_2_step(model, feat_vec, label, loss_fn):
    _, logits_2, _, _ = model(x=feat_vec) #gate logits unused, decoder 1 is frozen

    active = active_mask(label) #B x 1
    per_window = loss_fn(logits_2, label).mean(dim=-1, keepdim=True) #B x 1

    #divide by active count so gradient scale does not track batch composition
    return (per_window * active).sum() / active.sum().clamp(min=1)


#shared loop for both stages; step_fn does one batch and returns a scalar loss
def run_epochs(model, ds_loader, optim, step_fn, n_epochs, epoch_tll_terminate, min_delta,
               device, grad_clip, save_freq, save_pth, save_name, stage_desc):

    losses = []
    best_loss = float("inf")
    epochs_stuck = 0

    epoch_bar = tqdm(range(n_epochs), desc=stage_desc)
    model.train()

    for epoch in epoch_bar:

        #running sums for avg metric
        epoch_loss = torch.zeros((), device=device)
        n_steps = 0

        for batch in ds_loader:
            optim.zero_grad() #clear last .grad attributes

            loss = step_fn(batch)
            loss.backward() #compute .grad attribute

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optim.step() #update weights

            epoch_loss += loss.detach()
            n_steps += 1

        #compute average training metrics
        avg_loss = (epoch_loss / n_steps).item()
        losses.append(avg_loss)
        epoch_bar.set_postfix(loss=avg_loss, stuck=epochs_stuck)

        #stuck means loss failed to improve by at least min_delta this epoch
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_stuck = 0
        else:
            epochs_stuck += 1

        if (epoch + 1) % save_freq == 0:
            save_motor2a_model(model, optim, epoch+1, save_pth, save_name)

        if epochs_stuck >= epoch_tll_terminate:
            print(f"loss stuck for {epochs_stuck} epochs, stopping early at epoch {epoch+1}")
            break

    save_motor2a_model(model, optim, epoch+1, save_pth, save_name)
    return losses, epoch+1


#the frozen decoder must be deterministic, otherwise its dropout makes latent 1 stochastic
def assert_gate_frozen(model, feat_vec):
    model.train()
    with torch.no_grad():
        first, _ = model.attend_gate(x=feat_vec)
        second, _ = model.attend_gate(x=feat_vec)

    assert torch.equal(first, second), "decoder 1 is not deterministic after freezing, dropout is still active"


#for two decoders using MLP architecture; stage one trains the gate, stage two the task heads
def train_decoder(eegenv, model, ds_loader, device, lr_1, stage1_epochs, stage2_epochs,
                  stage1_patience, stage2_patience, lr_2=None, min_delta=1e-4,
                  input_mode="coeffs", noise_shape=None, grad_clip=1.0, save_freq=50,
                  save_pth="train/det_motor2a/exp_2/saves/", save_name="mlp2d"):

    assert input_mode in INPUT_MODES, f"mode must be one of {INPUT_MODES}, got {input_mode}"
    assert input_mode != "noise" or noise_shape is not None, "noise mode requires noise_shape"

    lr_2 = lr_1 if lr_2 is None else lr_2

    #move model to device
    model.to(device)

    det_solver, _ = eegenv.extract_solver_operator(is_torch=True, device=device) #get det solver for coeffs; blank is an identity matrix, ignore it

    #batch preparation is identical across stages
    def prepare(batch):
        windows, labels, _ = batch #run index unused
        return prepare_batch(windows=windows, labels=labels, device=device, input_mode=input_mode,
                             det_solver=det_solver, noise_shape=noise_shape, clip=model.electrode_clip)

    #===stage one, gate===
    gate_loss_fn = nn.BCEWithLogitsLoss() #applies sigmoid internally
    optim_1 = torch.optim.Adam(model.decode_1.parameters(), lr=lr_1)

    def step_1(batch):
        feat_vec, label = prepare(batch)
        return stage_1_step(model=model, feat_vec=feat_vec, label=label, loss_fn=gate_loss_fn)

    stage1_losses, stage1_end = run_epochs(model=model, ds_loader=ds_loader, optim=optim_1, step_fn=step_1,
                                           n_epochs=stage1_epochs, epoch_tll_terminate=stage1_patience,
                                           min_delta=min_delta, device=device, grad_clip=grad_clip,
                                           save_freq=save_freq, save_pth=save_pth,
                                           save_name=f"{save_name}_s1", stage_desc=f"stage 1 [{input_mode}]")

    #===freeze the gate before stage two===
    model.freeze_decode_1()

    feat_vec, _ = prepare(next(iter(ds_loader)))
    assert_gate_frozen(model=model, feat_vec=feat_vec)

    #===stage two, task heads===
    #reduction none so rest windows can be excluded before the loss is reduced
    task_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    #built after the freeze so no optimiser state is held for frozen parameters
    optim_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_2)

    def step_2(batch):
        feat_vec, label = prepare(batch)
        return stage_2_step(model=model, feat_vec=feat_vec, label=label, loss_fn=task_loss_fn)

    stage2_losses, stage2_end = run_epochs(model=model, ds_loader=ds_loader, optim=optim_2, step_fn=step_2,
                                           n_epochs=stage2_epochs, epoch_tll_terminate=stage2_patience,
                                           min_delta=min_delta, device=device, grad_clip=grad_clip,
                                           save_freq=save_freq, save_pth=save_pth,
                                           save_name=f"{save_name}_s2", stage_desc=f"stage 2 [{input_mode}]")

    training_metrics = {
        'Stage 1/Probability Loss': stage1_losses,
        'Stage 2/Probability Loss': stage2_losses,
    }

    save_metrics(training_metrics, save_pth, {'stage_1': stage1_end, 'stage_2': stage2_end},
                 filename=f"{save_name}_metrics.json")

    model.eval()
    return model, training_metrics