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


#for one decoder using MLP architecture
def train_decoder(eegenv, model, ds_loader, device, lr, n_epochs, epoch_tll_terminate,
                  min_delta=1e-4, input_mode="coeffs", noise_shape=None, grad_clip=1.0, save_freq=50,
                  save_pth="train/det_motor2a/exp_2/saves/", save_name="mlp"):

    assert input_mode in INPUT_MODES, f"mode must be one of {INPUT_MODES}, got {input_mode}"
    assert input_mode != "noise" or noise_shape is not None, "noise mode requires noise_shape"

    best_loss = float("inf")
    epochs_stuck = 0

    #move model to device
    model.to(device)

    det_solver, _ = eegenv.extract_solver_operator(is_torch=True, device=device) #get det solver for coeffs; blank is an identity matrix, ignore it

    #objectives 
    loss_fn = nn.BCEWithLogitsLoss() #applies sigmoid internally
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    #store the average over epochs
    training_metrics = {
        'Training/Probability Loss': [],
    }
    epoch_bar = tqdm(range(n_epochs), desc=f"epochs [{input_mode}]")
    model.train()


    for epoch in epoch_bar: #train over all samples per epoch

        #running sums for avg metric
        epoch_loss = torch.zeros((), device=device)
        n_steps = 0

        for windows, labels, _ in ds_loader: #for each window over entire training dist (for ffns only)
            window = windows.to(device) #B x seq_len=1 x nchns x F
            labels = labels.to(device) #B x seq_len=1 x classes

            #computed with no_grad() internally if input is coefficients
            inputs, _ = build_inputs(mode=input_mode, windows=window, solver=det_solver, 
                                     residual_operator=None, noise_shape=noise_shape, clip=model.electrode_clip)

            optim.zero_grad() #clear last .grad attributes
            feat_vec = inputs.squeeze(1) #remove seq=1 dim; B x inputdim x F
            label = labels.squeeze(1) #remove seq=1 dim; B x classes

            logits, _ = model(feat_vec) #B x output dim
            loss = loss_fn(logits, label)
            loss.backward() #compute .grad attribute

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optim.step() #update weights

            epoch_loss += loss.detach()
            n_steps += 1

        #compute average training metrics
        avg_loss = (epoch_loss / n_steps).item()
        training_metrics['Training/Probability Loss'].append(avg_loss)
        epoch_bar.set_postfix(loss=avg_loss, stuck=epochs_stuck)

        #stuck means loss failed to improve by at least min_delta this epoch
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_stuck = 0
        else:
            epochs_stuck += 1


        if (epoch + 1) % save_freq == 0:
            save_metrics(training_metrics, save_pth, epoch, filename=f"{save_name}_e{epoch+1}_metrics.json")
            save_motor2a_model(model, optim, epoch+1, save_pth, save_name)

        if epochs_stuck >= epoch_tll_terminate:
            print(f"loss stuck for {epochs_stuck} epochs, stopping early at epoch {epoch+1}")
            break

    save_metrics(training_metrics, save_pth, epoch, filename=f"{save_name}_e{epoch+1}_metrics.json")
    save_motor2a_model(model, optim, epoch+1, save_pth, save_name)
    
    model.eval()
    return model, training_metrics