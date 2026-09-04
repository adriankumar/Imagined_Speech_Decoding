import torch
from torch.utils.data import DataLoader
from ...input_helpers import build_inputs

#gate probability is where the two architectures differ
#1D has no gate head
def gate_from_model(model, feat_vec):
    if hasattr(model, 'decode_1'):
        logits_1, logits_2, _, _ = model(x=feat_vec)
        return torch.sigmoid(logits_1), torch.sigmoid(logits_2) #B x 1, B x classes

    logits, _ = model(x=feat_vec)
    probs = torch.sigmoid(logits) #B x classes
    return probs.max(dim=-1, keepdim=True).values, probs #B x 1, B x classes


#runs one source end to end, returns stacked probabilities and targets
def collect_outputs(model, source, eegenv, device, input_mode, batch_size=64):
    loader = DataLoader(dataset=source, batch_size=batch_size, shuffle=False)

    det_solver, _ = eegenv.extract_solver_operator(is_torch=True, device=device) #blank is an identity matrix, ignore it

    model.to(device)
    model.eval()

    all_gate, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for windows, labels, _ in loader: #run index unused
            window = windows.to(device) #B x seq_len=1 x nchns x F
            labels = labels.to(device) #B x seq_len=1 x classes

            inputs, _ = build_inputs(mode=input_mode, windows=window, solver=det_solver,
                                     residual_operator=None, noise_shape=None,
                                     clip=model.electrode_clip)

            feat_vec = inputs.squeeze(1) #remove seq=1 dim; B x input_dim x F
            label = labels.squeeze(1) #remove seq=1 dim; B x classes

            gate_probs, probs = gate_from_model(model=model, feat_vec=feat_vec)

            all_gate.append(gate_probs.cpu())
            all_probs.append(probs.cpu())
            all_targets.append(label.cpu())

    return {'gate_probs': torch.cat(all_gate).numpy(), #N x 1
            'probs': torch.cat(all_probs).numpy(), #N x classes
            'targets': torch.cat(all_targets).numpy(), #N x classes
            'window_seconds': source.window_seconds}


#models is name -> loaded model, sources is the list of caches for one split
#returns name -> window_seconds -> outputs
def collect_all(models, sources, eegenv, device, input_mode, batch_size=64):
    results = {}

    for name, model in models.items():
        per_window = {}

        for source in sources:
            outputs = collect_outputs(model=model, source=source, eegenv=eegenv, device=device,
                                      input_mode=input_mode, batch_size=batch_size)

            per_window[outputs['window_seconds']] = outputs

        results[name] = per_window

    return results