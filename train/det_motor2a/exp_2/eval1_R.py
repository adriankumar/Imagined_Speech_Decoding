import os, json
import torch

from models.det_motor2a import Motor2aMLP1D, Motor2aMLP2D, load_motor2a_model
from global_lvl import load_eegenv
from ..ds.loaders import build_window_loader
from .eval_helpers.helpers import collect_all
from .eval_helpers.metrics import alignment_per_window
from .eval_helpers.plots import plot_loss_curves, plot_task_vs_idle, plot_per_head

#file subject to change to accomodate for the ltc version
#eval1 is the evaluation for "which is the best peforming on average?"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#===load both splits===
wsizes = [0.2, 0.5, 0.7] #window sizes of cached motor2a features
train_root = "F:/EEG_datasets/cached/deterministic_motor"
test_root = "F:/EEG_datasets/cached/deterministic_motor/test"

train_paths = [f"{train_root}/wz_{str(wz)}" for wz in wsizes]
test_paths = [f"{test_root}/wz_{str(wz)}" for wz in wsizes]

thresh = 1.0 #matches the training condition, partial windows are for the winner sweep
batch_size = 64

classes = ["hand", "feet", "tongue", "left", "right"]

#decoder classes x motor2a labels
class_encodings = {
    0: (0, 0, 0, 0, 0), #rest implicit — no positive head, derived later
    1: (1, 0, 0, 1, 0), #left hand
    2: (1, 0, 0, 0, 1), #right hand
    3: (0, 1, 0, 1, 1), #both feet
    4: (0, 0, 1, 0, 0)  #tongue
}

#loader unused, the sources are passed to eval so window size stays separable
train_sources, _ = build_window_loader(cache_paths=train_paths, class_encoding=class_encodings,
                                       classes=classes, batch_size=batch_size,
                                       threshold=thresh, shuffle=False)

test_sources, _ = build_window_loader(cache_paths=test_paths, class_encoding=class_encodings,
                                      classes=classes, batch_size=batch_size,
                                      threshold=thresh, shuffle=False)

#===env loading===
eegenv_pth = "train/det_motor2a/ds/motor2a_env.json"
eegenv = load_eegenv(config_path=eegenv_pth, print_channel_resolve=False)
L_degree = 2 #9 coefficients for 22 electrodes
eegenv.change_L(L_degree=L_degree)

#===models under comparison===
#the two decoder loads its stage two checkpoint, which holds the whole model
one_decoder_pth = "train/det_motor2a/exp_2/saves/one_decoder/mlp_one_decoder_coeff_L2_210.pt"
two_decoder_pth = "train/det_motor2a/exp_2/saves/two_decoder/mlp_two_decoder_coeff_L2_s2_157.pt"

checkpoints = [(one_decoder_pth, Motor2aMLP1D), (two_decoder_pth, Motor2aMLP2D)]

#display name is the checkpoint filename without its extension
models = {os.path.splitext(os.path.basename(path))[0]: load_motor2a_model(path, cls, device=device)
          for path, cls in checkpoints}

#the two decoder is the only one with a real gate, so only it gets a gated score
gated_names = [name for name, model in models.items() if hasattr(model, 'decode_1')]

#===collect outputs===
input_mode = "coeffs"

outputs = {'train': collect_all(models=models, sources=train_sources, eegenv=eegenv, device=device,
                                input_mode=input_mode, batch_size=batch_size),
           'test': collect_all(models=models, sources=test_sources, eegenv=eegenv, device=device,
                               input_mode=input_mode, batch_size=batch_size)}

#===alignment===
threshold = 0.5 #fixed baseline, sweeping the gate is reserved for the winner

scores, pooled, gated_scores, gated_pooled = {}, {}, {}, {}

for split, per_model in outputs.items():
    scores[split], pooled[split] = {}, {}
    gated_scores[split], gated_pooled[split] = {}, {}

    for name, per_window in per_model.items():
        scores[split][name], pooled[split][name] = alignment_per_window(per_window_outputs=per_window,
                                                                        threshold=threshold, gated=False)

        #wrongly-closed windows score all-negative, the denominator stays the active set
        if name in gated_names:
            gated_scores[split][name], gated_pooled[split][name] = alignment_per_window(
                per_window_outputs=per_window, threshold=threshold, gated=True)

#===summary for the table===
summary_pth = "train/det_motor2a/exp_2/saves/eval_summary.json"
os.makedirs(os.path.dirname(summary_pth), exist_ok=True)

with open(summary_pth, "w", encoding="utf-8") as f:
    json.dump({'threshold': threshold, 'partial_window_threshold': thresh, 'L_degree': L_degree,
               'per_window': scores, 'pooled': pooled,
               'per_window_gated': gated_scores, 'pooled_gated': gated_pooled}, f, indent=1)

#===plots===
metrics_paths = ["train/det_motor2a/exp_2/saves/one_decoder/mlp_one_decoder_coeff_L2_e210_metrics.json",
                 "train/det_motor2a/exp_2/saves/two_decoder/mlp_two_decoder_coeff_L2_metrics.json"]

plot_loss_curves(metrics_paths=metrics_paths, names=list(models.keys()))

for split in ('train', 'test'):
    plot_task_vs_idle(scores=scores[split], split=split)
    plot_per_head(scores=scores[split], split=split,
                  gated_scores=gated_scores[split] if gated_scores[split] else None)