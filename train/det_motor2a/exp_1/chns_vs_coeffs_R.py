from ..ds.loaders import build_window_loader
from ..ds.stats import (report_feature_stats, compute_feature_stats)

from .compression_stats import (report_operator_stats, residual_variance_ratio, 
                                compute_operator_feature_stats)

from .metrics import (plot_channel_stats, plot_retention)

from global_lvl import load_eegenv
import torch 
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#===load dataset===
wsizes = [0.2, 0.5, 0.7] #window sizes of cached motor2a features
paths = [f"F:/EEG_datasets/cached/deterministic_motor/wz_{str(wz)}" for wz in wsizes]

#placeholders for loader, which we dont use here
thresh = 1.0 
batch_size = 64 
shuffle = True 
drop_irregular_sequence = False 

#cache srcs is a list of Motor2aCache classes, there should be 3
cache_srcs, _ = build_window_loader(cache_paths=paths, batch_size=batch_size, 
                                    threshold=thresh, shuffle=shuffle, 
                                    drop_last=drop_irregular_sequence)

assert len(cache_srcs) == len(wsizes), f"there are only 3 window sizes in the cache, but loaded {len(cache_srcs)}"
#==================================


#===eegenv loading===
eegenv_pth = "train/det_motor2a/ds/motor2a_env.json"
eegenv = load_eegenv(config_path=eegenv_pth, print_channel_resolve=True) #motor2a has 25 channels but only 22 EEG ones are used
#==================================

#===training distribution recovery===
percentile = 99.9 
L_ablations = [0, 1, 2, 3]
solver, _ = eegenv.extract_solver_operator(is_torch=False)
res_op = eegenv.extract_residual_operator(is_torch=False)
feature_names = cache_srcs[0].feature_names

#compute stats
electrode_stats = report_feature_stats(sources=cache_srcs, percentile=percentile)

coeff_stats = report_operator_stats(sources=cache_srcs, operator=solver, 
                                    percentile=percentile, feature_src="coeff")

residual_stats = report_operator_stats(sources=cache_srcs, operator=res_op, 
                                       percentile=percentile, feature_src="chn")

for i, name in enumerate(cache_srcs[0].feature_names):
    ratio = residual_variance_ratio(residual_stats, electrode_stats)[i]
    print(f"{name:<14}residual keeps {ratio:.4g} | basis retains {1 - ratio:.4g}")

#==================================
#==================================
#plot stats
def collect_source_stats(sources, percentile=99.9):
    return {ds.window_seconds: compute_feature_stats([ds], percentile) for ds in sources}

source_stats = collect_source_stats(cache_srcs, percentile)
plot_channel_stats(source_stats, feature_names)

#==================================
#==================================
#plot L retention
def retention_over_L(eegenv, sources, electrode_stats, L_values, percentile):
    feature_names = sources[0].feature_names
    retention = {}
 
    for L in L_values:
        eegenv.change_L(L)
        res_op = eegenv.extract_residual_operator(is_torch=False)
 
        stats = compute_operator_feature_stats(sources, res_op, percentile)
        ratio = residual_variance_ratio(stats, electrode_stats)
 
        retention[L] = {name: float(1 - ratio[i]) for i, name in enumerate(feature_names)}
 
    return retention

def collect_retention(eegenv, sources, electrode_stats, L_values, percentile=99.9):
    retention = {}
 
    for ds in sources:
        own_stats = compute_feature_stats([ds], percentile)
        retention[str(ds.window_seconds)] = retention_over_L(eegenv, [ds], own_stats, L_values, percentile)
 
    retention["all"] = retention_over_L(eegenv, sources, electrode_stats, L_values, percentile)
 
    for label, by_L in retention.items():
        for L, values in by_L.items():
            print(f"{label:<6} L={L} | " + " | ".join(f"{n} {v:.4g}" for n, v in values.items()))
 
    return retention

retention = collect_retention(eegenv, cache_srcs, electrode_stats, L_ablations, percentile)
plot_retention(retention, feature_names)
#==================================

plt.show()