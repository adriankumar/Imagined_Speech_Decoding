import numpy as np 
from ..ds.stats import (stack_features, compute_feature_clips)

#coeff and residual space, where the operator is the solver or the residual operator

#get mean coeffs or resiudal from the mean features with linear transform mean(Ax) = A mean(x)
def mean_through_operator(mean_features, operator):
    return np.einsum("pe, ef -> pf", operator, mean_features).astype(np.float32) #p x F
 
#one window moved into the operator's space, p is coeffs or nchns depending which
def apply_operator(windows, operator):
    return np.einsum("pe, wef -> wpf", operator, windows) #w x p x F
 
#variance is not linear
#so the coeff and residual variance need their own pass over the windows
def compute_operator_feature_means(sources, clip, operator):
    total, count = None, 0
 
    for windows in stack_features(sources):
        moved = apply_operator(np.minimum(windows, clip), operator)
        block = moved.sum(axis=0)
 
        total = block if total is None else total + block
        count += moved.shape[0]
 
    return (total / count).astype(np.float32) #p x F
 
def compute_operator_feature_variance(sources, clip, operator, mean):
    total, count = None, 0
 
    for windows in stack_features(sources):
        moved = apply_operator(np.minimum(windows, clip), operator)
        block = ((moved - mean) ** 2).sum(axis=0)
 
        total = block if total is None else total + block
        count += moved.shape[0]
 
    return (total / count).astype(np.float32) #p x F
 
#the clip stays in electrode space before any operator is applied
def compute_operator_feature_stats(sources, operator, percentile=99.9):
    clip = compute_feature_clips(sources, percentile)
    mean = compute_operator_feature_means(sources, clip, operator)
    variance = compute_operator_feature_variance(sources, clip, operator, mean)
 
    n_windows = sum(len(ds.features) for ds in sources)
    return {"clip": clip, "mean": mean, "variance": variance, "n_windows": n_windows}
 
 
#------------
#reporting
#------------
def print_stats_row(label, stats, feature_names):
    clip, mean, variance = stats["clip"], stats["mean"], stats["variance"]
 
    for i, name in enumerate(feature_names):
        row = [label, name,
               f"{clip[i]:.4g}", #for each feature, {percentile}% of values are below this clip
               f"{mean[:, i].mean():.4g}", #mean of position means
               f"{np.sqrt(variance[:, i]).mean():.4g}", #mean of position std
               f"{mean[:, i].std():.4g}"] #std of position means
 
        print("".join(cell.ljust(24) for cell in row))
 
#same table for a coeff or residual operator, one row per source then the pooled row
def report_operator_stats(sources, operator, percentile=99.9, feature_src="coeff"):
    assert feature_src in ["coeff", "chn"], f"feature_src must be either coeff or chn, got {feature_src}"

    feature_names = sources[0].feature_names
 
    headers = ["source", "feature", f"clip ({percentile}% below)",
               f"mean of {feature_src} means", f"mean of {feature_src} stds", f"std of {feature_src} means"]
 
    print("=" * 144)
    print("".join(h.ljust(24) for h in headers))
 
    for ds in sources:
        stat = compute_operator_feature_stats([ds], operator, percentile)
        print_stats_row(f"wz {ds.window_seconds}", stat, feature_names)
 
    stats = compute_operator_feature_stats(sources, operator, percentile)
 
    print("-" * 144)
    print_stats_row("all", stats, feature_names)
    print(f"\n{stats['n_windows']} windows across {len(sources)} window sizes | percentile {percentile}")
 
    return stats
 
#how much of the electrode variance is recoverable by the residual per feature
#use 1 - var_ratio when operator stats = residual stats as it shows the 'retention' of L
#using solver as operator is meaningless because its in SH space not electrode-space
def residual_variance_ratio(residual_stats, electrode_stats):
    return residual_stats["variance"].mean(axis=0) / electrode_stats["variance"].mean(axis=0) #F