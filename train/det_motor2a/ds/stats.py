import numpy as np

#one pass over the raw cached arrays
#sources is a list of Motor2aCache
def stack_features(sources, dtype=np.float64):
    return [np.array(ds.features, dtype=dtype) for ds in sources] #each n x nchns x F

#per feature upper bound, i.e what value do 99.9% of this feature's readings
#fall below? anything above it gets clipped to that bound; computed across entire training set
#and is the value a model will use to clip its input to avoid outlier magnitudes
#affecting gradients
# clip = np.percentile(features[..., complexity_idx], 99.9)  # one float per feature
# x = x.clamp(max=clip)
def compute_feature_clips(sources, percentile=99.9):
    flat_features = np.concatenate([w.reshape(-1, w.shape[-1]) for w in stack_features(sources, np.float32)],
                            axis=0) #total x F

    return np.percentile(flat_features, percentile, axis=0).astype(np.float32) #F

#per chns per feature mean over every window
#clipped first so the stored mean is not dragged by values the model never sees at runtime
def compute_electrode_feature_means(sources, clip):
    total, count = None, 0

    for windows in stack_features(sources):
        windows = np.minimum(windows, clip) #F, broadcasts across window and chns
        block = windows.sum(axis=0)

        total = block if total is None else total + block
        count += windows.shape[0]

    return (total / count).astype(np.float32) #nchns x F

#per chns per feature variance over every window
def compute_electrode_feature_variance(sources, clip, mean):
    total, count = None, 0

    for windows in stack_features(sources):
        windows = np.minimum(windows, clip)
        block = ((windows - mean) ** 2).sum(axis=0)

        total = block if total is None else total + block
        count += windows.shape[0]

    return (total / count).astype(np.float32) #nchns x F

#clip, mean and variance for one set of sources, plus the window count 
def compute_feature_stats(sources, percentile=99.9):
    clip = compute_feature_clips(sources, percentile)
    mean = compute_electrode_feature_means(sources, clip)
    variance = compute_electrode_feature_variance(sources, clip, mean)

    n_windows = sum(len(ds.features) for ds in sources)
    return {"clip": clip, "mean": mean, "variance": variance, "n_windows": n_windows}


def print_stats_row(label, stats, feature_names):
    clip, mean, variance = stats["clip"], stats["mean"], stats["variance"]

    for i, name in enumerate(feature_names):
        row = [label, name,
               f"{clip[i]:.4g}", #for each feature, {percentile}% of values are below this clip
               f"{mean[:, i].mean():.4g}", #mean of channel means
               f"{np.sqrt(variance[:, i]).mean():.4g}", #mean of channel std
               f"{mean[:, i].std():.4g}"] #std of channel means

        print("".join(cell.ljust(24) for cell in row))

#per window size first, then every window size flattened as one; only the flattened stats are
#returned, since those are what the model is constructed with
def report_feature_stats(sources, percentile=99.9):
    feature_names = sources[0].feature_names

    headers = ["source", "feature", f"clip ({percentile}% below)",
               "mean of chn means", "mean of chn stds", "std of chn means"]

    print("=" * 144)
    print("".join(h.ljust(24) for h in headers))

    for ds in sources:
        print_stats_row(f"wz {ds.window_seconds}", compute_feature_stats([ds], percentile),
                        feature_names)

    stats = compute_feature_stats(sources, percentile)

    print("-" * 76)
    print_stats_row("all", stats, feature_names)
    print(f"\n{stats['n_windows']} windows across {len(sources)} window sizes | percentile {percentile}")

    return stats