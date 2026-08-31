import numpy as np
import matplotlib.pyplot as plt

#retention against basis degree, one line per feature
#the log panel shows what the linear one cannot, since everything above L=0 sits near 1
def plot_retention(retention, feature_names, save_path=None):
    labels = list(retention)
    degrees = sorted(retention[labels[0]])
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
 
    for label in labels:
        style = "-" if label == "all" else "--"
        width = 2.0 if label == "all" else 1.0
 
        for name in feature_names:
            values = [retention[label][L][name] for L in degrees]
 
            axes[0].plot(degrees, values, marker="o", markersize=4, linestyle=style,
                         linewidth=width, label=f"{name}, wz {label}")
            axes[1].plot(degrees, [1 - v for v in values], marker="o", markersize=4,
                         linestyle=style, linewidth=width, label=f"{name}, wz {label}")
 
    axes[0].set_ylabel("variance retained")
    axes[0].set_title("electrode variance retained at each L")
    axes[0].set_ylim(0, 1.02)
 
    axes[1].set_ylabel("variance loss (log-scale)")
    axes[1].set_title("compression loss of variance at each L")
    axes[1].set_yscale("log")
 
    for ax in axes:
        ax.set_xlabel("basis degree L")
        ax.set_xticks(degrees)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
 
    fig.suptitle("basis retention over entire training distribution")
    fig.tight_layout()
 
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
 
    return fig
 
 
#electrode space only, since coeff space has a different position count and the solver
#is not norm preserving, so the two cannot share an axis
#left panel is how much one channel moves over windows, right is how much the channels
#differ from each other, which is the quantity the input centring removes
def plot_channel_stats(source_stats, feature_names, save_path=None):
    window_sizes = sorted(source_stats)
    x = np.arange(len(window_sizes))
    width = 0.8 / len(feature_names)
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
 
    for i, name in enumerate(feature_names):
        chn_stds = [np.sqrt(source_stats[wz]["variance"][:, i]).mean() for wz in window_sizes]
        chn_means = [source_stats[wz]["mean"][:, i].std() for wz in window_sizes]
 
        offset = (i - (len(feature_names) - 1) / 2) * width
        axes[0].bar(x + offset, chn_stds, width, label=name)
        axes[1].bar(x + offset, chn_means, width, label=name)
 
    axes[0].set_ylabel("mean of channel stds")
    axes[0].set_title("mean of channel stds, per window size")
 
    axes[1].set_ylabel("std of channel means")
    axes[1].set_title("std of channel means, per window size")
 
    for ax in axes:
        ax.set_xlabel("window size (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(window_sizes)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
 
    fig.suptitle("electrode space stats over entire training distribution")
    fig.tight_layout()
 
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
 
    return fig