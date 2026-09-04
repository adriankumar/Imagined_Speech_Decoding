import json
import numpy as np
import matplotlib.pyplot as plt

from .metrics import CLASS_NAMES

#metric keys written by the two training functions
STAGE_1_KEY = 'Stage 1/Probability Loss'
STAGE_2_KEY = 'Stage 2/Probability Loss'
ONE_DECODER_KEY = 'Training/Probability Loss'

BAR_WIDTH = 0.8

#draws a baseline marker spanning the width of one bar
def baseline_marker(ax, centre, width, value, label=None):
    ax.hlines(y=value, xmin=centre - width/2, xmax=centre + width/2,
              colors='black', linestyles='dashed', linewidth=1.2, label=label, zorder=3)

#evenly spaced bar centres for n series within one group slot
def bar_offsets(n_series, group_width=BAR_WIDTH):
    width = group_width / n_series
    return np.arange(n_series) * width - group_width/2 + width/2, width

#stage 1 is the gate, stage 2 is task-specific; the one decoder sits in the stage 2 panel
#since that is where task-specific decoding is learnt, it has nothing to compare against in stage 1
def plot_loss_curves(metrics_paths, names):
    curves = {}
    for path, name in zip(metrics_paths, names):
        with open(path, "r", encoding="utf-8") as f:
            curves[name] = json.load(f)["metrics"]

    panels = [(STAGE_1_KEY, "stage 1, task-active vs idle", [STAGE_1_KEY]),
              (STAGE_2_KEY, "stage 2, task-specific", [STAGE_2_KEY, ONE_DECODER_KEY])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (_, title, keys) in zip(axes, panels):
        plotted = []

        for name, metrics in curves.items():
            for key in keys:
                if key in metrics:
                    ax.plot(range(1, len(metrics[key]) + 1), metrics[key], label=name)
                    plotted.extend(metrics[key])

        #shared ylim within the panel so the curves stay relative to each other
        if plotted:
            margin = (max(plotted) - min(plotted)) * 0.05
            ax.set_ylim(min(plotted) - margin, max(plotted) + margin)

        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE loss")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("training loss")
    fig.tight_layout()
    plt.show()

#scores is name -> window_seconds -> alignment dict
def plot_task_vs_idle(scores, split):
    window_sizes = sorted({wz for per_window in scores.values() for wz in per_window})
    names = list(scores.keys())

    offsets, width = bar_offsets(len(names))
    positions = np.arange(len(window_sizes))

    fig, ax = plt.subplots(figsize=(9, 5))

    for offset, name in zip(offsets, names):
        values = [scores[name][wz]['task_vs_idle'] for wz in window_sizes]
        centres = positions + offset

        ax.bar(centres, values, width=width, label=name)

        #each window size has its own class balance, so its own baseline
        for centre, wz in zip(centres, window_sizes):
            baseline_marker(ax, centre, width, scores[name][wz]['baseline_task_vs_idle'])

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{wz}s" for wz in window_sizes])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("window size")
    ax.set_ylabel("alignment to label assumptions")
    ax.set_title(f"task-active vs idle [{split}]\ndashed marks the majority-class rate")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    plt.show()

#one panel per window size; gated_scores is optional and only exists for the two decoder
def plot_per_head(scores, split, gated_scores=None):
    window_sizes = sorted({wz for per_window in scores.values() for wz in per_window})

    #a gated series is a separate bar beside its ungated counterpart
    series = [(name, scores, name) for name in scores]
    if gated_scores is not None:
        series += [(name, gated_scores, f"{name} (gated)") for name in gated_scores]

    offsets, width = bar_offsets(len(series))
    positions = np.arange(len(CLASS_NAMES))

    fig, axes = plt.subplots(1, len(window_sizes), figsize=(6 * len(window_sizes), 5), squeeze=False)

    for ax, wz in zip(axes[0], window_sizes):
        for offset, (name, source, label) in zip(offsets, series):
            values = [source[name][wz]['per_head'][head] for head in CLASS_NAMES]
            centres = positions + offset

            ax.bar(centres, values, width=width, label=label)

            for centre, head in zip(centres, CLASS_NAMES):
                baseline_marker(ax, centre, width, source[name][wz]['baseline_per_head'][head])

        ax.set_xticks(positions)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("head")
        ax.set_ylabel("alignment to label assumptions")
        ax.set_title(f"window {wz}s")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f"task-specific per head, active windows only [{split}]\ndashed marks the majority-class rate")
    fig.tight_layout()
    plt.show()