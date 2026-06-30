import io
import base64
import numpy as np

import matplotlib
matplotlib.use("Agg")  #headless backend, the gui never shows a window
import matplotlib.pyplot as plt

#encode a figure to a base64 png data url and close it, the single exit every renderer funnels through
def _fig_to_url(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

#====================================================================
#feature representation, image is the interpolated field, stack is the electrode scatter
#====================================================================
#draw an (array, names) pair to a base64 png, one panel per feature, each panel self-normalises
#image draws the interpolated field, stack scatters raw values at the electrode positions
#self-normalising per panel is the stage-3 carry-over, the shared-scale mode lands with the vis panel rework
def _render_array(env, array, names, kind):
    F = array.shape[2] if kind == "image" else array.shape[1]
    cols = min(F, 3)
    rows = int(np.ceil(F / cols))
    #fixed figure size keeps every render the same pixel dimensions, no relayout jitter across windows
    fig = plt.figure(figsize=(cols * 3.0, rows * 2.8), dpi=100)
    pos_2d = env.electrode_pos_2d if kind == "stack" else None

    for i in range(F):
        ax = fig.add_subplot(rows, cols, i + 1)
        if kind == "image":
            m = ax.imshow(array[:, :, i], origin="lower", cmap="viridis")
        else:
            m = ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c=array[:, i], cmap="viridis", s=30)
            ax.set_aspect("equal")
        ax.set_title(names[i], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    #fixed margins instead of tight_layout so the panel box never shifts between renders
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.04, wspace=0.25, hspace=0.30)
    return _fig_to_url(fig)

#render one feature stack, image kind interpolates through M, stack kind scatters at the electrodes
def render_stack_image(env, stack, names, kind):
    array = env.to_image(stack) if kind == "image" else stack
    return _render_array(env, array, names, kind)

#====================================================================
#recording strip, a channels-by-time raster of an arbitrary span of the timeline
#====================================================================
#compressed raster over [start, stop), each column one time bin, each cell the per-channel mean absolute amplitude
#reads in bounded chunks through the span primitive so a long span never sits in memory at once
#the span makes the zoom and the full-recording strip the one call, the view supplies start and stop
def build_strip_png(env, start, stop, width=1000):
    start = max(0, int(start))
    stop = min(int(stop), env.timepoints)
    span = max(1, stop - start)
    width = int(min(width, span))
    edges = np.linspace(start, stop, width + 1, dtype=int)  #absolute sample edges over the span
    n_ch = env.n_chns
    raster = np.empty((n_ch, width), dtype=np.float32)

    budget = 100000  #samples per disk read, caps the transient memory of each chunk
    c0 = 0
    while c0 < width:
        bin_span = max(1, int(edges[c0 + 1] - edges[c0]))
        c1 = min(c0 + max(1, budget // bin_span), width)
        s0, s1 = int(edges[c0]), int(edges[c1])
        data = np.abs(env.read_referenced_span(s0, s1))
        for j in range(c0, c1):
            a, b = int(edges[j]) - s0, int(edges[j + 1]) - s0
            raster[:, j] = data[:, a:b].mean(axis=1) if b > a else 0.0
        c0 = c1

    buf = io.BytesIO()
    plt.imsave(buf, raster, cmap="viridis", format="png")  #imsave normalises to the raster's own range
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii"), n_ch, width

#====================================================================
#decode simulation, current, delta, decoded, true next, one row per feature
#include_delta False drops the delta column for matrix-delta mode, where the matrix is drawn separately on a canvas
#====================================================================
#a placeholder cell for the forward columns at the last window, where no future window exists to decode against
def _blank_panel(ax, text="end of stream"):
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=8, color="#9a978f", transform=ax.transAxes)

#render a decode preview, current and decoded and true-next share a viridis scale per feature
#delta is a symmetric diverging map centred at zero, at stream end the forward columns blank
def render_decode_png(env, preview, names, include_delta=True):
    before = preview.before_image            #the current window image, always present
    delta = preview.delta_image              #none at stream end
    after = preview.after_image              #none at stream end
    target = preview.target_image            #none at stream end
    has_forward = delta is not None

    F = before.shape[2]
    cols = ["current", "delta", "decoded", "true next"] if include_delta else ["current", "decoded", "true next"]
    ncol = len(cols)
    fig = plt.figure(figsize=(ncol * 1.9, F * 1.7), dpi=100)

    for i in range(F):
        #shared viridis range across the value columns present for this feature
        viridis_panels = [before[:, :, i]]
        if has_forward:
            viridis_panels += [after[:, :, i], target[:, :, i]]
        lo = float(min(p.min() for p in viridis_panels))
        hi = float(max(p.max() for p in viridis_panels))
        dmax = (float(np.abs(delta[:, :, i]).max()) or 1e-9) if has_forward else 1.0

        for j, col in enumerate(cols):
            ax = fig.add_subplot(F, ncol, i * ncol + j + 1)
            if i == 0:
                ax.set_title(col, fontsize=9)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=8)

            m = None
            if col == "current":
                m = ax.imshow(before[:, :, i], origin="lower", cmap="viridis", vmin=lo, vmax=hi)
            elif not has_forward:
                _blank_panel(ax)
            elif col == "delta":
                m = ax.imshow(delta[:, :, i], origin="lower", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
            elif col == "decoded":
                m = ax.imshow(after[:, :, i], origin="lower", cmap="viridis", vmin=lo, vmax=hi)
            else:  #true next
                m = ax.imshow(target[:, :, i], origin="lower", cmap="viridis", vmin=lo, vmax=hi)

            ax.set_xticks([]); ax.set_yticks([])
            if m is not None:
                fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    fig.subplots_adjust(left=0.10, right=0.96, top=0.93, bottom=0.04, wspace=0.30, hspace=0.20)
    return _fig_to_url(fig)

#raw mode, the referenced eeg window beside the computed feature matrix, the features before any interpolation
#left is channels by time, right is channels by feature, the pre-image stack the decode and image paths both start from
def render_raw_computed(env, raw_window, stack, names):
    fig = plt.figure(figsize=(9, 4), dpi=100)

    ax1 = fig.add_subplot(1, 2, 1)
    m1 = ax1.imshow(raw_window, origin="lower", cmap="viridis", aspect="auto")
    ax1.set_title("raw window  channels x time", fontsize=9)
    ax1.set_xlabel("time", fontsize=8); ax1.set_ylabel("channel", fontsize=8)
    fig.colorbar(m1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 2, 2)
    m2 = ax2.imshow(stack, origin="lower", cmap="viridis", aspect="auto")
    ax2.set_title("computed features  channels x F", fontsize=9)
    ax2.set_ylabel("channel", fontsize=8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    fig.colorbar(m2, ax=ax2, fraction=0.046, pad=0.04)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.20, wspace=0.35)
    return _fig_to_url(fig)

#operator mode, the interpolation operator M as a coverage field, the summed absolute electrode spread over the grid
#window-independent, it changes only with margin and image resolution, so it shows margin's spatial reach directly
def render_operator(env):
    H, W = env.img_res
    coverage = np.abs(env.M).sum(axis=1).reshape(H, W)

    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    m = ax.imshow(coverage, origin="lower", cmap="magma")
    ax.set_title(f"operator M coverage  (margin {env.margin:.2f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _fig_to_url(fig)