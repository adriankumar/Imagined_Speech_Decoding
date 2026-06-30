import io
import base64
import numpy as np

import matplotlib
matplotlib.use("Agg")  #headless backend, the gui never shows a window
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

#====================================================================
#dark theme, matched to the gui's app.css so every render sits flush in its panel
#====================================================================
BG = "#1a1a1c"          #figure background, the page colour
PANEL = "#232327"       #axes facecolor, the panel colour
TXT = "#ecebef"         #ticks and labels
DIM = "#9a96a2"         #blank-panel placeholder text
TITLE_C = "#c77dff"     #luminous magenta title
SPINE = "#3a3a40"
SEQ = "magma"           #sequential, black -> purple -> orange, every value heatmap and the strip
DIV = "RdBu_r"          #diverging, blue/red around zero, the signed delta column

#encode a figure to a base64 png on the dark background and close it, the single exit every renderer funnels through
def _fig_to_url(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

#dark axis styling, panel facecolor with light ticks and dim spines
def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    for s in ax.spines.values():
        s.set_color(SPINE)

#a luminous magenta title with a soft glow halo against the dark background
def _glow_title(ax, text, fontsize=9):
    t = ax.set_title(text, fontsize=fontsize, color=TITLE_C)
    t.set_path_effects([pe.withStroke(linewidth=3, foreground=TITLE_C, alpha=0.30)])

#light colourbar ticks and a dim outline
def _style_cbar(cb):
    cb.ax.tick_params(colors=TXT, labelsize=7)
    cb.outline.set_edgecolor(SPINE)

#====================================================================
#feature representation, image is the interpolated field, stack is the electrode scatter
#figsize sets the display size now that the css cap no longer upscales, adjust the multipliers to taste
#====================================================================
#draw an (array, names) pair to a base64 png, one panel per feature, each panel self-normalises
def _render_array(env, array, names, kind):
    F = array.shape[2] if kind == "image" else array.shape[1]
    cols = min(F, 3)
    rows = int(np.ceil(F / cols))
    fig = plt.figure(figsize=(cols * 2.6, rows * 2.4), dpi=100, facecolor=BG)
    pos_2d = env.electrode_pos_2d if kind == "stack" else None

    for i in range(F):
        ax = fig.add_subplot(rows, cols, i + 1)
        if kind == "image":
            m = ax.imshow(array[:, :, i], origin="lower", cmap=SEQ)
        else:
            m = ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c=array[:, i], cmap=SEQ, s=30)
            ax.set_aspect("equal")
        _glow_title(ax, names[i])
        ax.set_xticks([]); ax.set_yticks([])
        _style_ax(ax)
        _style_cbar(fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04))

    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.04, wspace=0.25, hspace=0.30)
    return _fig_to_url(fig)

#render one feature stack, image kind interpolates through M, stack kind scatters at the electrodes
def render_stack_image(env, stack, names, kind):
    array = env.to_image(stack) if kind == "image" else stack
    return _render_array(env, array, names, kind)

#====================================================================
#recording strip, a channels-by-time raster of an arbitrary span of the timeline
#====================================================================
#compressed raster over [start, stop), each column one time bin, each cell the per-channel mean absolute amplitude
def build_strip_png(env, start, stop, width=1000):
    start = max(0, int(start))
    stop = min(int(stop), env.timepoints)
    span = max(1, stop - start)
    width = int(min(width, span))
    edges = np.linspace(start, stop, width + 1, dtype=int)
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
    plt.imsave(buf, raster, cmap=SEQ, format="png")  #magma to match the theme, normalised to the raster's own range
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii"), n_ch, width

#====================================================================
#raw mode, the referenced eeg window beside the computed feature matrix, the features before interpolation
#====================================================================
def render_raw_computed(env, raw_window, stack, names):
    fig = plt.figure(figsize=(8.0, 3.6), dpi=100, facecolor=BG)

    ax1 = fig.add_subplot(1, 2, 1)
    m1 = ax1.imshow(raw_window, origin="lower", cmap=SEQ, aspect="auto")
    _glow_title(ax1, "raw window  channels x time")
    ax1.set_xlabel("time", fontsize=8); ax1.set_ylabel("channel", fontsize=8)
    _style_ax(ax1)
    _style_cbar(fig.colorbar(m1, ax=ax1, fraction=0.046, pad=0.04))

    ax2 = fig.add_subplot(1, 2, 2)
    m2 = ax2.imshow(stack, origin="lower", cmap=SEQ, aspect="auto")
    _glow_title(ax2, "computed features  channels x F")
    ax2.set_ylabel("channel", fontsize=8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=TXT)
    _style_ax(ax2)
    _style_cbar(fig.colorbar(m2, ax=ax2, fraction=0.046, pad=0.04))

    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.22, wspace=0.35)
    return _fig_to_url(fig)

#====================================================================
#operator mode, the interpolation operator M as a coverage field, window-independent, shows margin's reach
#====================================================================
def render_operator(env):
    H, W = env.img_res
    coverage = np.abs(env.M).sum(axis=1).reshape(H, W)

    fig = plt.figure(figsize=(3.4, 3.0), dpi=100, facecolor=BG)
    ax = fig.add_subplot(1, 1, 1)
    m = ax.imshow(coverage, origin="lower", cmap=SEQ)
    _glow_title(ax, f"operator M coverage  (margin {env.margin:.2f})")
    ax.set_xticks([]); ax.set_yticks([])
    _style_ax(ax)
    _style_cbar(fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04))
    fig.tight_layout()
    return _fig_to_url(fig)

#====================================================================
#decode simulation, current, delta, decoded, true next, one row per feature
#value columns are sequential magma, delta is diverging around zero, forward columns blank at stream end
#====================================================================
def _blank_panel(ax, text="end of stream"):
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=8, color=DIM, transform=ax.transAxes)

def render_decode_png(env, preview, names, include_delta=True):
    before = preview.before_image            #the current window image, always present
    delta = preview.delta_image              #none at stream end
    after = preview.after_image              #none at stream end
    target = preview.target_image            #none at stream end
    has_forward = delta is not None

    F = before.shape[2]
    cols = ["current", "delta", "decoded", "true next"] if include_delta else ["current", "decoded", "true next"]
    ncol = len(cols)
    fig = plt.figure(figsize=(ncol * 1.9, F * 1.4), dpi=100, facecolor=BG)

    for i in range(F):
        viridis_panels = [before[:, :, i]]
        if has_forward:
            viridis_panels += [after[:, :, i], target[:, :, i]]
        lo = float(min(p.min() for p in viridis_panels))
        hi = float(max(p.max() for p in viridis_panels))
        dmax = (float(np.abs(delta[:, :, i]).max()) or 1e-9) if has_forward else 1.0

        for j, col in enumerate(cols):
            ax = fig.add_subplot(F, ncol, i * ncol + j + 1)
            if i == 0:
                _glow_title(ax, col)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=8)

            m = None
            if col == "current":
                m = ax.imshow(before[:, :, i], origin="lower", cmap=SEQ, vmin=lo, vmax=hi)
            elif not has_forward:
                _blank_panel(ax)
            elif col == "delta":
                m = ax.imshow(delta[:, :, i], origin="lower", cmap=DIV, vmin=-dmax, vmax=dmax)
            elif col == "decoded":
                m = ax.imshow(after[:, :, i], origin="lower", cmap=SEQ, vmin=lo, vmax=hi)
            else:  #true next
                m = ax.imshow(target[:, :, i], origin="lower", cmap=SEQ, vmin=lo, vmax=hi)

            ax.set_xticks([]); ax.set_yticks([])
            _style_ax(ax)
            if m is not None:
                _style_cbar(fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04))

    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.04, wspace=0.30, hspace=0.20)
    return _fig_to_url(fig)