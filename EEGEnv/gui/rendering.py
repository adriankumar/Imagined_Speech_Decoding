import io
import base64
import numpy as np

import matplotlib
matplotlib.use("Agg")  #headless backend, the gui never shows a window
import matplotlib.pyplot as plt

#draw an (array, names) pair to a base64 png, one panel per feature, each panel self-normalises
#kind 'image' draws the interpolated field, 'stack' scatters raw values at the electrode positions
def _render_array(env, array, names, kind):
    F = array.shape[2] if kind == "image" else array.shape[1]
    cols = min(F, 3)
    rows = int(np.ceil(F / cols))
    #fixed figure size keeps every render the same pixel dimensions, no relayout jitter across windows
    fig = plt.figure(figsize=(cols * 3.0, rows * 2.8), dpi=100)
    pos_2d = env.SH_dict["pos_2d"] if kind == "stack" else None

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

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

#render one feature stack to the feature image, image kind interpolates through M, stack kind scatters
def render_stack_image(env, stack, names, kind):
    array = env.encode_feature_image(stack) if kind == "image" else stack
    return _render_array(env, array, names, kind)

#build a compressed channels-by-time raster of the whole recording and render it to a viridis png
#each column is one time bin, each cell the per-channel mean absolute amplitude over that bin
#read in bounded chunks so the full recording never sits in memory, computed once per load
def build_strip_png(env, width=1000):
    n_time = env.get_timepoints
    width = int(min(width, n_time))
    edges = np.linspace(0, n_time, width + 1, dtype=int)
    n_ch = env.get_n_chns
    raster = np.empty((n_ch, width), dtype=np.float32)

    budget = 100000  #samples per disk read, caps the transient memory of each chunk
    c0 = 0
    while c0 < width:
        bin_span = max(1, int(edges[c0 + 1] - edges[c0]))
        c1 = min(c0 + max(1, budget // bin_span), width)
        s0, s1 = int(edges[c0]), int(edges[c1])
        data = np.abs(env.get_referenced_window(s0, s1))
        for j in range(c0, c1):
            a, b = int(edges[j]) - s0, int(edges[j + 1]) - s0
            raster[:, j] = data[:, a:b].mean(axis=1) if b > a else 0.0
        c0 = c1

    buf = io.BytesIO()
    plt.imsave(buf, raster, cmap="viridis", format="png")  #imsave normalises to the raster's own range
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii"), n_ch, width


#render the decode simulation as before, delta, after columns, one row per feature
#before and after share a colour scale per feature, delta is a symmetric diverging map centred at zero
def render_decode_png(env, before, delta, after, names):
    F = before.shape[2]
    fig = plt.figure(figsize=(3 * 2.7, F * 2.5), dpi=100)
    titles = ["before", "delta", "after"]
    for i in range(F):
        lo = float(min(before[:, :, i].min(), after[:, :, i].min()))
        hi = float(max(before[:, :, i].max(), after[:, :, i].max()))
        dmax = float(np.abs(delta[:, :, i]).max()) or 1e-9
        for j, img in enumerate([before, delta, after]):
            ax = fig.add_subplot(F, 3, i * 3 + j + 1)
            if j == 1:
                m = ax.imshow(img[:, :, i], origin="lower", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
            else:
                m = ax.imshow(img[:, :, i], origin="lower", cmap="viridis", vmin=lo, vmax=hi)
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.03, wspace=0.30, hspace=0.20)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")