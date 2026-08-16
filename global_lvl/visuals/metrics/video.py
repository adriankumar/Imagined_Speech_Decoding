import numpy as np
from ...constants import (DEFAULT_CMAP, DELTA_CMAP)
from .viewers import (_img_panel, _scatter_panel, DEFAULT_SAVE_PATH)
#===================================================================
# sequence panels written straight to gif
#===================================================================
#the figure is built once and only the data on each artist is swapped per frame,
#so layout, ticks, colourbars and axes extents are identical in every frame;
#nothing that could shift geometry is allowed inside the loop

#===================================================================
# shared
#===================================================================
#ranges come from the whole sequence so a frame cannot rescale itself, and are
#always taken from the arrays as they will be drawn, i.e after any masking
def _shared_range(*stacks):
    lo = min(float(s.min()) for s in stacks)
    hi = max(float(s.max()) for s in stacks)
    return (lo, hi) if hi > lo else (lo - 1.0, hi + 1.0)

#signed quantities get a symmetric range so zero sits at the centre of the map
def _symmetric_range(stack):
    a = float(np.abs(stack).max())
    return (-a, a) if a > 0 else (-1.0, 1.0)

#scale multiplies the data, so the colourbar is unreadable unless the factor is stated
def _scale_note(fig, scale):
    if scale == 1.0:
        return
    fig.text(0.99, 0.005, f"values scaled by {scale:g}", ha="right", va="bottom",
             fontsize=7, color="#9a978f")

#fig.text sits outside tight_layout, so a changing counter cannot move the panels;
#zero padded anyway so its width is constant
def _frame_label(fig, n_frames):
    pad = len(str(n_frames))
    label = fig.text(0.01, 0.005, "", ha="left", va="bottom", fontsize=7, color="#9a978f")
    return label, pad

#===================================================================
# writer
#===================================================================
#update(i) mutates the existing artists; the figure is never cleared or relaid out
def _write_gif(fig, update, n_frames, save_path, file_name, layout_rect=None, dpi=100, fps=10):
    import os
    import matplotlib.pyplot as plt #lazy import
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    #these figures are never shown, so agg is attached explicitly rather than
    #relying on whatever interactive backend happens to be active
    canvas = FigureCanvasAgg(fig)

    #dpi first, then layout once, both outside the loop
    fig.set_dpi(dpi)
    fig.tight_layout(rect=layout_rect)

    frames = []
    for i in range(n_frames):
        update(i)
        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba())[..., :3]
        img = Image.fromarray(rgb)

        #the first frame fixes the palette and every later frame is quantised onto it;
        #a per-frame adaptive palette would shimmer even with the ranges frozen
        frames.append(img.convert("P", palette=Image.ADAPTIVE) if not frames
                      else img.quantize(palette=frames[0]))

    plt.close(fig)

    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, file_name)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / fps)), loop=0, disposal=2)
    return out

#===================================================================
# reconstruction, one feature across a sequence
#===================================================================
#(N, H, W) draws as images, (N, n_chns) draws as electrode scatters and needs pos_2d
def reconstruction_gif(true_seq, recon_seq, feature_name, pos_2d=None, subtitle=None,
                       cmap=DEFAULT_CMAP, delta_cmap=DELTA_CMAP, scale=1.0,
                       save_path=DEFAULT_SAVE_PATH, file_name="reconstruction.gif",
                       dpi=100, fps=10):

    import matplotlib.pyplot as plt #lazy import

    true_seq = np.asarray(true_seq, dtype=float)
    recon_seq = np.asarray(recon_seq, dtype=float)

    assert true_seq.shape == recon_seq.shape, f"shape mismatch: {true_seq.shape} vs {recon_seq.shape}"

    as_img = true_seq.ndim == 3
    if not as_img:
        assert true_seq.ndim == 2, f"expected (N, H, W) or (N, n_chns), got {true_seq.shape}"
        assert pos_2d is not None, "electrode-space panels need the 2d electrode positions"

    N = true_seq.shape[0]
    assert N > 0, "empty sequence"

    #scale lands once, up front, so the ranges below are already in display units
    true_seq, recon_seq = true_seq * scale, recon_seq * scale
    diff_seq = true_seq - recon_seq #positive is under-constructed, negative is over

    #true and recon share a range so a poor reconstruction cannot read as a good one;
    #the difference gets its own symmetric range so its structure stays visible
    fr = _shared_range(true_seq, recon_seq)
    dr = _symmetric_range(diff_seq)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6), squeeze=False)
    ax = axes[0]

    panel = (lambda a, v, cm, lo, hi: _img_panel(a, v, cm, lo, hi)) if as_img else \
            (lambda a, v, cm, lo, hi: _scatter_panel(a, v, pos_2d, cm, lo, hi))

    handles = [panel(ax[0], true_seq[0], cmap, *fr),
               panel(ax[1], recon_seq[0], cmap, *fr),
               panel(ax[2], diff_seq[0], delta_cmap, *dr)]

    for a, title in zip(ax, ["true", "recon", "difference"]):
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel(feature_name, fontsize=10)

    fig.suptitle(subtitle or ("image-space" if as_img else "electrode-space"))
    _scale_note(fig, scale)
    label, pad = _frame_label(fig, N)

    set_values = (lambda m, v: m.set_data(v)) if as_img else (lambda m, v: m.set_array(v))

    def _update(i):
        set_values(handles[0], true_seq[i])
        set_values(handles[1], recon_seq[i])
        set_values(handles[2], diff_seq[i])
        label.set_text(f"window {i + 1:0{pad}d} / {N}")

    return _write_gif(fig, _update, N, save_path, file_name,
                      layout_rect=(0, 0.03, 1, 0.93), dpi=dpi, fps=fps)

#===================================================================
# sobel, one feature across a sequence
#===================================================================
#gradients first then mask, matching sobel_loss and avoiding a false edge at the
#boundary; true_seq and recon_seq must arrive unmasked
def _sobel_stacks(true_seq, recon_seq, mask):
    from ...metrics.image_space import sobel_stack #lazy import

    N = true_seq.shape[0]
    tgx, tgy, rgx, rgy = [], [], [], []

    #sobel_stack indexes its spatial axes as 0 and 1, so a leading sequence axis
    #would be read as height; each frame goes in on its own with a trailing F of 1
    for i in range(N):
        gx, gy = sobel_stack(img=true_seq[i][..., None], per_axis=True)
        tgx.append(gx[..., 0]); tgy.append(gy[..., 0])

        gx, gy = sobel_stack(img=recon_seq[i][..., None], per_axis=True)
        rgx.append(gx[..., 0]); rgy.append(gy[..., 0])

    stacks = [true_seq, recon_seq, np.stack(tgx), np.stack(tgy), np.stack(rgx), np.stack(rgy)]

    if mask is None:
        return stacks

    return [s * mask[None, ...] for s in stacks] #(N, H, W) against (1, H, W)

#3x3 for one feature: rows are field, dx, dy; columns are true, recon, difference
def sobel_gif(true_seq, recon_seq, feature_name, mask=None, subtitle=None,
              cmap=DEFAULT_CMAP, delta_cmap=DELTA_CMAP, scale=1.0,
              save_path=DEFAULT_SAVE_PATH, file_name="sobel.gif", dpi=100, fps=10):

    import matplotlib.pyplot as plt #lazy import
    from .sobel import SOBEL_NOTE

    true_seq = np.asarray(true_seq, dtype=float)
    recon_seq = np.asarray(recon_seq, dtype=float)

    assert true_seq.shape == recon_seq.shape, f"shape mismatch: {true_seq.shape} vs {recon_seq.shape}"
    assert true_seq.ndim == 3, f"expected (N, H, W), got {true_seq.shape}"

    N = true_seq.shape[0]
    assert N > 0, "empty sequence"

    #sobel is linear, so scaling the field up front is the same as scaling the gradients
    field_t, field_r, dx_t, dy_t, dx_r, dy_r = _sobel_stacks(true_seq * scale, recon_seq * scale, mask)

    field_d, dx_d, dy_d = field_t - field_r, dx_t - dx_r, dy_t - dy_r

    #field shares one range across true and recon; the difference gets its own
    f_range = _shared_range(field_t, field_r)
    fd_range = _symmetric_range(field_d)
    #both directions share one range so dx and dy stay comparable to each other
    g_range = _symmetric_range(np.stack([dx_t, dx_r, dy_t, dy_r]))
    #residual gradients are much smaller than the gradients, so they get their own
    gd_range = _symmetric_range(np.stack([dx_d, dy_d]))

    rows = [
        ("field", [(field_t, cmap, f_range), (field_r, cmap, f_range), (field_d, delta_cmap, fd_range)]),
        (r"$\partial x$", [(dx_t, delta_cmap, g_range), (dx_r, delta_cmap, g_range), (dx_d, delta_cmap, gd_range)]),
        (r"$\partial y$", [(dy_t, delta_cmap, g_range), (dy_r, delta_cmap, g_range), (dy_d, delta_cmap, gd_range)]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.5), squeeze=False)
    col_titles = ["true", "recon", "difference"]

    #each artist is paired with the stack it reads from, so the frame update is one loop
    pairs = []
    for r, (row_label, cells) in enumerate(rows):
        for c, (stack, cm, (vmin, vmax)) in enumerate(cells):
            ax = axes[r][c]
            pairs.append((_img_panel(ax, stack[0], cm, vmin, vmax), stack))
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=10)

    fig.suptitle(f"{subtitle or 'sobel'} · {feature_name}")

    #the scale factor joins the kernel note rather than getting its own corner
    note = SOBEL_NOTE if scale == 1.0 else f"{SOBEL_NOTE}, values scaled by {scale:g}"
    fig.text(0.99, 0.005, note, ha="right", va="bottom", fontsize=7, color="#9a978f")
    label, pad = _frame_label(fig, N)

    def _update(i):
        for m, stack in pairs:
            m.set_data(stack[i])
        label.set_text(f"window {i + 1:0{pad}d} / {N}")

    return _write_gif(fig, _update, N, save_path, file_name,
                      layout_rect=(0, 0.03, 1, 0.96), dpi=dpi, fps=fps)

#===================================================================
# metric bars, all features across a sequence
#===================================================================
#values_seq is (N, F), one scalar per feature per window; ylim is frozen over the
#whole sequence and always includes zero so signed metrics keep a fixed baseline
def metric_bar_gif(values_seq, feature_names, metric_name="metric", subtitle=None,
                   colour="#4a4843", scale=1.0, save_path=DEFAULT_SAVE_PATH,
                   file_name="metric.gif", dpi=100, fps=10):

    import matplotlib.pyplot as plt #lazy import

    values_seq = np.asarray(values_seq, dtype=float) * scale

    assert values_seq.ndim == 2, f"expected (N, F), got {values_seq.shape}"

    N, F = values_seq.shape
    assert N > 0, "empty sequence"
    assert F == len(feature_names), f"{F} values against {len(feature_names)} names"

    lo, hi = min(0.0, float(values_seq.min())), max(0.0, float(values_seq.max()))
    span = hi - lo
    pad_y = 0.05 * span if span > 0 else 1.0

    fig, ax = plt.subplots(figsize=(1.8 + 0.7 * F, 3.2))
    bars = ax.bar(range(F), values_seq[0], color=colour)

    ax.set_xticks(range(F))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.axhline(0.0, color="#d8d4cd", linewidth=0.8) #visible baseline for signed metrics
    ax.set_ylim(lo - pad_y, hi + pad_y)
    ax.set_title(subtitle or metric_name, fontsize=10)

    _scale_note(fig, scale)
    label, pad = _frame_label(fig, N)

    def _update(i):
        for rect, v in zip(bars, values_seq[i]):
            rect.set_height(v)
        label.set_text(f"window {i + 1:0{pad}d} / {N}")

    return _write_gif(fig, _update, N, save_path, file_name,
                      layout_rect=(0, 0.03, 1, 1), dpi=dpi, fps=fps)