import numpy as np
from ...constants import (DEFAULT_CMAP, DELTA_CMAP)
from .viewers import _img_panel
#===================================================================
# sobel panels, one feature at a time, cycled with the arrow keys
#===================================================================
#note kept on the figure since scipy's sobel kernel is unnormalised
SOBEL_NOTE = "sobel kernel unnormalised, values are 8x the discrete derivative"

#gradients first then mask, matching sobel_loss and avoiding a false edge at the boundary
def _sobel_arrays(true_img, recon_img, mask):
    from ...metrics.image_space import sobel_stack #lazy import, adjust to the actual module

    tgx, tgy = sobel_stack(true_img, per_axis=True)   #each (H, W, F)
    rgx, rgy = sobel_stack(recon_img, per_axis=True)

    if mask is None:
        return true_img, recon_img, tgx, tgy, rgx, rgy

    m = mask[..., None] #(H, W, 1)
    return true_img * m, recon_img * m, tgx * m, tgy * m, rgx * m, rgy * m

#one symmetric range shared across a group, so weaker panels read as weaker
def _shared_symmetric(arrays):
    a = max(np.abs(x).max() for x in arrays)
    return (-a, a) if a > 0 else (-1.0, 1.0)

#fills a 3x3 grid for one feature: rows are field, dx, dy; columns are true, recon, difference
def _fill_sobel_axes(axes, f, arrays, cmap, delta_cmap, scale):
    true_img, recon_img, tgx, tgy, rgx, rgy = arrays

    field_t, field_r = true_img[..., f] * scale, recon_img[..., f] * scale
    dx_t, dx_r = tgx[..., f] * scale, rgx[..., f] * scale
    dy_t, dy_r = tgy[..., f] * scale, rgy[..., f] * scale

    field_d, dx_d, dy_d = field_t - field_r, dx_t - dx_r, dy_t - dy_r

    #field shares one range across true and recon; the difference gets its own
    f_lo, f_hi = min(field_t.min(), field_r.min()), max(field_t.max(), field_r.max())
    #both directions share one range so dx and dy stay comparable to each other
    g_range = _shared_symmetric([dx_t, dx_r, dy_t, dy_r])
    #residual gradients are much smaller than the gradients, so they get their own
    gd_range = _shared_symmetric([dx_d, dy_d])
    fd_range = _shared_symmetric([field_d])

    rows = [
        ("field", [(field_t, cmap, (f_lo, f_hi)), (field_r, cmap, (f_lo, f_hi)), (field_d, delta_cmap, fd_range)]),
        (r"$\partial x$", [(dx_t, delta_cmap, g_range), (dx_r, delta_cmap, g_range), (dx_d, delta_cmap, gd_range)]),
        (r"$\partial y$", [(dy_t, delta_cmap, g_range), (dy_r, delta_cmap, g_range), (dy_d, delta_cmap, gd_range)]),
    ]

    col_titles = ["true", "recon", "difference"]
    for r, (row_label, cells) in enumerate(rows):
        for c, (data, cm, (vmin, vmax)) in enumerate(cells):
            ax = axes[r][c]
            _img_panel(ax, data, cm, vmin, vmax)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=10)

#static 3x3 for a single feature, used when writing files
def sobel_feature_fig(true_img, recon_img, feature_names, feature_index, mask=None, subtitle=None,
                      cmap=DEFAULT_CMAP, delta_cmap=DELTA_CMAP, scale=1.0):
    
    import matplotlib.pyplot as plt  #lazy import

    arrays = _sobel_arrays(true_img, recon_img, mask)

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.5), squeeze=False)
    _fill_sobel_axes(axes, feature_index, arrays, cmap, delta_cmap, scale)

    fig.suptitle(f"{subtitle or 'sobel'} · {feature_names[feature_index]}")
    fig.text(0.99, 0.005, SOBEL_NOTE, ha="right", va="bottom", fontsize=7, color="#9a978f")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig

#interactive 3x3 cycling one feature at a time; left/right step through features
def sobel_cycle_fig(true_img, recon_img, feature_names, mask=None, subtitle=None,
                    cmap=DEFAULT_CMAP, delta_cmap=DELTA_CMAP, scale=1.0):
    
    import matplotlib.pyplot as plt  #lazy import

    assert true_img.shape == recon_img.shape, f"shape mismatch: {true_img.shape} vs {recon_img.shape}"
    assert true_img.ndim == 3, f"expected (H, W, F), got {true_img.shape}"

    F = true_img.shape[-1]
    assert F == len(feature_names), f"{F} features against {len(feature_names)} names"

    arrays = _sobel_arrays(true_img, recon_img, mask)
    state = {"f": 0}

    fig = plt.figure(figsize=(10.5, 9.5))

    #the whole figure is cleared each frame; colourbars are their own axes and
    #would otherwise accumulate one per redraw
    def _draw():
        f = state["f"]
        fig.clf()
        axes = fig.subplots(3, 3, squeeze=False)
        _fill_sobel_axes(axes, f, arrays, cmap, delta_cmap, scale)

        fig.suptitle(f"{subtitle or 'sobel'} · {feature_names[f]}  ({f + 1}/{F})  ←  → to cycle")
        fig.text(0.99, 0.005, SOBEL_NOTE, ha="right", va="bottom", fontsize=7, color="#9a978f")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.canvas.draw_idle()

    def _on_key(event):
        if event.key in ("right", "down"):
            state["f"] = (state["f"] + 1) % F
        elif event.key in ("left", "up"):
            state["f"] = (state["f"] - 1) % F
        else:
            return
        _draw()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _draw()
    return fig