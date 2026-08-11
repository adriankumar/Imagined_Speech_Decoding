import numpy as np
from ...constants import (DEFAULT_CMAP, DELTA_CMAP)

DEFAULT_SAVE_PATH = "figures"

#===================================================================
# shared
#===================================================================
#writes a built figure to disk before it is shown, since closing the window
#can destroy the figure on some backends
def save_fig(fig, save_path=DEFAULT_SAVE_PATH, file_name="figure.png", dpi=150):
    import os
    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, file_name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out

#display-only rescale; scale is a constant factor, norm divides by the peak
def _prepare(values, scale, norm):
    assert not (norm and scale != 1.0), "scale and norm both rescale, pick one"
    if norm:
        a = np.abs(values).max()
        return values / a if a > 0 else values
    return values * scale

#===================================================================
# metric bars, one metric across features
#===================================================================
#values is (F,), one scalar per feature
def metric_bar_fig(values, feature_names, metric_name="metric", subtitle=None,
                   colour="#4a4843", scale=1.0, norm=False):
    
    import matplotlib.pyplot as plt  #lazy import

    values = np.asarray(values, dtype=float) #(F,)
    assert values.ndim == 1, f"expected one scalar per feature, got shape {values.shape}"
    assert len(values) == len(feature_names), f"{len(values)} values against {len(feature_names)} names"

    #norm here is across features, since each feature is already a single number
    plotted = _prepare(values, scale, norm)

    fig, ax = plt.subplots(figsize=(1.8 + 0.7 * len(feature_names), 3.2))
    ax.bar(range(len(feature_names)), plotted, color=colour)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.axhline(0.0, color="#d8d4cd", linewidth=0.8) #visible baseline for signed metrics

    if norm:
        ax.set_ylim(0.0 if plotted.min() >= 0 else -1.0, 1.0)

    ax.set_title(subtitle or metric_name, fontsize=10)
    fig.tight_layout()
    return fig

#===================================================================
# reconstruction panels, features down the rows, true/recon/difference across
#===================================================================
#one field as an image cell, anterior up via origin lower
def _img_panel(ax, field_2d, cmap, vmin, vmax):
    m = ax.imshow(field_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.figure.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

#one field as an electrode scatter at the 2d projection
def _scatter_panel(ax, values, pos_2d, cmap, vmin, vmax):
    m = ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=30)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.figure.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

#true and recon share a range so a poor reconstruction cannot read as a good one;
#the difference gets its own symmetric range so its structure stays visible
def _feature_panels(true_f, recon_f, scale, norm):
    diff_f = true_f - recon_f #positive is under-constructed, negative is over

    if norm:
        s = max(np.abs(true_f).max(), np.abs(recon_f).max())
        d = np.abs(diff_f).max()
        true_f = true_f / s if s > 0 else true_f
        recon_f = recon_f / s if s > 0 else recon_f
        diff_f = diff_f / d if d > 0 else diff_f

        lo = 0.0 if min(true_f.min(), recon_f.min()) >= 0 else -1.0
        return [true_f, recon_f, diff_f], [(lo, 1.0), (lo, 1.0), (-1.0, 1.0)]

    true_f, recon_f, diff_f = true_f * scale, recon_f * scale, diff_f * scale
    lo = min(true_f.min(), recon_f.min())
    hi = max(true_f.max(), recon_f.max())
    a = np.abs(diff_f).max()
    return [true_f, recon_f, diff_f], [(lo, hi), (lo, hi), (-a, a)]

#(H, W, F) draws as images, (n_chns, F) draws as electrode scatters and needs pos_2d
def reconstruction_fig(true_field, recon_field, feature_names, pos_2d=None, subtitle=None,
                       cmap=DEFAULT_CMAP, delta_cmap=DELTA_CMAP, scale=1.0, norm=False):
    import matplotlib.pyplot as plt  #lazy import

    assert true_field.shape == recon_field.shape, f"shape mismatch: {true_field.shape} vs {recon_field.shape}"
    assert not (norm and scale != 1.0), "scale and norm both rescale, pick one"

    as_img = true_field.ndim == 3
    if not as_img:
        assert true_field.ndim == 2, f"expected (H, W, F) or (n_chns, F), got {true_field.shape}"
        assert pos_2d is not None, "electrode-space panels need the 2d electrode positions"

    F = true_field.shape[-1]
    assert F == len(feature_names), f"{F} features against {len(feature_names)} names"

    col_titles = ["true", "recon", "difference"]
    cmaps = [cmap, cmap, delta_cmap]

    fig, axes = plt.subplots(F, 3, figsize=(9.5, F * 2.8), squeeze=False)
    for r in range(F):
        datas, ranges = _feature_panels(true_field[..., r], recon_field[..., r], scale, norm)

        for c, (data, cm, (vmin, vmax)) in enumerate(zip(datas, cmaps, ranges)):
            ax = axes[r][c]
            if as_img:
                _img_panel(ax, data, cm, vmin, vmax)
            else:
                _scatter_panel(ax, data, pos_2d, cm, vmin, vmax)

            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(feature_names[r], fontsize=10)

    fig.suptitle(subtitle or ("image-space" if as_img else "electrode-space"))
    fig.tight_layout()
    return fig