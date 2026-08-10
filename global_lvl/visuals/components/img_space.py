import numpy as np

#electrode coverage of the interpolation operator M, per-pixel summed weight
def img_transform_fig(M, img_dims, subtitle=None):
    import matplotlib.pyplot as plt  #lazy import

    H, W = img_dims
    coverage = np.abs(M).sum(axis=1).reshape(H, W)  #abs, not signed — signed is ~1 everywhere
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    m = ax.imshow(coverage, origin="lower", cmap="nipy_spectral_r")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(subtitle or "Electrode-to-Image Transform", fontsize=10)
    fig.tight_layout()
    return fig