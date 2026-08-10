import numpy as np 
from ...constants import (DEFAULT_CMAP, DELTA_CMAP)

#returns figure
def basis_matrix_fig(Y, subtitle=None):
    import matplotlib.pyplot as plt  #lazy import

    a = np.abs(Y).max()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    m = ax.imshow(Y, origin="upper", cmap=DELTA_CMAP, vmin=-a, vmax=a, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    rows, cols = Y.shape

    ax.set_title(subtitle or f"Y  ({rows} modes x {cols} electrodes)", fontsize=10)
    fig.tight_layout()
    return fig

