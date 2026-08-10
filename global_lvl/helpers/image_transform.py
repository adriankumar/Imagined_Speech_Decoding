import numpy as np
#to-image transformation/interpolation tensor M (H*W, F), where F= number of toggled features

#returns M plus the pixel-space electrode positions and grid it was built on, so the mask can share them
def build_img_interpolation(electrode_pos_2d, img_res, margin=0.9):
    from scipy.interpolate import RBFInterpolator #laz imports
    H, W = img_res

    #centre on the projection vertex and scale so the furthest electrode sits inside the grid with a margin
    scale = (margin * min(H, W) / 2) / np.max(np.linalg.norm(electrode_pos_2d, axis=1))
    pix = electrode_pos_2d * scale + np.array([W / 2, H / 2])  #electrode positions in pixel coords, (x, y)

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid = np.column_stack([xs.ravel(), ys.ravel()])  #pixel centres as (x, y), row-major over (H, W)

    rbf = RBFInterpolator(pix, np.eye(len(electrode_pos_2d)), kernel="thin_plate_spline")
    M = rbf(grid)  #each column is the spread of one electrode, shape (H*W, n_channels)
    return M, pix, grid  #pix and grid are in M's pixel space, feed them straight to the mask


#topographic mask in M's pixel space, ignores img interpolation artefacts outside the electrodes
def build_topo_mask(electrode_pix_2d, grid_2d, img_dims, keep_scale=1.5, taper_scale=1.0):
    #electrode_pix_2d: n_chns x 2 and grid_2d: (h*w) x 2, both in M's pixel space and row order
    #distance from each pixel to its nearest simulated electrode
    d_pix = np.linalg.norm(grid_2d[:, None, :] - electrode_pix_2d[None, :, :], axis=-1).min(axis=1)

     #characteristic spacing = median nearest-neighbour electrode distance
    d_ee = np.linalg.norm(electrode_pix_2d[:, None, :] - electrode_pix_2d[None, :, :], axis=-1)
    np.fill_diagonal(d_ee, np.inf)
    spacing = float(np.median(d_ee.min(axis=1)))

    r_keep = keep_scale * spacing #full weight out to here past an electrode
    taper = taper_scale * spacing #smooth falloff width

    t = np.clip((d_pix - r_keep) / taper, 0.0, 1.0)
    mask = 0.5 * (1.0 + np.cos(np.pi * t))  #1 inside, smooth cos ramp to 0 outside
    mask = mask.reshape(*img_dims)  #h x w

    return mask

#zeroes an img outside the electrode region; works on a field, a sobel component,
#or any other map over the same grid, mask stays non-negative so signs survive
def apply_topo_mask(img, mask):
    assert img.shape[-3:-1] == mask.shape, f"img grid {img.shape[-3:-1]} does not match mask {mask.shape}"
    return img * mask[..., None] #(..., H, W, F)