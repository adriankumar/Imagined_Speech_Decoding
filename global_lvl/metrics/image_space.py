import numpy as np
from ..helpers import apply_topo_mask
#------------
#image space, inputs are H x W, xF feature images
#mask is an optional H x W weight map, pass it to score only inside the electrode area
#------------
#mean over pixels, weighted by mask if given, keeps the feature axis
def _pixel_mean(err, mask):
    if mask is None:
        return err.mean(axis=(0, 1)) #(F,)

    return apply_topo_mask(img=err, mask=mask).sum(axis=(0, 1)) / mask.sum() #(F,)

#pixel-wise reconstruction error over the image, per feature; loss
def pixel_loss(true_img, recon_img, mask=None):
    sq = (true_img - recon_img) ** 2  #H x W x F; sq square diff
    return _pixel_mean(sq, mask) #(F,)

#discrete image derivatives,
#note scipiy's values are scaled by a constant factor of 8, so ensure
#when reporting these metrics, to either scale down or note this
#public for image vis
def sobel_stack(img, per_axis=True):
    from scipy.ndimage import sobel  #lazy import
    gx = np.stack([sobel(img[..., f], axis=1, mode="reflect") for f in range(img.shape[-1])], axis=-1)  #(H, W, F) horizontal
    gy = np.stack([sobel(img[..., f], axis=0, mode="reflect") for f in range(img.shape[-1])], axis=-1)  #(H, W, F) vertical

    if per_axis: #main return for model loss
        return gx, gy #both H x W x F, derivative horizontal and vertical

    return np.hypot(gx, gy)  #(H, W, F), non-differentiable at zero, vis only

#error between the true and recon sobel components, per feature; loss
#low-L sh is smooth, so this exposes the sharp spatial detail it drops in img space for the model
def sobel_loss(true_img, recon_img, mask=None):
    true_gx, true_gy = sobel_stack(true_img)
    recon_gx, recon_gy = sobel_stack(recon_img)

    #mask enters as a weight in _pixel_mean, so it lands after the operator
    return (_pixel_mean((true_gx - recon_gx) ** 2, mask)
            + _pixel_mean((true_gy - recon_gy) ** 2, mask))  #(F,)