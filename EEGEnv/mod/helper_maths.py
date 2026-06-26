import numpy as np
from math import isqrt, factorial
from scipy.interpolate import RBFInterpolator
from scipy.special import lpmv

#===================================================================
# Spherical Harmonics stuff
#===================================================================
#largest harmonic order the channel count can resolve, the upper bound for the L control
def l_max(n_channels):
    return isqrt(n_channels) - 1

def compute_sphere_params(pos_3d):
    #algebraic least-squares, solve |p|^2 = 2c.p + (r^2 - |c|^2) for centre and the offset term\
    f = np.sum(pos_3d**2, axis=1)
    A = np.column_stack([2 * pos_3d, np.ones(len(pos_3d))])
    sol, *_ = np.linalg.lstsq(A, f, rcond=None)
    centre = sol[:3]
    radius = np.sqrt(sol[3] + centre @ centre)

    return centre, radius #radius does not affect basis

def compute_spherical_angles(pos_3d, centre):
    v = pos_3d - centre
    r = np.linalg.norm(v, axis=1)
    theta = np.arccos(v[:, 2] / r)  #polar angle
    phi = np.arctan2(v[:, 1], v[:, 0])  #azimuth
    return theta, phi

#azimuthal equidistant projection to a flat disk, polar angle becomes radial distance, azimuth becomes the disk angle
def azimuthal_2d(theta, phi):
    x = theta * np.cos(phi)
    y = theta * np.sin(phi)
    return np.column_stack([x, y])

#precomputed interpolation operator mapping per-channel values onto the fixed pixel grid, image = M @ features
def build_interpolation_operator(pos2d, img_res, margin=0.9):
    H, W = img_res
    #centre on the projection vertex and scale so the furthest electrode sits inside the grid with a margin
    scale = (margin * min(H, W) / 2) / np.max(np.linalg.norm(pos2d, axis=1))
    pix = pos2d * scale + np.array([W / 2, H / 2])  #electrode positions in pixel coords, (x, y)

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid = np.column_stack([xs.ravel(), ys.ravel()])  #pixel centres as (x, y), row-major over (H, W)

    rbf = RBFInterpolator(pix, np.eye(len(pos2d)), kernel="thin_plate_spline")
    return rbf(grid)  #each column is the spread of one electrode, shape (H*W, n_channels)

#real orthonormal spherical harmonic basis evaluated at each electrode, shape ((L+1)^2, n_channels)
def build_sh_basis(theta, phi, L):
    rows = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            #orthonormal real form, sqrt(2) on the off-axis modes, legendre carries the consistent sign
            norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
            legendre = lpmv(abs(m), l, np.cos(theta))
            if m > 0:
                row = np.sqrt(2) * norm * legendre * np.cos(m * phi)
            elif m < 0:
                row = np.sqrt(2) * norm * legendre * np.sin(abs(m) * phi)
            else:
                row = norm * legendre
            rows.append(row)
    return np.array(rows)  #row order is l=0..L, m=-l..l, the fixed mode order the model emits


#least squares spherical-harmonic coefficients of a per-channel block b against basis Y
#Y is ((L+1)^2, n_channels), b is (n_channels, F), returns coefficients ((L+1)^2, F)
#the regulariser keeps the gram solve stable when electrodes sample the modes poorly
def solve_sh_coefficients(Y, b, lam=1e-3):
    gram = Y @ Y.T
    gram[np.diag_indices_from(gram)] += lam
    return np.linalg.solve(gram, Y @ b)

#synthesise a per-channel block from coefficients through the load-bearing path Y^T c
#c is ((L+1)^2, F), returns (n_channels, F)
def synthesise_sh(Y, c):
    return Y.T @ c
#===================================================================
# Re-reference EEG voltage
#===================================================================
#deterministic re-reference of the data array, skipped when the source already matches the target
def re_reference(data, target_ref, ref_scheme, channel_names):
    if isinstance(target_ref, str) and ref_scheme.lower() == target_ref.lower():
        return data  #source already in the target reference, trust the assertion and do nothing
    if target_ref == "average":
        return data - data.mean(axis=0, keepdims=True)  #common average reference
    
    #channel-list reference, subtract the mean of the named channels
    keys = [c.lower() for c in channel_names]
    idx = [keys.index(c.lower()) for c in target_ref]
    
    return data - data[idx].mean(axis=0, keepdims=True)

#===================================================================
# Temporal Feature stack
#===================================================================
#per-channel variance along time, zero when the window is too short to form a difference
def time_variance(x):
    if x.shape[1] == 0:
        return np.zeros(x.shape[0])
    return np.var(x, axis=1)

#per-channel mean over the window, the raw activation level the other features are computed from
def feature_mean(window):
    return np.mean(window, axis=1)

#per-channel median over the window, robust central tendency
def feature_median(window):
    return np.median(window, axis=1)

#per-channel interquartile range over the window, robust scale
def feature_iqr(window):
    q75, q25 = np.percentile(window, [75, 25], axis=1)
    return q75 - q25

#hjorth mobility, time-domain proxy for mean frequency, zero where the signal does not vary
def hjorth_mobility(window):
    var_x = time_variance(window)
    var_dx = time_variance(np.diff(window, axis=1))
    return np.sqrt(np.divide(var_dx, var_x, out=np.zeros_like(var_x), where=var_x > 0))

#hjorth complexity, time-domain proxy for bandwidth, zero where the first derivative does not vary
def hjorth_complexity(window):
    dx = np.diff(window, axis=1)
    mob_x = hjorth_mobility(window)
    mob_dx = hjorth_mobility(dx)
    return np.divide(mob_dx, mob_x, out=np.zeros_like(mob_x), where=mob_x > 0)

#===================================================================
# Temporal accumulation
#===================================================================
#single ema step, initialises to current on the first call when prior is none, always returns a finite array
def ema_update(current, prior, alpha):
    if prior is None:
        return current.copy()  #cold start: no transient pull toward zero
    return alpha * current + (1.0 - alpha) * prior

#project the feature stack onto the harmonic basis, coefficients shape ((L+1)^2, F)
#Y is the basis matrix of shape ((L+1)^2, n_channels), stack is (n_channels, F)
def project_sh_coefficients(Y, stack):
    return Y @ stack  #linear projection, row i is mode i across all active features

#reconstruct electrode-space values from harmonic coefficients via the moore-penrose pseudoinverse of Y
#output shape matches stack: (n_channels, F), lossy only for modes beyond the resolved L
def reconstruct_from_sh(Y, coefficients):
    return np.linalg.pinv(Y) @ coefficients

#convert a duration in seconds to whole samples at the given rate, for intuitive window and span sizing
def seconds_to_samples(seconds, sfreq):
    return int(round(seconds * sfreq))

#convert a sample count back to seconds, for readable reporting
def samples_to_seconds(samples, sfreq):
    return samples / sfreq

#robust magnitude of an absolute-value array for deterministic feature scaling
def robust_magnitude(values, method="median"):
    if method == "median":
        return float(np.median(values))
    if method == "percentile":
        return float(np.percentile(values, 95))
    if method == "iqr":
        return float(np.subtract(*np.percentile(values, [75, 25])))
    if method == "max":
        return float(np.max(values))
    raise ValueError(f"Unknown calibration method: {method}; use median, percentile, iqr or max")
#===================================================================
# Image transform
#===================================================================
#encode a per-channel feature stack into the fixed image grid, image = M @ features reshaped to (H, W, F)
def encode_image(M, features, img_res):
    H, W = img_res
    return (M @ features).reshape(H, W, -1) #-1 allows F to be arbitrary depending on whats toggled