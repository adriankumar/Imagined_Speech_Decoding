import numpy as np
# from math import factorial 
from scipy.special import lpmv, sph_harm
from scipy.interpolate import RBFInterpolator

#----------------
#Spherical Harmonics: Y^ℓ_m(θ,ϕ)=N^ℓ_m P^ℓ_m(cosθ) e^(imϕ)
#----------------
def _fit_coords_to_a_sphere(pos_3d):
    #algebraic least-squares, solve |p|^2 = 2c.p + (r^2 - |c|^2) 
    #for centre and the offset term\
    f = np.sum(pos_3d**2, axis=1)
    A = np.column_stack([2 * pos_3d, np.ones(len(pos_3d))])
    sol, *_ = np.linalg.lstsq(A, f, rcond=None)
    centre = sol[:3]
    radius = np.sqrt(sol[3] + centre @ centre)

    return centre, radius #radius does not affect basis

def compute_spherical_angles(pos_3d):
    #approximate sphere radius and centre from 3d positions by using least-squares to fit
    #electrode positions onto a sphere
    centre, radius = _fit_coords_to_a_sphere(pos_3d)

    v = pos_3d - centre 
    r = np.linalg.norm(v, axis=1)
    theta = np.arccos(v[:, 2] / r) #polar angle
    phi = np.arctan2(v[:, 1], v[:, 0]) #azimuth

    return theta, phi

def build_sh_basis(theta, phi, L):
    rows = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y = sph_harm(m, l, phi, theta)  # note: scipy convention is (m, l, phi, theta)
            if m > 0:
                row = np.sqrt(2) * Y.real
            elif m < 0:
                row = np.sqrt(2) * Y.imag
            else:
                row = Y.real
            rows.append(row)

    return np.array(rows) #real-valued basis

#old version, builds iteratively with lpmv
# def build_sh_basis(theta, phi, L): #theta and phi are used to evaluate values from the sphere (this is where the approximation of the human skull as a sphere comes from)
#     rows = []
#     #evaluating at each L and m
#     for l in range(L+1):
#         for m in range(-l, l+1):
#             #N^ℓ_m part    
#             norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
#             legendre = lpmv(abs(m), l, np.cos(theta)) #P^ℓ_m(cosθ) part

#             #evaluated with e^(imϕ); real-valued
#             if m > 0:
#                 row = np.sqrt(2) * norm * legendre * np.cos(m * phi)
#             elif m < 0:
#                 row = np.sqrt(2) * norm * legendre * np.sin(abs(m) * phi)
#             else:
#                 row = norm * legendre
#             rows.append(row)

#     return np.array(rows) #Y shape (L+1)^2, n_electrodes; row order is l=0..L, m=-l..l, the fixed mode order the model emits

#b is shape (n_channels, F), the raw feature stack pre-image; Y is the shape above
#using least squares method to solve for coefficients that compress the current b; 
def solve_sh_coefficients(Y, b, lam=1e-3):
    gram = Y @ Y.T
    gram[np.diag_indices_from(gram)] += lam
    return np.linalg.solve(gram, Y @ b) #coefficients shape ((L+1)^2, F)

#the reverse of the above; assume the model's delta predictions have already applied to c for reconstruction
#shape ((L+1)^, F) --to-> (n_electrodes, F)
def reconstruct_from_sh(Y, c):
    return Y.T @ c #n_electrodes x F 


#----------------
#Feature Stack; assume x is a raw n_electrodes x window_size input
#----------------
def _window_variance(x):
    if x.shape[1] == 0: #if window_size < 1, can't compute anything
        return np.zeros(x.shape[0])
    
    return np.var(x, axis=1) #variance along time dim (columns)

#has robust central tendency
def compute_median(x):
    return np.median(x, axis=1)

def compute_iqr(x):
    q75, q25 = np.percentile(x, [75, 25], axis=1)
    return q75 - q25 

#hjorth mobility, 
#time-domain proxy for mean frequency, zero where the signal does not vary
def hjorth_mobility(x):
    var_x = _window_variance(x)
    var_dx = _window_variance(np.diff(x, axis=1))
    return np.sqrt(np.divide(var_dx, var_x, out=np.zeros_like(var_x), where=var_x > 0))

#hjorth complexity, 
#time-domain proxy for bandwidth, zero where the first derivative does not vary
def hjorth_complexity(x):
    dx = np.diff(x, axis=1)
    mob_x = hjorth_mobility(x)
    mob_dx = hjorth_mobility(dx)
    return np.divide(mob_dx, mob_x, out=np.zeros_like(mob_x), where=mob_x > 0)


#------------
#to-image transformation/interpolation tensor M (H*W, F), where F= number of toggled features
#------------
def azimuthal_2d_projection(theta, phi):
    x = theta * np.cos(phi)
    y = theta * np.sin(phi)
    return np.column_stack([x, y])

#returns M
def build_img_interpolation(electrode_pos_2d, img_res, margin=0.9):
    H, W = img_res

    #centre on the projection vertex and scale so the furthest electrode sits inside the grid with a margin
    scale = (margin * min(H, W) / 2) / np.max(np.linalg.norm(electrode_pos_2d, axis=1))
    pix = electrode_pos_2d * scale + np.array([W / 2, H / 2])  #electrode positions in pixel coords, (x, y)

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid = np.column_stack([xs.ravel(), ys.ravel()])  #pixel centres as (x, y), row-major over (H, W)

    rbf = RBFInterpolator(pix, np.eye(len(electrode_pos_2d)), kernel="thin_plate_spline")
    return rbf(grid)  #each column is the spread of one electrode, shape (H*W, n_channels)

#encode a per-electrode feature stack onto the image grid, image = M @ features reshaped to (H, W, F)
def encode_image(M, features, img_res):
    H, W = img_res
    return (M @ features).reshape(H, W, -1) #-1 lets F follow the active toggles

#----------------
#Reference
#----------------
#deterministic reference of a window, skipped when the recording already matches the target
def re_reference(data, target_ref, ref_scheme, channel_names):
    if isinstance(target_ref, str) and ref_scheme is not None and ref_scheme.lower() == target_ref.lower():
        return data #already in the target reference, do nothing
    
    if target_ref == "average":
        return data - data.mean(axis=0, keepdims=True) #common average reference
    
    keys = [c.lower() for c in channel_names]
    idx = [keys.index(c.lower()) for c in target_ref]
    
    return data - data[idx].mean(axis=0, keepdims=True) #channel list reference


#----------------
#Metrics
#----------------
#per-feature relative residual ||target - recon|| / ||target||, norm over every axis but the last (feature)
#takes an electrode stack (n_channels, F) or an image tensor (H, W, F), zero where a target feature is all zero
def relative_residual(recon, target):
    F = target.shape[-1]
    diff = (target - recon).reshape(-1, F)
    flat_target = target.reshape(-1, F)
    num = np.linalg.norm(diff, axis=0)
    den = np.linalg.norm(flat_target, axis=0)
    return np.where(den > 0, num / den, 0.0)