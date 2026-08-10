import numpy as np
import math
from scipy.special import sph_harm_y

#after getting the 3d positions of the electrode channels from the idealised head model
#from MNE, compute the idealised sphere's centre and radius
def _fit_coords_to_a_sphere(pos_3d):
    #algebraic least-squares, solve |p|^2 = 2c.p + (r^2 - |c|^2) 
    #for centre and the offset term\
    f = np.sum(pos_3d**2, axis=1)
    A = np.column_stack([2 * pos_3d, np.ones(len(pos_3d))])
    sol, *_ = np.linalg.lstsq(A, f, rcond=None)
    centre = sol[:3]
    radius = np.sqrt(sol[3] + centre @ centre)

    return centre, radius #radius does not affect basis, can be ignored

#for each electrode's 3d position on the idealised head model 
#get the theta and phi angles that also describe it's position in spherical coordinates
def compute_spherical_angles(pos_3d):
    #approximate sphere radius and centre from 3d positions by using least-squares to fit
    #electrode positions onto a sphere
    centre, radius = _fit_coords_to_a_sphere(pos_3d)

    v = pos_3d - centre 
    r = np.linalg.norm(v, axis=1)
    theta = np.arccos(v[:, 2] / r) #polar angle
    phi = np.arctan2(v[:, 1], v[:, 0]) #azimuth

    return theta, phi

#SH equation: Y^ℓ_m(θ,ϕ)=N^ℓ_m P^ℓ_m(cosθ) e^(imϕ)
#This builds the basis matrix Y of dims: (L+1)^2 x num_electrodes,
#which gets multiplied with the coefficients vector (L+1)^2 to deterministically construct a field vector
def build_sh_basis(theta, phi, L):
    rows = []
    for l in range(L + 1): #l index
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, m, theta, phi)  #(l, m, theta, phi)
            if m > 0:
                row = np.sqrt(2) * Y.real
            elif m < 0:
                row = np.sqrt(2) * Y.imag
            else:
                row = Y.real
            rows.append(row)

    return np.array(rows) #real-valued basis

def sh_mode_degrees(n_modes):
    #modes assumed ordered by ascending degree, m within each degree
    L = int(round(np.sqrt(n_modes))) - 1
    assert (L + 1)**2 == n_modes, "n_modes must be a perfect square (L+1)^2"
    degrees = np.arange(L + 1)
    return np.repeat(degrees, 2 * degrees + 1).astype(np.float64)

def solve_sh_coefficients(Y, field_vec, lam=1e-3, degree_weighted=False, p=1.0):
    #Y is ((L+1)^2, n_chns); A = Y^T is (n_chns, (L+1)^2); x = c is what we solve for
    #field_vec is (n_chns, F); b in the objective

    #self-overlap: A^T A = Y @ Y.T -> ((L+1)^2, (L+1)^2)
    #how much each retained mode overlaps every other mode, at these electrode positions
    self_overlap = Y @ Y.T

    #overlap: A^T b = Y @ field_vec -> ((L+1)^2, F)
    #how much each mode resembles the measured field
    overlap = Y @ field_vec

    #B^T B term: default B = I -> ridge-regression form, every mode penalised equally
    penalty = np.full(self_overlap.shape[0], lam)

    if degree_weighted:
        #B = diag( sqrt(l(l+1))^p ) -> penalises fine (high-degree) modes more,
        #so B^T B is diag( (l(l+1))^p ), still diagonal since B is diagonal
        l = sh_mode_degrees(self_overlap.shape[0])
        w = (l * (l + 1.0))**p
        w /= w.mean()                    #keeps lam comparable across p and L
        penalty = lam * np.maximum(w, 1e-3)

    #(A^T A + lam * B^T B) c = A^T b
    lhs = self_overlap.copy() #left hand side of formula
    lhs[np.diag_indices_from(lhs)] += penalty

    return np.linalg.solve(lhs, overlap)  #coefficients, shape ((L+1)^2, F)

#the deterministic construction back into the vector field
#Y shape: num_electrodes x (L+1)^2
#coeffs shape: ((L+1)^2 x F) -> (num_electrodes x F)
def decoded_coeffs(Y, coeffs):
    return Y @ coeffs #(num_electrodes x F)

#highest L degree whose coefficients fit a specified n_chns, 
def max_L_for_chns(nchns):
    return max(math.isqrt(max(nchns, 1)) - 1, 0)

def is_compatible(L, nchns):
    if nchns < (L+1)**2: #choose lower L as a compression on higher electrode counts
        return False 
    return True
