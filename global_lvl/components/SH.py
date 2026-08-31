#spherical harmonics deterministic compressor, and decoder

from ..helpers import (build_sh_basis, solve_sh_coefficients, decoded_coeffs,
                       max_L_for_chns, is_compatible)

from ..constants import SOLVER_TYPES

#no electrode sim directly passed, built from L, and spherical coords
class SphericalHarmonics:
    def __init__(self, L_degree, thetas, phis):

        #assert so eegenv wrapper is never partially built
        assert is_compatible(L=L_degree, nchns=len(thetas)), f"Lower value of L (L={L_degree} | coeffs={(L_degree+1)**2}) for data-field of size {len(thetas)}; max L degree that can fit is {max_L_for_chns(len(thetas))}"
            
        self._theta = thetas 
        self._phi = phis

        self._build(L=L_degree)

    def _build(self, L):
        self._L = L #highest degree of spatial pattern
        #shape coeffs x n_channels
        #esnure to use Y.T for decoding 
        self._Y = build_sh_basis(theta=self._theta,
                                 phi=self._phi,
                                 L=self._L)

    def change_L(self, L_degree):
        assert is_compatible(L=L_degree, nchns=len(self._theta)), f"Lower value of L (L={L_degree} | coeffs={(L_degree+1)**2}) for data-field of size {len(self._theta)}; max L degree that can fit is {max_L_for_chns(len(self._theta))}"
        self._build(L=L_degree)
        print("Successfully re-built basis matrix")
             
    #deterministic solvers
    #expects feature vector input in shape n_chns x F
    def solve_coeffs(self, feature_vectors, solver_type, lam=1e-3, p=1.0):
        assert solver_type in SOLVER_TYPES, f"unrecognised solver type: {solver_type}, current options are: {SOLVER_TYPES}"
        
        if solver_type == "B=diag":
            weighted = True
        else:
            weighted = False

        #returns shape coeffs x F
        return solve_sh_coefficients(Y=self._Y, field_vec=feature_vectors, degree_weighted=weighted, lam=lam, p=p)

    #coeffs expected shape coeffs x F, from any source (learned or solved for)
    def construct_data_field(self, coeffs):
        return decoded_coeffs(Y=self._Y.T, coeffs=coeffs) #returns n_chns x F

    @property 
    def basis_degree(self):
        return self._L

    @property 
    def basis_matrix(self):
        return self._Y #coeffs x n_chns

    @property 
    def total_coeffs(self):
        return (self._L + 1)**2 #also number of basis functions used from chosen L
    