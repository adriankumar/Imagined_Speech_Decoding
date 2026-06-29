from .helpers import (build_sh_basis, solve_sh_coefficients, reconstruct_from_sh,
                      check_harmonic_capacity, relative_residual,
                      )


#harmonic basis state and the operators over it, built from the interpolation geometry's spherical angles
#owns the degree L and the basis Y, the interpolation operator M stays on the image side, the two meet at the angles
class SphericalHarmonics:
    #build the basis at degree L from the electrode angles
    def __init__(self, theta, phi, L):
        self._theta = theta #private because they change from computation not user input
        self._phi = phi
        self._build(L)

    #rebuild the basis at a degree, the single place the capacity invariant and the build live
    def _build(self, L):
        #the trade off is higher channels need more modes (L+1)^ (or larger L), but a larger L means
        #any bci input where n_electrodes < (L+1)^2, then use another env module or lower the value of L
        check_harmonic_capacity(L, len(self._theta)) #Currently prevents electrodes less than (L+1)^2
        self.L = L
        self.Y = build_sh_basis(self._theta, self._phi, L)

    #change the harmonic degree, rebuilds the basis from the stored angles
    def set_L(self, L):
        self._build(L)

    #least squares solver for coefficients; 
    def compress(self, features):
        return solve_sh_coefficients(self.Y, features) #shape ((L+1)^2, F)

    #sh reconstruction from coefficients through Y^T; 
    def reconstruct(self, coeffs):
        return reconstruct_from_sh(self.Y, coeffs) #shape (n_channels, F)

    #electrode-space gap between a reconstruction from coeffs and a target stack, target - recon
    def gap(self, coeffs, target):
        return target - self.reconstruct(coeffs)

    #per-feature relative residual of a reconstruction from coeffs against a target stack
    #return_gap also hands back the raw electrode-space gap for inspection
    def residual(self, coeffs, target, return_gap=False):
        recon = self.reconstruct(coeffs)
        score = relative_residual(recon, target)
        if return_gap:
            return target - recon, score
        return score

    #compression residual of a stack against its own self-reconstruction, the current-window fidelity score
    def compression_residual(self, features, return_gap=False):
        return self.residual(self.compress(features), features, return_gap=return_gap)

    # def get_gap(self, features):
    #     recon = self.reconstruct(self.compress(features))
    #     return features - recon 
    
    # #the L2 norm of features - reconstruction; used for current window diagnostics
    # #not as the loss for ML or delta   
    # def get_residual(self, features, return_raw=False):
    #     gap = self.get_gap(features)
    #     num = np.linalg.norm(gap, axis=0)
    #     den = np.linalg.norm(features, axis=0)

    #     #‖features - recon‖ / ‖features‖ basically computing L2 norm on current window
    #     euclid_dist = np.where(den > 0, num / den, 0.0) #(condition, if true, else)

    #     if return_raw:
    #         return gap, euclid_dist

    #     return euclid_dist

    #the basis matrix ((L+1)^2, n_channels)
    @property
    def basis(self):
        return self.Y

    #the harmonic degree
    @property
    def degree(self):
        return self.L

    #the number of modes (L+1)^2
    @property
    def n_modes(self):
        return (self.L + 1) ** 2

    #the channel count the basis is sampled at
    @property
    def n_channels(self):
        return len(self._theta)