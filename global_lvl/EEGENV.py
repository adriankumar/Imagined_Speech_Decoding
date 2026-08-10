#EEGEnv holds 3 components
#electrode sim - always provided;
#   its the geometry root of the electrode features; 
#   image interpolation matrix and masking come from it

#feature field - only included when a raw signal is being provided and featurised
#   computes the toggled, time-collapsed features on a given window input
#   and image transforms a feature

# SphericalHarmonics - only when compression is needed; i.e deterministic solving or learnable model for basis matrix 
#   in all cases, it provides the coefficient decoder (SH basis matrix)
#   if only using for deterministic solver, provides the function for it


#for example a model doesn't need a feature field component in
#its EEGEnv as it never computes features, it only requires the SH
#component for reconstruction into the feature field

#EEGEnv is numpy-cpu first, torch extensions, like for the model
#will just extract the fixed arrays and move to torch like 
#transforms M an Y

from .constants import MONTAGE, FEATURE_NAMES
from .helpers import window_size_from_seconds, apply_topo_mask
from .visuals import (basis_matrix_fig, img_transform_fig)

from .components import (ElectrodeSim, FeatureField, SphericalHarmonics)

class EEGEnv:
    def __init__(self, src_chn_names=None, num_chns=None, sfreq=None,
                 L_degree=None, num_features=None, feature_toggles=None,
                 window_seconds=None, montage=MONTAGE, img_dims=(64, 64),
                 img_margin=0.75, print_channel_resolve=True):
 
        #electrode sim owns the src_chn_names XOR num_chns rule, so mode is
        #decided there and read back rather than re-validated here
        self._electrode_sim = ElectrodeSim(src_chn_names=src_chn_names,
                                           num_chns=num_chns,
                                           montage=montage,
                                           img_dims=img_dims,
                                           img_margin=img_margin,
                                           print_channel_resolve=print_channel_resolve)
 
        #source-signal metadata is here, not in the geometry; re-referencing
        #and anything else describing the incoming signal belongs alongside it
        self._sfreq = sfreq
        self._window_seconds = window_seconds
        self._declared_features = None
 
        self._build_features(num_features=num_features, feature_toggles=feature_toggles)
        self._build_sh(L_degree=L_degree)
 
    #-------- build --------
    def _build_features(self, num_features, feature_toggles):
        if self._electrode_sim.is_general:
            #no window ever enters a simulated env, so F must be declared and
            #there is no source signal for sfreq or a window to describe
            assert num_features is not None and num_features > 0, "a simulated env must declare num_features, is not derivable on its own"
            assert feature_toggles is None, "a simulated env has no FeatureField to toggle"
            assert self._sfreq is None, "a simulated env has no source signal, sfreq is meaningless"
            assert self._window_seconds is None, "a simulated env has no source signal to window"
 
            self._declared_features = int(num_features)
            self._feature_field = None
            return

        #if electrode sim is not a general EEG sim, then its specific to a source and therefore
        #features are expected to be computed from a source
        assert num_features is None, "num_features is derived from the toggles in source mode, do not declare it"
        assert self._sfreq is None or self._sfreq > 0, f"sfreq must be positive, got {self._sfreq}"
 
        self._feature_field = FeatureField(channels_order=self._electrode_sim.channels_order,
                                           feature_toggles=feature_toggles)
 
    def _build_sh(self, L_degree):
        if L_degree is None: 
            self._sh = None #the input-side EEGEnv for a model doesn't use SH so it isn't needed
            return
 
        coords = self._electrode_sim.spherical_coords
        self._sh = SphericalHarmonics(L_degree=L_degree,
                                      thetas=coords["thetas"],
                                      phis=coords["phis"])
 
    #-------- factories --------
    #source-side env; features always, basis only if L_degree given
    @classmethod
    def for_source(cls, src_chn_names, sfreq=None, L_degree=None, feature_toggles=None,
                   window_seconds=None, montage=MONTAGE, img_dims=(64, 64),
                   img_margin=0.75, print_channel_resolve=True):
        
        return cls(src_chn_names=src_chn_names,
                   sfreq=sfreq,
                   L_degree=L_degree,
                   feature_toggles=feature_toggles,
                   window_seconds=window_seconds,
                   montage=montage,
                   img_dims=img_dims,
                   img_margin=img_margin,
                   print_channel_resolve=print_channel_resolve)
 
    #simulated env for the model; no features computed, F declared
    @classmethod
    def simulated(cls, num_chns, num_features, L_degree, montage=MONTAGE,
                  img_dims=(64, 64), img_margin=0.75):
        return cls(num_chns=num_chns,
                   num_features=num_features,
                   L_degree=L_degree,
                   montage=montage,
                   img_dims=img_dims,
                   img_margin=img_margin,
                   print_channel_resolve=False)
 
    #-------- guards --------
    def _require_features(self):
        if self._feature_field is None:
            raise RuntimeError("this env has no FeatureField; build it with EEGEnv.for_source(...)")
 
    def _require_sh(self):
        if self._sh is None:
            raise RuntimeError("this env has no SphericalHarmonics; pass L_degree to the factory")
 
    #-------- pipeline --------
    #raw window (n_chns, T) -> feature vectors (n_chns, F)
    #ft_toggles as arg is for diagnostics only- F will not match num_features when it is used here
    def window_to_features(self, window, ft_toggles=None):
        self._require_features()
        return self._feature_field.window_to_vec(window=window, ft_toggles=ft_toggles)
 
    #electrode-space (n_chns, F) -> image-space (H, W, F); source is irrelevant,
    #vectors may come from features or from a constructed data field
    def to_img(self, feature_vectors, apply_mask=False):
        M = self._electrode_sim.img_transform
        img = (M @ feature_vectors).reshape(*feature_vectors.shape[:-2], *self.img_dims, -1)
        return apply_topo_mask(img, self.topo_mask) if apply_mask else img

    #(n_chns, F) -> coefficients (modes, F) by the regularised normal equations
    def deterministic_compress(self, feature_vectors, solver_type="B=I", lam=1e-3, p=1.0):
        self._require_sh()
        return self._sh.solve_coeffs(feature_vectors=feature_vectors,
                                     solver_type=solver_type,
                                     lam=lam, p=p)
 
    #coefficients (modes, F) from any source, solved or learned -> (n_chns, F)
    def decode_coeffs(self, coeffs):
        self._require_sh()
        return self._sh.construct_data_field(coeffs=coeffs)

    #-------- viewers --------
    def view_coeff_decoder(self):
        self._require_sh()
        
        import matplotlib.pyplot as plt #lazy import
        basis_matrix_fig(Y=self.basis_matrix, subtitle=f"Coefficient Decoder/Basis Matrix Y | coeffs={self.total_coeffs} x nchns={self.num_channels}")
        plt.show()

    def view_img_transform(self):
        import matplotlib.pyplot as plt 
        img_transform_fig(M=self.img_transform, img_dims=self.img_dims, subtitle="Electrode-to-Image Operator")
        plt.show()

    #interactive view of the basis; each row of Y is one mode, rendered as the
    #continuous Y_lm it samples, with the electrodes drawn as that row's columns
    def view_basis_sphere(self, n_theta=48, n_phi=96):
        self._require_sh()
 
        from .visuals.components.sh_space import view_basis_sphere #lazy import
 
        coords = self.spherical_coords
        view_basis_sphere(Y=self.basis_matrix,
                          thetas=coords["thetas"],
                          phis=coords["phis"],
                          n_theta=n_theta,
                          n_phi=n_phi,
                          subtitle=f"Basis Matrix Y | L={self.basis_degree} | coeffs={self.total_coeffs} x nchns={self.num_channels}")
    #-------- mutation, diagnostics only --------
    def change_L(self, L_degree):
        self._require_sh()
        self._sh.change_L(L_degree=L_degree)
 
    #-------- config --------
    #current constructor state; declared feature toggles are included so a
    #reload restores the same F
    @property
    def config(self):
        return {
            "mode": "simulated" if self.is_general else "source",
            "src_chn_names": self._electrode_sim.original_channels,
            "num_chns": self._electrode_sim.num_channels if self.is_general else None,
            "sfreq": self._sfreq,
            "L_degree": self.basis_degree,
            "num_features": self._declared_features,
            "feature_toggles": self._feature_field.declared_toggles if self.has_features else None,
            "window_seconds": self._window_seconds,
            "montage": self.montage,
            "img_dims": list(self.img_dims),
            "img_margin": self.img_margin,
        }
 
    @classmethod
    def from_config(cls, config, print_channel_resolve=False):
        cfg = dict(config)
        cfg.pop("mode", None) #redundant, the chn args already determine it
        cfg["img_dims"] = tuple(cfg["img_dims"])
        return cls(print_channel_resolve=print_channel_resolve, **cfg)
 
    #envs on either side of the model must agree on the image contract only;
    #montage and channel count are expected to diverge
    def assert_compatible(self, other):
        assert tuple(self.img_dims) == tuple(other.img_dims), f"image dims differ: {self.img_dims} vs {other.img_dims}"
        assert self.num_features == other.num_features, f"feature count differs: {self.num_features} vs {other.num_features}"
 
    #-------- predicates --------
    @property
    def is_general(self):
        return self._electrode_sim.is_general
 
    @property
    def has_features(self):
        return self._feature_field is not None
 
    @property
    def has_sh(self):
        return self._sh is not None
 
    #-------- components --------
    @property
    def electrode_sim(self):
        return self._electrode_sim
 
    @property
    def feature_field(self):
        return self._feature_field
 
    @property
    def spherical_harmonics(self):
        return self._sh
 
    #-------- electrode space --------
    @property
    def montage(self):
        return self._electrode_sim.montage
 
    @property
    def num_channels(self):
        return self._electrode_sim.num_channels
 
    @property
    def original_channels(self):
        return self._electrode_sim.original_channels
 
    @property
    def resolved_channels(self):
        return self._electrode_sim.resolved_channels
 
    @property
    def excluded_channels(self):
        return self._electrode_sim.excluded_channels
 
    @property
    def channels_order(self):
        return self._electrode_sim.channels_order
 
    @property
    def electrode_3d_pos(self):
        return self._electrode_sim.electrode_3d_pos
 
    @property
    def electrode_2d_pos(self):
        return self._electrode_sim.electrode_2d_pos
 
    @property
    def spherical_coords(self):
        return self._electrode_sim.spherical_coords
 
    #-------- image space --------
    @property
    def img_dims(self):
        return self._electrode_sim.img_size
 
    @property
    def img_margin(self):
        return self._electrode_sim.img_margin
 
    @property
    def img_transform(self):
        return self._electrode_sim.img_transform #(H*W, n_chns)
 
    @property
    def topo_mask(self):
        return self._electrode_sim.electrode_mask #(H, W)
 
    #-------- features --------
    @property
    def toggled_features(self):
        return self._feature_field.toggled_features if self.has_features else None
 
    #declared in a simulated env, derived from the declared toggles in a source env
    @property
    def num_features(self):
        return self._declared_features if self.is_general else self._feature_field.num_features
 
    #-------- signal --------
    @property
    def sfreq(self):
        return self._sfreq
 
    @property
    def window_seconds(self):
        return self._window_seconds
 
    #fixed in seconds, so the same physical duration across differing sfreqs
    @property
    def window_size(self):
        assert self._window_seconds is not None, "no window_seconds set on this env"
        assert self._sfreq is not None, "no sfreq on this env, cannot resolve a window size"
        return window_size_from_seconds(window_seconds=self._window_seconds, sfreq=self._sfreq)
 
    #-------- basis --------
    @property
    def basis_matrix(self):
        return self._sh.basis_matrix if self.has_sh else None #(modes, n_chns)
 
    @property
    def basis_degree(self): #returns L
        return self._sh.basis_degree if self.has_sh else None
 
    @property
    def total_coeffs(self):
        return self._sh.total_coeffs if self.has_sh else None