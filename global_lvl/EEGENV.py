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

from .constants import MONTAGE
from .helpers import window_size_from_seconds, apply_topo_mask
from .visuals import (basis_matrix_fig, img_transform_fig)
from .components import (ElectrodeSim, FeatureField, SphericalHarmonics)
import os, json

#rebuild an env from a saved config
def load_eegenv(config_path, print_channel_resolve=False):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return EEGEnv._from_config(config, print_channel_resolve=print_channel_resolve)

class EEGEnv:
    def __init__(self, src_chn_names=None, num_chns=None, sfreq=None,
                 L_degree=None, num_features=None, feature_toggles=None, reference=None,
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
 
        self._build_features(num_features=num_features, feature_toggles=feature_toggles, reference=reference)
        self._build_sh(L_degree=L_degree)
 
    #-------- build/changers --------
    def _build_features(self, num_features, feature_toggles, reference):
        if self._electrode_sim.is_general:
            #no window ever enters a simulated env, so F must be declared and
            #there is no source signal for sfreq or a window to describe
            assert num_features is not None and num_features > 0, "a simulated env must declare num_features, is not derivable on its own"
            assert feature_toggles is None, "a simulated env has no FeatureField to toggle"
            assert reference is None, "a simulated env has no FeatureField to re-reference"
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
                                           feature_toggles=feature_toggles, reference=reference)
 
    def _build_sh(self, L_degree):
        if L_degree is None: 
            self._sh = None #the input-side EEGEnv for a model doesn't use SH so it isn't needed
            return
 
        coords = self._electrode_sim.spherical_coords
        self._sh = SphericalHarmonics(L_degree=L_degree,
                                      thetas=coords["thetas"],
                                      phis=coords["phis"])

    #handlers for when a source eeg input needs/wants to change a specific attribute
    def _rebuild_dependents(self, L_degree=None):
        self._feature_field = FeatureField(channels_order=self._electrode_sim.channels_order,
                                           feature_toggles=self._feature_field.declared_toggles,
                                           reference=self._feature_field.reference)

        if self.has_sh:
            #coords moved with the electrodes, so the basis is rebuilt rather than re-degreed
            self._build_sh(L_degree=self.basis_degree if L_degree is None else L_degree)

    #a new source, or the same names read against a different montage
    #names and montage move together, a list from one montage resolves to nothing against another
    def change_source(self, src_chn_names=None, montage=None, L_degree=None, print_channel_resolve=False):
        assert not self.is_general, "a simulated env has no source to change"
        assert src_chn_names is not None or montage is not None, "give src_chn_names, montage, or both"

        self._electrode_sim.rebuild(src_chn_names=src_chn_names,
                                    montage=montage,
                                    print_channel_resolve=print_channel_resolve)
        
        self._rebuild_dependents(L_degree=L_degree)

    #how far the interpolation reaches past the outermost electrode
    def change_img_margin(self, img_margin, L_degree=None):
        assert not self.is_general, "a simulated env's image space is fixed at construction"

        self._electrode_sim.rebuild(img_margin=img_margin)
        self._rebuild_dependents(L_degree=L_degree)

    #signal description only, nothing in the geometry depends on either
    def change_sfreq(self, sfreq):
        assert not self.is_general, "a simulated env has no source signal"
        assert sfreq is None or sfreq > 0, f"sfreq must be positive, got {sfreq}"
        self._sfreq = sfreq

    def change_window_seconds(self, window_seconds):
        assert not self.is_general, "a simulated env has no source signal to window"
        assert window_seconds is None or window_seconds > 0, f"window_seconds must be positive, got {window_seconds}"
        self._window_seconds = window_seconds

    #removed assertion for model for now as for results discussion on model coeff predictions against lower L's
    #but during actual deployment, L should not change for the model; this is just to see robustness against SH decoder
    def change_L(self, L_degree):
        # assert not self.is_general, "the model's basis is fixed, only a source env's degree is diagnostic"
        self._require_sh()
        self._sh.change_L(L_degree=L_degree)

    #raw sources are re-referenced, preprocessed ones already are
    def change_reference(self, reference):
        assert not self.is_general, "a simulated env has no FeatureField to re-reference"

        self._feature_field = FeatureField(channels_order=self._electrode_sim.channels_order,
                                           feature_toggles=self._feature_field.declared_toggles,
                                           reference=reference)

    #-------- constructors --------
    #source-side env; features always, basis only if L_degree given
    @classmethod
    def for_source(cls, src_chn_names, sfreq=None, L_degree=None, feature_toggles=None, 
                   reference=None, window_seconds=None, montage=MONTAGE, img_dims=(64, 64),
                   img_margin=0.75, print_channel_resolve=True):
        
        return cls(src_chn_names=src_chn_names,
                   sfreq=sfreq,
                   L_degree=L_degree,
                   feature_toggles=feature_toggles,
                   reference=reference,
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

    #from saved config; used in loader function
    @classmethod
    def _from_config(cls, config, print_channel_resolve=False):
        cfg = dict(config)
        cfg.pop("mode", None) #redundant, the chn args already determine it
        cfg["img_dims"] = tuple(cfg["img_dims"])
        return cls(print_channel_resolve=print_channel_resolve, **cfg)
 
    #-------- guards --------
    def _require_features(self):
        if self._feature_field is None:
            raise RuntimeError("this env has no FeatureField; build it with EEGEnv.for_source(...)")
 
    def _require_sh(self):
        if self._sh is None:
            raise RuntimeError("this env has no SphericalHarmonics Module; pass L_degree to the constructor")

    #when two envs are used (one for feature input, another for SH basis for model to use)
    #ensure img dims, and num features match
    #montage and channel count are expected to diverge
    def assert_compatible(self, other):
        assert tuple(self.img_dims) == tuple(other.img_dims), f"image dims differ: {self.img_dims} vs {other.img_dims}"
        assert self.num_features == other.num_features, f"feature count differs: {self.num_features} vs {other.num_features}"
 

    #-------- forward methods --------
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

    #one scalar per feature, e.g. a loss or a score
    def view_metric_bar(self, values, metric_name="metric", feature_names=None, subtitle=None,
                        scale=1.0, norm=False, save=False, save_path=None, file_name="metric.png"):

        import matplotlib.pyplot as plt #lazy import
        from .visuals.metrics import metric_bar_fig, save_fig, DEFAULT_SAVE_PATH
 
        fig = metric_bar_fig(values=values,
                             feature_names=feature_names or self.toggled_features,
                             metric_name=metric_name,
                             subtitle=subtitle,
                             scale=scale,
                             norm=norm)
 
        if save:
            save_fig(fig, save_path or DEFAULT_SAVE_PATH, file_name)

        plt.show()
 
    #electrode-space panels; expects (n_chns, F) for both
    def view_electrode_fields(self, true_field, recon_field, subtitle=None, scale=1.0, norm=False,
                            save=False, save_path=None, file_name="reconstruction.png"):

        import matplotlib.pyplot as plt #lazy import
        from .visuals.metrics import reconstruction_fig, save_fig, DEFAULT_SAVE_PATH
 
        fig = reconstruction_fig(true_field=true_field,
                                 recon_field=recon_field,
                                 feature_names=self.toggled_features,
                                 pos_2d=self.electrode_2d_pos,
                                 subtitle=subtitle,
                                 scale=scale,
                                 norm=norm)
 
        if save:
            save_fig(fig, save_path or DEFAULT_SAVE_PATH, file_name)

        plt.show()
 
    #image-space panels; takes the same (n_chns, F) fields and transforms both,
    #M is linear so the difference panel is the pixel difference 
    def view_image_fields(self, true_field, recon_field, apply_mask=False, subtitle=None, scale=1.0, norm=False,
                   save=False, save_path=None, file_name="pixel.png"):
        
        import matplotlib.pyplot as plt #lazy import
        from .visuals.metrics import reconstruction_fig, save_fig, DEFAULT_SAVE_PATH
 
        fig = reconstruction_fig(true_field=self.to_img(true_field, apply_mask=apply_mask),
                                 recon_field=self.to_img(recon_field, apply_mask=apply_mask),
                                 feature_names=self.toggled_features,
                                 subtitle=subtitle,
                                 scale=scale,
                                 norm=norm)
 
        if save:
            save_fig(fig, save_path or DEFAULT_SAVE_PATH, file_name)

        plt.show()

    #3x3 per feature: rows are field, dx, dy; columns are true, recon, difference
    #cycled with the arrow keys; save writes every feature as its own file, since
    #which one is on screen when the window closes is not knowable here
    def view_sobel_fields(self, true_field, recon_field, apply_mask=True, subtitle=None, scale=1.0,
                          save=False, save_path=None, file_stem="sobel"):
        
        import matplotlib.pyplot as plt #lazy import
        from .visuals.metrics import (sobel_cycle_fig, sobel_feature_fig, save_fig, DEFAULT_SAVE_PATH)
 
        #images stay unmasked here, the builder applies the mask after the operator
        true_img = self.to_img(true_field)   #(H, W, F)
        recon_img = self.to_img(recon_field) #(H, W, F)
        mask = self.topo_mask if apply_mask else None
 
        names = self.toggled_features
 
        if save:
            out = save_path or DEFAULT_SAVE_PATH
            for f, name in enumerate(names):
                one = sobel_feature_fig(true_img=true_img,
                                        recon_img=recon_img,
                                        feature_names=names,
                                        feature_index=f,
                                        mask=mask,
                                        subtitle=subtitle,
                                        scale=scale)
                save_fig(one, out, f"{file_stem}_{name}.png")
                plt.close(one)
 
        sobel_cycle_fig(true_img=true_img,
                        recon_img=recon_img,
                        feature_names=names,
                        mask=mask,
                        subtitle=subtitle,
                        scale=scale)
        plt.show()

    #-------- gif viewers --------
    #single feature per gif, resolved by name against the toggled set
    def _feature_index(self, feature):
        names = self.toggled_features
        assert names is not None, "this env has no FeatureField, no toggled features to index"
        assert feature in names, f"'{feature}' is not a toggled feature, have {names}"
        return names.index(feature)

    #electrode-space panels across a sequence; expects (N, n_chns, F) for both
    def view_electrode_fields_gif(self, true_fields, recon_fields, feature, subtitle=None, scale=1.0,
                                  save_path=None, file_name=None, dpi=100, fps=10):

        from .visuals.metrics import reconstruction_gif, DEFAULT_SAVE_PATH

        f = self._feature_index(feature)

        return reconstruction_gif(true_seq=true_fields[..., f],
                                  recon_seq=recon_fields[..., f],
                                  feature_name=feature,
                                  pos_2d=self.electrode_2d_pos,
                                  subtitle=subtitle,
                                  scale=scale,
                                  save_path=save_path or DEFAULT_SAVE_PATH,
                                  file_name=file_name or f"reconstruction_{feature}.gif",
                                  dpi=dpi, fps=fps)

    #image-space panels across a sequence; the whole stack transforms in one call
    #since M broadcasts over the leading axis, and the mask lands before the range
    def view_image_fields_gif(self, true_fields, recon_fields, feature, apply_mask=True, subtitle=None,
                              scale=1.0, save_path=None, file_name=None, dpi=100, fps=10):

        from .visuals.metrics import reconstruction_gif, DEFAULT_SAVE_PATH

        f = self._feature_index(feature)

        true_img = self.to_img(true_fields, apply_mask=apply_mask) #(N, H, W, F)
        recon_img = self.to_img(recon_fields, apply_mask=apply_mask) #(N, H, W, F)

        return reconstruction_gif(true_seq=true_img[..., f],
                                  recon_seq=recon_img[..., f],
                                  feature_name=feature,
                                  subtitle=subtitle,
                                  scale=scale,
                                  save_path=save_path or DEFAULT_SAVE_PATH,
                                  file_name=file_name or f"pixel_{feature}.gif",
                                  dpi=dpi, fps=fps)

    #3x3 for one feature across a sequence; rows are field, dx, dy
    def view_sobel_fields_gif(self, true_fields, recon_fields, feature, apply_mask=True, subtitle=None,
                              scale=1.0, save_path=None, file_name=None, dpi=100, fps=10):

        from .visuals.metrics import sobel_gif, DEFAULT_SAVE_PATH

        f = self._feature_index(feature)

        true_img = self.to_img(true_fields)   #(N, H, W, F)
        recon_img = self.to_img(recon_fields) #(N, H, W, F)

        return sobel_gif(true_seq=true_img[..., f],
                         recon_seq=recon_img[..., f],
                         feature_name=feature,
                         mask=self.topo_mask if apply_mask else None,
                         subtitle=subtitle,
                         scale=scale,
                         save_path=save_path or DEFAULT_SAVE_PATH,
                         file_name=file_name or f"sobel_{feature}.gif",
                         dpi=dpi, fps=fps)

    #one scalar per feature per window; expects (N, F), all features in one gif
    def view_metric_bar_gif(self, values_seq, metric_name="metric", feature_names=None, subtitle=None,
                            scale=1.0, save_path=None, file_name=None, dpi=100, fps=10):

        from .visuals.metrics import metric_bar_gif, DEFAULT_SAVE_PATH

        return metric_bar_gif(values_seq=values_seq,
                              feature_names=feature_names or self.toggled_features,
                              metric_name=metric_name,
                              subtitle=subtitle,
                              scale=scale,
                              save_path=save_path or DEFAULT_SAVE_PATH,
                              file_name=file_name or f"{metric_name}.gif",
                              dpi=dpi, fps=fps)
    
 
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

    #the constructor state as json, enough for load_eegenv to rebuild this env exactly
    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=1)

        return path
    
    #-------- solver extractors --------
    def extract_solver_operator(self, is_torch=False, device=None):
        import numpy as np
        identity = np.eye(self.num_channels) #nchns x nchns
        coeffs = np.array(self.deterministic_compress(feature_vectors=identity)) #coeffs x nchns

        if is_torch:
            import torch
            assert device is not None, "Please specify device if extracting as a Pytorch tensor"
            return torch.as_tensor(coeffs, dtype=torch.float32, device=device), torch.as_tensor(identity, dtype=torch.float32, device=device)

        #numpy, coeffs x nchns is the linear transform of the solver 
        return coeffs, identity

    #residual is what the basis could not represent
    def extract_residual_operator(self, is_torch=False, device=None):
        import numpy as np
        solver, identity = self.extract_solver_operator(is_torch=False) #coeffs x nchns as np
        recon = np.array(self.decode_coeffs(coeffs=solver)) #recon of identity
        residual = identity - recon

        if is_torch:
            import torch
            assert device is not None, "Please specify device if extracting as a Pytorch tensor"
            return torch.as_tensor(residual, dtype=torch.float32, device=device)


        return residual #numpy, nchns x nchns
    
    #-------- component split --------
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

    @property
    def reference(self):
        return self._feature_field.reference if self.has_features else None
 
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
    def total_coeffs(self): #is (L+1)^2, same number as modes
        return self._sh.total_coeffs if self.has_sh else None