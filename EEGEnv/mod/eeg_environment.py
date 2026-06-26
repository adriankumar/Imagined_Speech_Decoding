import numpy as np 
from .sphereical_harmonics import SphericalHarmonics as sh
from .feature_stack import FeatureStack as fs
from .helpers import (DEFAULT_EXCLUDE,
                      check_img_res, check_margin, validate_explicit_args, 
                      is_string, remove_nonex_chns, classify_chns_w_montage,
                      reconcile_target_ref, validate_sampling_rate,
                      reconcile_window_size, _lowercase_key, build_pick_order,
                      get_channel_positions, compute_spherical_angles, 
                      azimuthal_2d_projection, build_img_interpolation,
                      encode_image, re_reference, get_window_data, check_window_range,
                      relative_residual,
                      )

from collections import namedtuple

#paired decode output, the predicted tensors alongside the scores that measure them, produced together from the same locals
DecodePreview = namedtuple("DecodePreview",
    ["predicted_stack", "target_stack", "delta_coeffs",
     "electrode_residual", "image_residual",
     "before_image", "delta_image", "after_image"])


class EEGEnv:
    def __init__(self, 
                 #explicit args; everything downstream is re-computed/re-evaluated when src or montage selection changes
                 current_src=None, #current bci recording source; .edf files for now
                 current_montage=None,  #BCI electrode placement standard; 
                 #arg for meta-data
                 current_ref_scheme=None, #trivial, indicates what reference style the voltage values in the current recording for each electrodes was computed 
                 #default args, can be re-specified
                 re_ref_target="average", #re-reference voltage for multiple recordings for shared representation of values; or specify electrode list
                 img_res=(64, 64), #img resolution for heatmap representation of features
                 
                 #channel args
                 excluded_chns=None, #takes list of channel names from current bci src to manually exclude (and used by auto-exclude)
                 auto_exclude=True, #prompts user on terminal to exclude channels it found unresolvable for current selected montage, or change montage
                 
                 #derivable args but can be specified
                 sfreq=None, #sampling rate of current recording 
                 window_size=500, #window size of EEG for feature image
                 L_sh_degree=4, #variable that should remain fixed when attached to a model; but chanegable for diagnostics independent of model
                 f_weights=None, #scalar weights to make feature values of equal magnitude to balance model attention and spherical harmonics coefficients
                 margin=0.9,
                 dtype=np.float64,
                 ):
        
        self._validate_recording_ref(current_ref_scheme) #meta-data for current recording, not inferrable
        self._recording_ref = current_ref_scheme #trivial arg just for clarity 

        check_img_res(img_res)
        self.img_res = img_res 
        self.window_size = window_size
        self.dtype = dtype 
        check_margin(margin)
        self.margin = margin 
        self.L_degree = L_sh_degree
        self.features = fs() #holds and computes the the raw features; shape (n_electrodes, F)

        #add weights if init with it; can change arbitrarily for diagnostics, but kept constant for ml
        if f_weights is not None:
            self.features.change_weights(f_weights)
        
        self.window_cursor = 0 #window start index

        #configure per-recording variables and operators on instantiation, buffers build inside _configure now
        self._configure(src=current_src, montage=current_montage, target_ref=re_ref_target,
                        excluded_chns=excluded_chns, sfreq=sfreq, auto_exclude=auto_exclude, 
                        print_out=True)
        
    #=====
    #meta data extraction
    #====
    def _get_channel_names(self, raw):
        return raw.ch_names

    def _get_sampling_rate(self, raw):
        return raw.info['sfreq']

    def _get_timepoints(self, raw):
        return raw.n_times

    def _validate_recording_ref(self, ref):
            if ref is not None:
                is_string(ref)

    #=====
    #private helpers
    #====
    #load header for meta-data
    def _bind_src(self, src, preload=False, verbose=False):
        import mne #lazy import
        return mne.io.read_raw_edf(src, preload=preload, verbose=verbose)
    
    #convenience function, auto-exclude unresolved channels or manually do it
    def _accept_auto_exclude(self, unresolved, montage, auto_exclude):
        if auto_exclude is True: #if set to true automatically
            return True 
        
        if auto_exclude == "prompt": #for internal setting
            print(f"channels {unresolved} are not recognised in {montage}")
            return input("auto-exclude these channels for this recording? [y/n]: ").strip().lower() == "y"
        
        return False #else

    #full reconfigure, every read and build lands in locals, the harmonic build runs the capacity check, commit happens once at the end
    def _configure(self, src, montage, target_ref, excluded_chns, sfreq, 
                   auto_exclude=False, print_out=True):
        
        #1. validate explicit args
        validate_explicit_args(src, montage)

        #2. read the header and infer channels
        raw_header = self._bind_src(src)
        ichns = self._get_channel_names(raw_header)

        #3. reconcile channels and classify against the montage
        existing_chns_to_exclude = remove_nonex_chns(excluded_chns, ichns)
        classification = classify_chns_w_montage(ichns, montage, existing_chns_to_exclude)
        unresolved_chns = classification["unresolved"]

        if unresolved_chns and not self._accept_auto_exclude(unresolved_chns, montage, auto_exclude):
            print(f"channels {unresolved_chns} are not recognised in {montage}; "
                  f"auto-exclude them, add them to exclude_chns, or change the montage")

        resolved_chns = classification["resolved"]
        n_chns = len(resolved_chns)

        #4. reconcile reference and window size, capacity is enforced once at the harmonic build below
        reconciled_target_ref = reconcile_target_ref(target_ref, resolved_chns)
        inferred_sfreq = self._get_sampling_rate(raw_header)
        validate_sampling_rate(sfreq, inferred_sfreq)
        timepoints = self._get_timepoints(raw_header)
        window_size = reconcile_window_size(self.window_size, timepoints)

        #5. geometry and operators into locals, sh build raises on a capacity failure before any state is committed
        pick_order = build_pick_order(ichns, resolved_chns)
        electrode_pos_3d = get_channel_positions(montage, resolved_chns)
        theta, phi = compute_spherical_angles(electrode_pos_3d)
        electrode_pos_2d = azimuthal_2d_projection(theta, phi)
        M, harmonics = self._build_operators(electrode_pos_2d, theta, phi)

        #6. commit, the env moves to the new recording now that every step has cleared
        self.src = src 
        self.montage = montage 
        self.auto_exclude = auto_exclude
        self.target_ref = reconciled_target_ref

        self.default_ex_chns = classification["default_dropped"]
        self.auto_ex_chns = unresolved_chns #empty when all resolve
        self.ex_chns = [c for c in existing_chns_to_exclude if _lowercase_key(c) not in DEFAULT_EXCLUDE]
        self.chns_list = resolved_chns
        self.n_chns = n_chns

        self.sfreq = inferred_sfreq
        self.timepoints = timepoints 
        self.window_size = window_size
        self._recording_header = raw_header
        self._pick_order = pick_order

        self.electrode_pos_3d = electrode_pos_3d
        self.theta, self.phi = theta, phi
        self.electrode_pos_2d = electrode_pos_2d
        self.M = M
        self.harmonics = harmonics

        if print_out:
            if self.default_ex_chns:
                print(f"auto-dropped auxiliary channels: {self.default_ex_chns}")
            if self.auto_ex_chns:
                print(f"auto-excluded unresolved channels: {self.auto_ex_chns}")
            print(f"Successfully configured ({self.n_chns} channels, L fixed at {self.L_degree})")

    #pure builder, returns the image operator and the harmonics object from geometry, no state mutation
    def _build_operators(self, electrode_pos_2d, theta, phi):
        M = build_img_interpolation(electrode_pos_2d, self.img_res, self.margin)
        harmonics = sh(theta, phi, self.L_degree) #owns L and Y, capacity check fires on build
        return M, harmonics
    
    #=====
    #read & stream functions
    #====   
    #read one window and cast to the env dtype, reference applied so the values match what features see
    def _read_window(self, start, apply_ref=True):
        check_window_range(start, self.window_size, self.timepoints)
        data = get_window_data(self._recording_header, self._pick_order, start, start + self.window_size)
        
        if apply_ref:
            data = re_reference(data, self.target_ref, self._recording_ref, self.chns_list)

        return np.ascontiguousarray(data, dtype=self.dtype)

    #the raw per-channel window before any feature, shape (n_channels, window_size), defaults to the cursor
    def get_raw_window(self, start=None, apply_ref=True):
        start = self.window_cursor if start is None else start
        return self._read_window(start, apply_ref) #(n_channels, window_size)

    #read the cursor window, fold it into the feature stack advancing the lag cache, step the cursor one window on
    def advance_window_features(self):
        window = self._read_window(self.window_cursor)
        stack = self.features.compute_features(window, advance_lag=True)
        self.window_cursor += self.window_size
        return stack #(n_electrodes, F); not image representation

    #read any window read-only, no cursor move and no lag advance, for an isolated diagnostic look
    #lag terms here are relative to the live stream's last advanced window, reset_stream first for a cold preview
    def preview_window_features(self, start, feature_toggles=None):
        window = self._read_window(start)
        return self.features.compute_features(window, feature_toggles=feature_toggles, advance_lag=False)

    #clear the cursor and the feature stack's lag cache so the next advance reinitialises both
    def reset_stream(self):
        self.window_cursor = 0
        self.features.reset()

    #interpolate a feature stack onto the image grid
    def to_image(self, features):
        return encode_image(self.M, features, self.img_res) #shape (H, W, F)

    #compress a feature stack onto the harmonic basis
    def to_sh_compression(self, features):
        return self.harmonics.compress(features) #shape ((L+1)^2, F)

    #per-feature relative reconstruction residual of a stack, a current-window diagnostic not a training loss
    def harmonic_residual(self, features, return_raw=False):
        return self.harmonics.compression_residual(features, return_gap=return_raw)

    #=====
    #decoder functions
    #====   
    #reconstruct an electrode stack from coeffs, Y^T c, shape (n_channels, F)
    def decode_to_electrodes(self, coeffs):
        return self.harmonics.reconstruct(coeffs)

    #image-space decode for viewing, before is the true current image, after grafts the predicted nudge on
    #delta_coeffs is the predicted change applied to the current state, pass zeros for a pure current-window view
    def decode_to_image(self, stack, delta_coeffs):
        before = self.to_image(stack)
        delta_electrode = self.decode_to_electrodes(delta_coeffs)
        delta_image = self.to_image(delta_electrode)
        after = before + delta_image
        return before, delta_image, after
    
    #decode preview at the current cursor, reads the current and the one-step-future window read-only, advances nothing
    #delta_coeffs None forms the true delta c_next - c_current (the floor), zeros gives persistence, a model array gives the prediction
    #lag stays relative to the live stream: a scratch copy reproduces the current window then computes next against it, live state untouched
    def preview_decode_at_cursor(self, delta_coeffs=None):
        start = self.window_cursor
        next_start = start + self.window_size
        check_window_range(next_start, self.window_size, self.timepoints) #the future window must fit

        scratch = self.features.copy() #inherits the live lag cache, absorbs both writes so the live stream is untouched

        b_current = scratch.compute_features(self._read_window(start), advance_lag=True) #reproduces the live current window, primes scratch prev with current
        b_next = scratch.compute_features(self._read_window(next_start), advance_lag=False) #lag is next relative to current

        c_current = self.harmonics.compress(b_current)
        c_next = self.harmonics.compress(b_next)
        delta = (c_next - c_current) if delta_coeffs is None else delta_coeffs

        #electrode space scores the pure reconstruction Y^T(c_current + delta) against the true next stack
        predicted_stack = self.harmonics.reconstruct(c_current + delta)
        electrode_residual = relative_residual(predicted_stack, b_next)

        #image space scores the true current image plus the predicted nudge against the encoded next window
        before, delta_image, after = self.decode_to_image(b_current, delta)
        image_residual = relative_residual(after, self.to_image(b_next))

        return DecodePreview(predicted_stack=predicted_stack, target_stack=b_next, delta_coeffs=delta,
                             electrode_residual=electrode_residual, image_residual=image_residual,
                             before_image=before, delta_image=delta_image, after_image=after)
    
    #=====
    #changer functions
    #====   
    #image operator tier, geometry and the basis untouched 
    #rebuild M at a new resolution
    def change_img_res(self, img_res):
        check_img_res(img_res)
        self.M = build_img_interpolation(self.electrode_pos_2d, img_res, self.margin)
        self.img_res = img_res

    #rebuild M at a new disk-fill margin
    def change_margin(self, margin):
        check_margin(margin)
        self.M = build_img_interpolation(self.electrode_pos_2d, self.img_res, margin)
        self.margin = margin

    #harmonic tier, M untouched, set_L raises on capacity before mutating so the basis stays valid on failure
    def change_L(self, L):
        self.harmonics.set_L(L)
        self.L_degree = L

    #scalar tier, no buffers, lag cache cleared since the window content shifts 
    def change_window_size(self, window_size):
        self.window_size = reconcile_window_size(window_size, self.timepoints)
        self.reset_stream()

    def change_target_ref(self, target_ref):
        self.target_ref = reconcile_target_ref(target_ref, self.chns_list)
        self.reset_stream()

    #feature tier, delegated to the stack 
    #weights only scale magnitude so the lag cache stays valid, no reset
    def change_feature_weights(self, weights):
        self.features.change_weights(weights)

    #the stack resets its own lag cache when the active set changes
    def change_feature_toggles(self, toggles):
        self.features.change_toggles(toggles)

    #full reconfigure tier
    def change_source(self, src):
        self._configure(src=src, montage=self.montage, target_ref=self.target_ref,
                        excluded_chns=self.ex_chns, sfreq=None, auto_exclude=self.auto_exclude)
        self.reset_stream()

    def change_montage(self, montage):
        self._configure(src=self.src, montage=montage, target_ref=self.target_ref,
                        excluded_chns=self.ex_chns, sfreq=None, auto_exclude=self.auto_exclude)
        self.reset_stream()

    def change_excluded_chns(self, excluded_chns):
        self._configure(src=self.src, montage=self.montage, target_ref=self.target_ref,
                        excluded_chns=excluded_chns, sfreq=None, auto_exclude=self.auto_exclude)
        self.reset_stream()

    @property
    def Y(self):  #harmonic basis ((L+1)^2, n_channels)
        return self.harmonics.basis

    @property
    def L(self):  #harmonic degree,
        return self.harmonics.degree

    @property
    def n_modes(self):  #harmonic mode count (L+1)^2
        return self.harmonics.n_modes

    @property
    def at_stream_end(self):  #true when the cursor window would run past the recording
        return self.window_cursor + self.window_size > self.timepoints