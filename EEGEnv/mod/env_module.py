import numpy as np
from .helper_functions import (bind_source, lowercase_key, DEFAULT_EXCLUDE,
                               get_channel_names, get_sampling_rate, get_timepoints,
                               get_montage_lookup, get_channel_positions, 
                               get_interpolation_operator, get_sh_basis,
                               fit_window_size, fit_target_ref, fit_excludes,
                               build_pick_order, get_window_data, to_tensor)

from .helper_val_functions import (validate_montage, classify_channels, UnresolvedChannelsError,
                                   check_re_ref_validity, check_window_size, 
                                   check_harmonic_capacity,
                                   check_margin, check_img_res,
                                   check_window_range, check_tensor_type, check_feature_flags)

from .helper_maths import re_reference, encode_image, seconds_to_samples, robust_magnitude, synthesise_sh, solve_sh_coefficients
from .feature_stack import FeatureStack, FEATURE_NAMES



class EEGEnv:
    def __init__(self, source=None, montage=None, ref_scheme=None, #explicit arguments
                 target_ref="average", img_res=(64, 64), #default args, can be re-specified
                 chn_list=None, sampling_rate=None, exclude_chns=None, auto_exclude=False, #inferrable + cross validation
                 segment_length=300, L_degree=4, margin=0.9, feature_weight=None,
                 dtype=np.float64): #ML related inputs

        #stable configuration the source-level changers do not touch
        self.img_res = img_res #image size/shape for representation (and input into model)
        self.window_size = segment_length #temporal window size to compute feature image
        self.dtype = dtype #float64 for ML precision, float32 for lightweight computation
        self.margin = margin #fraction of the half-grid the furthest electrode reaches, intrinsic to M
        self.L_degree = L_degree #harmonic order fixed at the model level, every recording must resolve it

        self.features = FeatureStack()  #toggleable temporal features, lag carried across stepped windows
        if feature_weight is not None:
            self.features.set_weight(feature_weight)  #optional manual emphasis on the way in
   
        self._cursor = 0  #current window start for stepped reads

        check_img_res(img_res) #bound the stable config on the way in
        check_margin(margin)

        #validate and commit the source-level configuration, structure-coupled params are auto-fitted
        self._configure(source, montage, ref_scheme, target_ref, chn_list,
                        exclude_chns if exclude_chns else [], sampling_rate,
                        auto_exclude=auto_exclude, print_out=True)

        self._compute_buffers(print_out=True)

    def _validate_explct_args(self, src, montage, ref):
        #source is not None and is a valid EEG file (for now is just edf)
        if src is None or not src.endswith(".edf"):
            raise ValueError(f"Source must be specified or be an .edf file; got: {src}")

        validate_montage(montage) #self-explanatory lol

        #check ref scheme
        if ref is None or not isinstance(ref, str):
            raise ValueError(f"Reference Scheme must be specified as a string; got {ref} | type: {type(ref)}")
    
    #classify the working channel set against the montage, returns the resolved, unresolved and auxiliary breakdown, no self mutation
    def _classify_chns_with_montage(self, ichns, chns, montage, excluded_chns):
        lookup_dict = get_montage_lookup(montage) #montage electrode name look-up dictionary

        return classify_channels(specified_chns=chns, inferred_chns=ichns,
                                 chns_in_montage=lookup_dict, exclude_chns=excluded_chns)

    #decide whether unresolved channels may be dropped: False rejects so the caller raises, True accepts, 'prompt' asks on the terminal
    def _approve_auto_exclude(self, unresolved, montage, auto_exclude):
        if auto_exclude is True:
            return True
        if auto_exclude == "prompt":
            print(f"channels {unresolved} are not recognised in {montage}")
            return input("auto-exclude these channels for this recording? [y/n]: ").strip().lower() == "y"
        return False
    
    #validate the source-level config, fit the structure-coupled params to it, then commit and report
    def _configure(self, source, montage, ref_scheme, target_ref, 
                   chn_list, exclude_chns, sampling_rate, auto_exclude=False, 
                   print_out=True):
        
        #1. validate the direct user assertions, these still raise
        self._validate_explct_args(source, montage, ref_scheme)

        raw_header = bind_source(source)
        ichns = get_channel_names(raw_header) #inferred channel names from the source

        #fit the exclusions to the source, dropping any names this file does not contain
        fitted_excludes = fit_excludes(exclude_chns, ichns)
        dropped = [c for c in exclude_chns if c not in fitted_excludes]
        if dropped:
            print(f"warning: excluded channels not in source were dropped: {dropped}")

        #classify the working set, then apply the unresolved policy before any capacity check or commit
        classification = self._classify_chns_with_montage(ichns, chn_list, montage, fitted_excludes)
        resolved_chns = classification["resolved"]
        unresolved = classification["unresolved"]
        default_dropped = classification["default_dropped"]

        if unresolved and not self._approve_auto_exclude(unresolved, montage, auto_exclude):
            raise UnresolvedChannelsError(unresolved, montage)
        auto_excluded = unresolved  #empty when all resolve, the approved per-recording drop set otherwise

        n_chans = len(resolved_chns)

        #L is a fixed model-level constant, the recording must resolve it, raises here before any commit
        check_harmonic_capacity(self.L_degree, n_chans)

        fitted_ref = fit_target_ref(target_ref, resolved_chns)
        if fitted_ref != target_ref:
            print("warning: target reference no longer resolves against the channel set, reverting to average")

        isfreq = get_sampling_rate(raw_header)
        if sampling_rate is not None and sampling_rate != isfreq: #specified rate must match the source
            raise ValueError(f"Specified sampling rate: {sampling_rate} does not match inferred sampling rate: {isfreq}; leave as None or correct argument")
        sfreq = isfreq if sampling_rate is None else sampling_rate

        time_points = get_timepoints(raw_header) #upper bound of window segment
        fitted_window = fit_window_size(self.window_size, time_points)
        if fitted_window != self.window_size:
            print(f"warning: window size reduced from {self.window_size} to {fitted_window} to fit the recording length")

        #2. commit, only now that nothing further can raise
        self.source = source
        self.montage = montage
        self.ref_scheme = ref_scheme
        self.target_ref = fitted_ref
        #three disjoint exclusion buckets: auxiliary auto-drops, per-recording unresolved drops, manual non-auxiliary drops
        self.default_excluded = default_dropped
        self.auto_excluded = auto_excluded
        self.excluded_chns = [c for c in fitted_excludes if lowercase_key(c) not in DEFAULT_EXCLUDE]
        self.chn_list = resolved_chns #the resolved set, the working channel list

        self.n_chans = n_chans
        self.sfreq = sfreq
        self.time_points = time_points
        self.window_size = fitted_window
        self._raw = raw_header  #preload=False handle, window reads pull only their slice from disk
        self._pick_order = build_pick_order(ichns, resolved_chns)  #raw rows in reconciled order
        
        if print_out:
            if default_dropped:
                print(f"auto-dropped auxiliary channels: {default_dropped}")
            if auto_excluded:
                print(f"auto-excluded unresolved channels: {auto_excluded}")
            print(f"Successfully Instantiated ({n_chans} channels, L fixed at {self.L_degree})")

    #full buffer rebuild, positions then operator then basis, for source, montage or channel changes
    def _compute_buffers(self, print_out=False):
        
        pos_3d = get_channel_positions(self.montage, self.chn_list)
        self.SH_dict = get_interpolation_operator(pos_3d, self.img_res, self.margin)
        self.SH_dict.update(get_sh_basis(self.SH_dict['theta'], self.SH_dict['phi'], self.L_degree))
        
        if print_out:
            print("Spherical Harmonic buffers successfully computed")

    #re-/build only the interpolation operator and geometry from the stored 3d positions, preserves Y
    def _compute_operator(self):
        self.SH_dict.update(get_interpolation_operator(self.SH_dict['pos_3d'], self.img_res, self.margin))

    #re-/build only the spherical harmonic basis from the stored angles
    def _compute_sh_basis(self):
        self.SH_dict.update(get_sh_basis(self.SH_dict['theta'], self.SH_dict['phi'], self.L_degree))

    #resolve a window length, seconds first then an explicit sample count then the configured default
    def _resolve_length(self, window_size, window_in_seconds):
        if window_in_seconds is not None:
            return seconds_to_samples(window_in_seconds, self.sfreq)
        
        if window_size is not None:
            return window_size
        
        return self.window_size

    #read one window, re-reference it, and collapse it through the given feature stack, the shared compute core
    def _compute_window(self, start, length, features, feature_toggles=None, update=True):
        window = get_window_data(self._raw, self._pick_order, start, start + length)
        window = re_reference(window, self.target_ref, self.ref_scheme, self.chn_list)

        stack = features.compute(window, feature_toggle=feature_toggles, update=update)  #(n_channels, F)

        return stack

    #shared prep for the vis methods, resolve the window and validate before any read, default to the cursor
    def _vis_prep(self, start, window_size, window_in_seconds, feature_toggles):
        start = self._cursor if start is None else start
        length = self._resolve_length(window_size, window_in_seconds)
        check_window_range(start, length, self.time_points)
        if feature_toggles is not None:
            check_feature_flags(feature_toggles, FEATURE_NAMES)
        return start, length
    
    #changing the source requires a full rebuild, unspecified args reuse the current config so validation can surface any mismatch
    def change_source(self, source_path, montage=None, reference=None, target_ref=None,
                      chns=None, exclude_chns=None, sampling_rate=None, auto_exclude=False, print_out=True):

        self._configure(
            source=source_path,
            montage=self.montage if montage is None else montage,
            ref_scheme=self.ref_scheme if reference is None else reference,
            target_ref=self.target_ref if target_ref is None else target_ref,
            chn_list=chns, #none infers from the new source
            exclude_chns=self.excluded_chns if exclude_chns is None else exclude_chns,
            sampling_rate=sampling_rate, #none infers and validates against the new source
            auto_exclude=auto_exclude,
            print_out=print_out,
        )

        self._compute_buffers(print_out=print_out)

    #changing the montage re-resolves channels and rebuilds every buffer, routed through the full source path
    def change_montage(self, montage, exclude_chns=None, auto_exclude=False, print_out=True):
        self.change_source(self.source, montage=montage, exclude_chns=exclude_chns,
                           auto_exclude=auto_exclude, print_out=print_out)

    #changing the image resolution rebuilds only the interpolation operator
    def change_image_res(self, img_res):
        check_img_res(img_res)
        self.img_res = tuple(img_res)
        self._compute_operator()

    #changing the projection margin rebuilds only the interpolation operator
    def change_margin(self, margin):
        check_margin(margin)
        self.margin = margin
        self._compute_operator()

    #changing the harmonic order rebuilds only the spherical harmonic basis
    def change_L(self, L_degree):
        check_harmonic_capacity(L_degree, self.n_chans)
        self.L_degree = L_degree
        self._compute_sh_basis()

    #target reference must sit within the resolved set, re-references data once that layer exists
    def change_target_ref(self, target_ref):
        check_re_ref_validity(target_ref, self.chn_list)
        self.target_ref = target_ref

    #reference scheme is a free assertion about the source, re-references data once that layer exists
    def change_ref_scheme(self, ref_scheme):
        if ref_scheme is None or not isinstance(ref_scheme, str):
            raise ValueError(f"Reference Scheme must be a string; got {ref_scheme} | type: {type(ref_scheme)}")
        self.ref_scheme = ref_scheme

    #window size now arrives in seconds, converted to samples against the source rate before validation
    def change_window_size(self, segment_seconds):
        segment_length = seconds_to_samples(segment_seconds, self.sfreq)
        check_window_size(segment_length, self.time_points)
        self.window_size = segment_length

    #manual per-feature emphasis multiplied on top of the calibration scale, the deliberate-intent knob
    def change_feature_weight(self, weight):
        self.features.set_weight(weight) #set as dictionary

    #enable or reconfigure the electrode-level ema accumulation per base feature
    def change_feature_accum(self, accum, alpha=None):
        self.features.set_accum(accum)
        if alpha is not None:
            self.features.set_alpha(alpha)

    #flip one or more base feature toggles, validated against the known names, survives a stream reset
    def change_feature_toggle(self, toggle):
        check_feature_flags(toggle, FEATURE_NAMES)
        self.features.feature_toggle.update(toggle)

    #restart the stepped stream, clearing the cursor position, the lag state and the harmonic accum state
    def reset_stream(self, start=0):
        self._cursor = start
        self.features.reset()

    #measure a robust magnitude per feature over a deterministic contiguous sweep and freeze the reciprocal as the equalising scale
    def calibrate_feature_scale(self, n_windows=50, window_size=None, start=0, step=None, method="median"):
        length = self.window_size if window_size is None else window_size
        step = length if step is None else step
        check_window_range(start, length, self.time_points)

        probe = FeatureStack()  #all features, unit scale and weight, isolated lag so the live stream is untouched
        columns = {name: [] for name in FEATURE_NAMES}
        s = start
        for _ in range(n_windows):
            if s + length > self.time_points:
                break
            stack = self._compute_window(s, length, probe, update=True)  #raw, every feature in fixed order
            for i, name in enumerate(FEATURE_NAMES):
                columns[name].append(stack[:, i])
            s += step

        #robust magnitude per feature across all sampled channels and windows, reciprocal equalises the magnitudes
        scale = {}
        for name in FEATURE_NAMES:
            magnitude = robust_magnitude(np.abs(np.concatenate(columns[name])), method)
            scale[name] = 1.0 / magnitude if magnitude > 0 else 1.0

        self.features.set_scale(scale)
        return scale
    
    #compute the feature image (or raw stack) for one window, stepping the cursor when no start is given
    def get_feature_stack(self, start=None, end=None, window_size=None, window_in_seconds=None,
                          feature_toggles=None, encode_as_img=True, tensor_type="numpy"):
        
        positioned = start is not None  #an explicit start is a positioned read that leaves the cursor alone
        start = self._cursor if start is None else start

        #resolve the window length, seconds first then an end index then explicit length then the default
        if window_in_seconds is not None:
            window_size = seconds_to_samples(window_in_seconds, self.sfreq)
        elif end is not None:
            window_size = end - start
        elif window_size is None:
            window_size = self.window_size

        #validate against scalars before touching disk
        check_window_range(start, window_size, self.time_points)
        check_tensor_type(tensor_type)

        if feature_toggles is not None:
            check_feature_flags(feature_toggles, FEATURE_NAMES)

        #forward read, the live stream computes and advances its lag cache and harmonic accum state
        stack = self._compute_window(start, window_size, self.features, feature_toggles=feature_toggles,
                                     update=True)
        
        out = encode_image(self.SH_dict['interpol_operator'], stack, self.img_res) if encode_as_img else stack

        #step the cursor only for a default read, report the next start or none when no full window remains
        if not positioned:
            self._cursor = start + window_size

        next_start = start + window_size

        if next_start + window_size > self.time_points:
            next_start = None

        return to_tensor(out, tensor_type, self.dtype), next_start

    #least squares projection of a per-channel feature stack onto the harmonic basis
    #returns coefficients ((L+1)^2, F) and the relative reconstruction residual per feature (F,)
    def project_coefficients(self, stack):
        Y = self.SH_dict['SH_basis']
        coeffs = solve_sh_coefficients(Y, stack)
        recon = synthesise_sh(Y, coeffs)
        num = np.linalg.norm(stack - recon, axis=0)
        den = np.linalg.norm(stack, axis=0)
        residual = np.where(den > 0, num / den, 0.0)
        return coeffs, residual

    #interpolate a per-channel stack to the topographic image through the frozen operator M
    def encode_feature_image(self, stack):
        return encode_image(self.SH_dict['interpol_operator'], stack, self.img_res)

#read-only/GUI specific
    #read-only feature window for diagnostics, returns the (array, panel_names) pair without advancing lag, ema or the cursor
    #kind 'image' encodes through M to (H, W, F), kind 'stack' returns the raw (n_channels, F) at the electrodes
    def get_feature_window(self, start, length, feature_toggles=None, kind="image"):
        check_window_range(start, length, self.time_points)
        if feature_toggles is not None:
            check_feature_flags(feature_toggles, FEATURE_NAMES)

        stack = self._compute_window(start, length, self.features, feature_toggles=feature_toggles,
                                     update=False)
        names = self.features.enabled_names(feature_toggles)

        array = encode_image(self.SH_dict['interpol_operator'], stack, self.img_res) if kind == "image" else stack
        return array, names
    
    #read-only re-referenced raw signal over a sample range, the same read and reference path the feature stack uses
    #returns (n_channels, stop - start), for diagnostics that need recording values without features
    def get_referenced_window(self, start, stop):
        window = get_window_data(self._raw, self._pick_order, start, stop)
        return re_reference(window, self.target_ref, self.ref_scheme, self.chn_list)
    
    #read-only peek that primes the lag cache from the previous window so lag updates while sweeping
    #an isolated feature stack mirrors the live config, the live stream is never touched, ema stays cold
    def peek_stack(self, start, length, feature_toggles=None):
        check_window_range(start, length, self.time_points)
        if feature_toggles is not None:
            check_feature_flags(feature_toggles, FEATURE_NAMES)

        base_toggle = {k: self.features.feature_toggle[k]
                       for k in ("raw", "median", "iqr", "mobility", "complexity",
                                 "raw_lag", "median_lag", "iqr_lag")}
        probe = FeatureStack(**base_toggle, scale=self.features.scale, weight=self.features.weight,
                             accum=self.features.accum, alpha=self.features.alpha)

        if start >= length:
            probe.prime_lag(self.get_referenced_window(start - length, start))

        window = self.get_referenced_window(start, start + length)
        stack = probe.compute(window, feature_toggle=feature_toggles, update=False)
        names = probe.enabled_names(feature_toggles)
        return stack, names

    #advance the live stream one window, lag and ema build across calls, used by playback
    def advance_stack(self, start, length, feature_toggles=None):
        check_window_range(start, length, self.time_points)
        if feature_toggles is not None:
            check_feature_flags(feature_toggles, FEATURE_NAMES)
        stack = self._compute_window(start, length, self.features, feature_toggles=feature_toggles, update=True)
        names = self.features.enabled_names(feature_toggles)
        return stack, names

#functions related to generating and saving images and videos of current configurations in code outside of GUI

    #view one window's panels, image (interpolated field) or stack (raw electrode values), live stream untouched
    def view_plot(self, start=None, window_size=None, window_in_seconds=None, feature_toggles=None, kind="image"):
        from .helper_vis import view_window #lazy imports

        start, length = self._vis_prep(start, window_size, window_in_seconds, feature_toggles)
        view_window(self, start, length, feature_toggles, kind)

    #save one window's panels as png or jpeg
    def save_image(self, path, start=None, window_size=None, window_in_seconds=None, feature_toggles=None, kind="image"):
        from .helper_vis import save_window #lazy imports

        start, length = self._vis_prep(start, window_size, window_in_seconds, feature_toggles)

        save_window(self, path, start, length, feature_toggles, kind)

    #view an animation stepping consecutive windows on an isolated lag cache, live stream untouched
    def view_video(self, start=None, n_frames=60, step=None, window_size=None,
                   window_in_seconds=None, feature_toggles=None, kind="image"):
        
        from .helper_vis import view_animation #lazy imports

        start, length = self._vis_prep(start, window_size, window_in_seconds, feature_toggles)

        view_animation(self, start, length, n_frames, length if step is None else step,
                                  feature_toggles, kind)

    #save an animation as a gif
    def save_video(self, path, start=None, n_frames=60, step=None, window_size=None,
                   window_in_seconds=None, feature_toggles=None, kind="image", fps=10):
        
        from .helper_vis import save_animation #lazy imports

        start, length = self._vis_prep(start, window_size, window_in_seconds, feature_toggles)

        save_animation(self, path, start, length, n_frames, length if step is None else step,
                                  feature_toggles, kind, fps)

    @property
    def get_feature_scale(self):
        return self.features.scale

    @property
    def get_feature_weight(self):
        return self.features.weight

    @property
    def get_feature_accum(self):
        return self.features.accum

    @property 
    def src(self):
        return self.source
    
    @property 
    def get_montage(self):
        return self.montage 
    
    @property
    def get_ref_scheme(self):
        return self.ref_scheme 

    @property 
    def get_target_ref(self):
        return self.target_ref
    
    @property 
    def get_img_res(self):
        return self.img_res 

    @property
    def get_margin(self):
        return self.margin

    @property
    def get_buffer_shapes(self):
        return {"M": self.SH_dict['interpol_operator'].shape,
                "Y": self.SH_dict['SH_basis'].shape}

    @property 
    def get_chns(self):
        return self.chn_list
    
    @property 
    def get_e_chns(self):
        return self.excluded_chns

    @property
    def get_default_excluded(self):
        return self.default_excluded  #auxiliary names dropped by the default filter

    @property
    def get_auto_excluded(self):
        return self.auto_excluded  #unresolved names dropped per recording after approval

    @property 
    def get_n_chns(self):
        return self.n_chans
    
    @property 
    def get_timepoints(self):
        return self.time_points

    @property 
    def get_sfreq(self):
        return self.sfreq

    @property 
    def get_L(self):
        return self.L_degree 
    
    @property
    def get_seg_len(self):
        return self.window_size

    @property
    def get_pos_2d(self):
        return self.SH_dict['pos_2d']
    
    @property
    def get_sh_basis(self):
        return self.SH_dict['SH_basis']