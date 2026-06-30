import numpy as np

from ..mod import EEGEnv, MNE_MONTAGES

#convert numpy scalars and arrays to plain python so flask's json encoder accepts the payload
def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj

#single env session, owns the one env instance plus the view-only state the snapshot reports
#the env owns the cursor and all signal processing, the session owns versions, locks, zoom, exclusion bookkeeping
class Session:
    def __init__(self):
        self.env = None
        self.state_version = 0     #bumped on every committed change, polled by the frontend
        self.locked = False        #true while playback runs, controls read-only then
        self.collapsed = set()     #panel ids the user has collapsed, view-only, survives a reload
        self.strip_version = 0     #bumped on load and re-resolve so the strip refetches its raster
        self.strip_cache = None    #cached raster, rebuilt only when the view or channels change
        self.geometry_version = 0  #bumped when the basis Y changes, so the harmonic view refetches it
        self.view_version = 0      #bumped when the zoom window moves, so the strip re-rasters its span
        self.view_start = 0        #zoom window start in samples
        self.view_span = 0         #zoom window length in samples, whole recording on load
        self.manual_excluded = []  #resolved names the user chose to drop, the accumulator, reset clears it

    def _bump(self):
        self.state_version += 1

    #=====
    #view, the zoom window over the recording timeline
    #====
    #reset the view to the whole recording, the default span on load and re-resolve
    def _reset_view(self):
        self.view_start = 0
        self.view_span = self.env.timepoints
        self.view_version += 1
        self.strip_cache = None

    #keep the current window visible, recentre the view on the cursor only when it falls outside the span
    def _ensure_visible(self, cursor):
        ws = self.env.window_size
        n = self.env.timepoints
        if cursor < self.view_start or cursor + ws > self.view_start + self.view_span:
            self.view_start = max(0, min(cursor - self.view_span // 2, n - self.view_span))
            self.view_version += 1
            self.strip_cache = None

    #set the zoom span and re-raster, span clamped to at least one window and at most the whole recording
    def set_view(self, start, span):
        n = self.env.timepoints
        self.view_span = max(self.env.window_size, min(int(span), n))
        self.view_start = max(0, min(int(start), n - self.view_span))
        self.view_version += 1
        self.strip_cache = None
        self._bump()

    #=====
    #load, the only constructor of the env
    #====
    #construct a fresh env, the route inspects auto_ex_chns to decide whether to surface the unresolved modal
    def load(self, source, montage, ref_scheme, auto_exclude):
        self.env = EEGEnv(current_src=source, current_montage=montage,
                          current_ref_scheme=ref_scheme, auto_exclude=auto_exclude)
        self.locked = False
        self.manual_excluded = []
        self.strip_cache = None
        self.strip_version += 1
        self.geometry_version += 1
        self._reset_view()
        self._bump()

    #=====
    #argument edits, single dispatch, refused while locked so playback is never disturbed
    #====
    def set_argument(self, name, value):
        if self.locked:
            raise RuntimeError("controls are locked during playback")
        env = self.env
        if name == "feature_toggle":
            env.change_feature_toggles(value)            #{name: bool}
        elif name == "feature_weight":
            env.change_feature_weights(value)            #{name: float}
        elif name == "L_degree":
            env.change_L(int(value))
            self.geometry_version += 1                   #Y rebuilt, the harmonic view refetches
        elif name == "margin":
            env.change_margin(float(value))              #M rebuilt, Y untouched
        elif name == "img_res":
            env.change_img_res(tuple(value))             #M rebuilt at a new resolution
        elif name == "window_size":
            self._set_window_size(value)
        else:
            raise ValueError(f"unknown argument: {name}")
        self._bump()

    #window size from the panel, value carries the unit so seconds route through the conversion
    #either path resets the stream inside the env, the view recentres on the reset cursor
    def _set_window_size(self, value):
        if value.get("unit") == "seconds":
            self.env.change_window_size_seconds(float(value["value"]))
        else:
            self.env.change_window_size(int(value["value"]))
        self._ensure_visible(self.env.window_cursor)

    #=====
    #cursor, owned by the env, the session only drives and recentres the view
    #====
    def seek(self, start):
        if self.locked:
            raise RuntimeError("seeking is disabled during playback")
        pos = self.env.seek(start)        #env clamps, sets the cursor, primes the lag cache
        self._ensure_visible(pos)
        self._bump()
        return pos

    #=====
    #playback, advances the live stream one window per frame, lag chains naturally
    #====
    #re-prime the lag from the window before the cursor so the first played window carries correct lag, then lock
    def play_start(self):
        self.env.seek(self.env.window_cursor)
        self.locked = True
        self._bump()

    #unlock and leave the cursor where playback paused, a later seek or edit re-primes the lag
    def play_stop(self):
        self.locked = False
        self._bump()

    #=====
    #frame assembly, the vis payload for the active mode plus the decode payload, computed together
    #====
    #the visualisation payload for one mode, only the active mode's path is computed
    #image/stack/raw/operator render server-side to a png, harmonic ships the per-window arrays for the canvas
    def _vis_payload(self, stack, names, mode):
        from .rendering import render_stack_image, render_raw_computed, render_operator
        if mode in ("image", "stack"):
            return {"mode": mode, "render_url": render_stack_image(self.env, stack, names, mode)}
        if mode == "raw":
            return {"mode": mode,
                    "render_url": render_raw_computed(self.env, self.env.get_raw_window(), stack, names)}
        if mode == "operator":
            return {"mode": mode, "render_url": render_operator(self.env)}
        if mode == "harmonic":
            return {"mode": mode,
                    "coeffs": self.env.to_sh_compression(stack),
                    "residual": self.env.harmonic_residual(stack),
                    "stack": stack,
                    "names": names}
        raise ValueError(f"unknown vis mode: {mode}")

    #the decode payload from a preview, field mode renders the four-column image, matrix mode renders the
    #three value columns and ships the (L+1)^2 x F delta matrix for the canvas, residuals blank at stream end
    def _decode_payload(self, preview, names, delta_mode="field"):
        from .rendering import render_decode_png
        payload = {
            "delta_mode": delta_mode,
            "names": names,
            "electrode_residual": preview.electrode_residual,   #none at stream end
            "image_residual": preview.image_residual,           #none at stream end
            "at_end": preview.target_stack is None,
        }
        if delta_mode == "matrix":
            payload["render_url"] = render_decode_png(self.env, preview, names, include_delta=False)
            payload["delta_coeffs"] = preview.delta_coeffs      #none at stream end
        else:
            payload["render_url"] = render_decode_png(self.env, preview, names, include_delta=True)
        return payload

    #one decode preview at the cursor drives both panels, its current stack is reused as the vis stack
    def _build_frame(self, vis_mode, delta_mode):
        env = self.env
        names = env.features.enabled_names()
        preview = env.preview_decode_at_cursor(None)        #true delta floor at the cursor
        vis = self._vis_payload(preview.current_stack, names, vis_mode)
        decode = self._decode_payload(preview, names, delta_mode)
        return vis, decode

    #read-only frame at the cursor, used on load, seek, and edits, advances nothing
    def frame(self, vis_mode="image", delta_mode="field"):
        vis, decode = self._build_frame(vis_mode, delta_mode)
        return {"vis": vis, "decode": decode}

    #playback frame, builds at the current window then advances the live stream one window for lag continuity
    def play_advance(self, vis_mode="image", delta_mode="field"):
        vis, decode = self._build_frame(vis_mode, delta_mode)
        self.env.advance_window_features()
        cursor = self.env.window_cursor
        at_end = self.env.at_stream_end
        if at_end:
            self.locked = False
            self._bump()
        return {"vis": vis, "decode": decode, "cursor": int(cursor), "at_end": bool(at_end)}

    #=====
    #strip, the recording raster over the current zoom window, cached until the view or channels change
    #====
    def strip(self):
        if self.strip_cache is None:
            from .rendering import build_strip_png
            stop = self.view_start + self.view_span
            url, n, width = build_strip_png(self.env, self.view_start, stop, width=1000)
            self.strip_cache = {"raster_url": url, "n": int(n), "width": int(width),
                                "view_start": int(self.view_start), "view_span": int(self.view_span)}
        return self.strip_cache

    #=====
    #channels, exclusion is the manual accumulator, the unresolved set is re-derived by the env each re-resolve
    #====
    def electrode_action(self, action, channels):
        env = self.env
        if action == "reset":
            self.reset_exclusions()
            return
        if action == "exclude":
            ref = env.target_ref
            if isinstance(ref, (list, tuple)):
                clash = sorted(set(channels) & set(ref))
                if clash:
                    raise ValueError(f"cannot exclude {clash}; they are the current target re-reference, "
                                     f"unreference them first, then exclude")
            self.manual_excluded = sorted(set(self.manual_excluded) | set(channels))
            env.change_excluded_chns(self.manual_excluded)
        elif action == "reference":
            env.change_target_ref(channels)
        elif action == "average":
            env.change_target_ref("average")
        else:
            raise ValueError(f"unknown electrode action: {action}")
        self._after_channel_change()

    #restore the dropped resolved channels by clearing the manual set
    def reset_exclusions(self):
        self.manual_excluded = []
        self.env.change_excluded_chns([])
        self._after_channel_change()

    #shared aftermath of a channel or reference edit, the env already reset the stream, the view and strip resync
    def _after_channel_change(self):
        self.strip_cache = None
        self.strip_version += 1
        self.geometry_version += 1
        self._reset_view()
        self._bump()

    #=====
    #snapshot, full application state as plain json, env attributes read straight through
    #====
    def snapshot(self):
        if self.env is None:
            return {"loaded": False, "state_version": self.state_version,
                    "collapsed": sorted(self.collapsed)}

        env = self.env
        return _jsonable({
            "loaded": True,
            "state_version": self.state_version,
            "locked": self.locked,
            "cursor": env.window_cursor,
            "strip_version": self.strip_version,
            "geometry_version": self.geometry_version,
            "view_version": self.view_version,
            "collapsed": sorted(self.collapsed),
            "recording": {
                "source": env.src,
                "montage": env.montage,
                "sfreq": env.sfreq,
                "time_points": env.timepoints,
                "duration_s": env.timepoints / env.sfreq,
            },
            "config": {
                "ref_scheme": env.recording_ref,
                "target_ref": env.target_ref,
                "L": env.L,
                "L_ceiling": env.L_ceiling,
                "recommended_L": env.recommended_L,
                "margin": env.margin,
                "img_res": list(env.img_res),
                "window_size": env.window_size,
            },
            "channels": {
                "resolved": env.chns_list,
                "n_resolved": env.n_chns,
                "default_excluded": env.default_ex_chns,
                "auto_excluded": env.auto_ex_chns,
                "manual_excluded": self.manual_excluded,
            },
            "geometry": {
                "channel_names": env.chns_list,
                "pos_2d": env.electrode_pos_2d,
            },
            "shapes": {"M": list(env.M.shape), "Y": list(env.Y.shape)},
            "view": {"start": self.view_start, "span": self.view_span},
            "features": {
                "toggles": dict(env.features.feature_toggles),
                "weights": dict(env.features.feature_weights),
                "n_active": len(env.features.enabled_names()),
            },
            "n_modes": env.n_modes,
        })