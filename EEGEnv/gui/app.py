import threading
import webview
import numpy as np
from flask import Flask, jsonify, render_template, request

from ..mod import EEGEnv, MNE_MONTAGES, UnresolvedChannelsError

app = Flask(__name__)

#convert numpy scalars and arrays to plain python types so flask's json encoder accepts the snapshot
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

#single env session, holds the one env instance and the gui-only state the snapshot reports
class Session:
    def __init__(self):
        self.env = None          #none until a recording is loaded
        self.state_version = 0   #bumped on every committed change, polled by the frontend
        self.cursor = 0          #current window start, advanced during playback
        self.locked = False      #true while playback runs, controls are read-only then
        self.collapsed = set()   #panel ids the user has collapsed, view-only, survives a reload
        self.strip_version = 0   #bumped on load and retarget so the strip refetches its raster
        self.strip_cache = None  #cached raster png, rebuilt only when strip_version changes
        self.play_step = None    #playback step in samples, set on play, defaults to the window size
        self.auto_excluded = []    #unresolvable names approved at load, the env loses this across a re-resolve
        self.manual_excluded = []  #resolved names the user chose to drop, reset clears these
        self.geometry_version = 0  #bumped when the basis Y changes, so the sh panel refetches it

    #bump the version so the polling frontend sees a change
    def _bump(self):
        self.state_version += 1
        

    #construct a fresh env, the caller catches UnresolvedChannelsError to drive the popup
    def load(self, source, montage, ref_scheme, auto_exclude):
        self.env = EEGEnv(source=source, montage=montage, ref_scheme=ref_scheme,
                          auto_exclude=auto_exclude)
        self.cursor = 0
        self.locked = False
        self.strip_cache = None
        self.auto_excluded = list(self.env.get_auto_excluded)
        self.manual_excluded = []
        self.strip_version += 1
        self.geometry_version += 1
        self._bump()


    #render the current window as a base64 png, kind 'image' or 'stack', read-only against the live stream
    def render(self, kind="image", start=None, length=None):
        from .rendering import render_feature_window #lazy import
        start = self.cursor if start is None else int(start)
        length = self.env.get_seg_len if length is None else int(length)
        return render_feature_window(self.env, start, length, kind=kind)
    
    #single dispatch for a named argument edit, maps the name to the matching env change method
    #refused while locked so playback cannot be disturbed mid-stream
    def set_argument(self, name, value):
        if self.locked:
            raise RuntimeError("controls are locked during playback")
        env = self.env
        if name == "feature_toggle":
            env.change_feature_toggle(value)            #{feature_name: bool}
        elif name == "feature_weight":
            env.change_feature_weight(value)            #{feature_name: float}
        elif name == "feature_accum":
            env.change_feature_accum(accum=value["accum"], alpha=value.get("alpha"))
        elif name == "L_degree":
            env.change_L(int(value))
            self.geometry_version += 1
        elif name == "margin":
            env.change_margin(float(value))
        else:
            raise ValueError(f"unknown argument: {name}")
        self._bump()
    
    #move the cursor to a window start, clamped to the recording, refused while locked
    def seek(self, start):
        if self.locked:
            raise RuntimeError("seeking is disabled during playback")
        length = self.env.get_seg_len
        self.cursor = max(0, min(int(start), self.env.get_timepoints - length))
        self._bump()
        return self.cursor

    #build the recording raster once and cache it, rebuilt only when strip_version changes
    def strip(self):
        if self.strip_cache is None:
            from .rendering import build_strip_png #lazy import
            url, n, width = build_strip_png(self.env)
            self.strip_cache = {"raster_url": url, "n": n, "width": width}
        return self.strip_cache

    #start playback from the cursor, resets lag and ema so they build cleanly, locks the controls
    def play_start(self, step=None):
        self.play_step = int(step) if step else self.env.get_seg_len
        self.env.features.reset()
        self.locked = True
        self._bump()

    #assemble one frame from a single stack: the feature image plus the cheap sh projection data
    def build_frame(self, stack, names, kind):
        from .rendering import render_stack_image
        url = render_stack_image(self.env, stack, names, kind)
        coeffs, residual = self.env.project_coefficients(stack)
        return {
            "render_url": url,
            "sh": _jsonable({"coeffs": coeffs, "residual": residual, "stack": stack, "names": names}),
        }

    #seek/load frame: a read-only peek at the cursor, lag primed, drives both panels
    def frame(self, kind="image"):
        stack, names = self.env.peek_stack(self.cursor, self.env.get_seg_len)
        return self.build_frame(stack, names, kind)

    #playback frame: advance the live stream once, build from that one stack, then move the cursor
    def play_advance(self, kind="image"):
        env = self.env
        length = env.get_seg_len
        stack, names = env.advance_stack(self.cursor, length)
        frame = self.build_frame(stack, names, kind)
        nxt = self.cursor + self.play_step
        at_end = nxt + length > env.get_timepoints
        if at_end:
            self.locked = False
            self._bump()
        else:
            self.cursor = int(nxt)
        frame["cursor"] = int(self.cursor)
        frame["at_end"] = bool(at_end)
        return frame

    #stop playback, clear the stream state, unlock the controls
    def play_stop(self):
        self.locked = False
        self.env.features.reset()
        self._bump()

    #apply an electrode selection from the popup, all paths re-resolve through change_source like the env expects
    def electrode_action(self, action, channels):
        env = self.env
        if action == "reset":
            self.reset_exclusions()
            return
        if action == "exclude":
            ref = env.get_target_ref
            if isinstance(ref, (list, tuple)):
                clash = sorted(set(channels) & set(ref))
                if clash:
                    raise ValueError(f"cannot exclude {clash}; they are the current reference, "
                                     f"unreference them first, then exclude")
            self.manual_excluded = sorted(set(self.manual_excluded) | set(channels))
            env.change_source(env.src, exclude_chns=self.auto_excluded + self.manual_excluded, auto_exclude=True)
        elif action == "reference":
            env.change_target_ref(channels)
        elif action == "average":
            env.change_target_ref("average")
        else:
            raise ValueError(f"unknown electrode action: {action}")
        self._after_channel_change()

    #reset manual exclusions back to only the auto-excluded set, restoring the dropped resolved channels
    def reset_exclusions(self):
            self.manual_excluded = []
            self.env.change_source(self.env.src, exclude_chns=self.auto_excluded, auto_exclude=True)
            self._after_channel_change()

    #shared aftermath of a channel or reference edit: the buffers rebuilt, so the strip and cursor resync
    def _after_channel_change(self):
        self.cursor = 0
        self.strip_cache = None
        self.strip_version += 1
        self.geometry_version += 1
        self._bump()

    #simulate a model coefficient delta on the current window, random directions scaled per feature
    #magnitude tracks each feature's own coefficient rms so every feature shows a comparable change
    def decode_simulation(self, scale=1.0, seed=None):
        from .rendering import render_decode_png
        env = self.env
        stack, names = env.peek_stack(self.cursor, env.get_seg_len)
        coeffs, _ = env.project_coefficients(stack)
        rms = np.sqrt((coeffs ** 2).mean(axis=0))
        rms = np.where(rms > 0, rms, 1.0)
        rng = np.random.default_rng(seed)
        delta_coeffs = scale * rng.standard_normal(coeffs.shape) * rms[None, :]
        before, delta_image, after = env.decode_delta(stack, delta_coeffs)
        return {"render_url": render_decode_png(env, before, delta_image, after, names)}

    #full application state as plain json, assembled from the env accessors only
    def snapshot(self):
        if self.env is None:
            return {"loaded": False, "state_version": self.state_version,
                    "collapsed": sorted(self.collapsed)}

        env = self.env
        shapes = env.get_buffer_shapes
        return _jsonable({
            "loaded": True,
            "state_version": self.state_version,
            "locked": self.locked,
            "cursor": self.cursor,
            "strip_version": self.strip_version,
            "geometry_version": self.geometry_version,
            "collapsed": sorted(self.collapsed),
            "recording": {
                "source": env.src,
                "montage": env.get_montage,
                "sfreq": env.get_sfreq,
                "time_points": env.get_timepoints,
                "duration_s": env.get_timepoints / env.get_sfreq,
            },
            "config": {
                "ref_scheme": env.get_ref_scheme,
                "target_ref": env.get_target_ref,
                "L": env.get_L,
                "margin": env.get_margin,
                "L_ceiling": int(np.floor(np.sqrt(env.get_n_chns)) - 1),
                "img_res": list(env.get_img_res),
                "window_size": env.get_seg_len,
            },
            "channels": {
                "resolved": env.get_chns,
                "n_resolved": env.get_n_chns,
                "default_excluded": env.get_default_excluded,
                "auto_excluded": self.auto_excluded,
                "manual_excluded": self.manual_excluded,
            },
            "geometry": {
                "channel_names": env.get_chns,
                "pos_2d": env.get_pos_2d,
            },
            "shapes": {"M": list(shapes["M"]), "Y": list(shapes["Y"])},
            "features": {
                "toggle": dict(env.features.feature_toggle),
                "weight": dict(env.features.weight),
                "accum": dict(env.features.accum),
                "alpha": dict(env.features.alpha),
            }
        })

session = Session()

@app.route("/")
def index():
    return render_template("index.html")


#full snapshot for the polling frontend
@app.route("/state")
def state():
    return jsonify(session.snapshot())


#the montage list for the loading dropdown, straight from the env's montage registry
@app.route("/montages")
def montages():
    return jsonify(MNE_MONTAGES)


#load attempt, returns ok with a snapshot, an unresolved list for the popup, or an error message
@app.route("/load", methods=["POST"])
def load():
    data = request.get_json()
    source = data.get("source")
    montage = data.get("montage")
    ref_scheme = (data.get("ref_scheme") or "average").strip()
    auto_exclude = bool(data.get("auto_exclude", False))

    try:
        session.load(source, montage, ref_scheme, auto_exclude)
    except UnresolvedChannelsError as e:
        return jsonify({"ok": False, "kind": "unresolved",
                        "unresolved": e.unresolved, "montage": e.montage})
    except Exception as e:
        return jsonify({"ok": False, "kind": "error", "message": str(e)})

    return jsonify({"ok": True, "snapshot": session.snapshot()})

#native edf dialog opened on the server, the frontend posts here and reads back the chosen path
#opening the dialog through a route avoids passing a js_api object into the window, which is what triggers the accessibility recursion on the webview2 backend
@app.route("/pick_edf", methods=["POST"])
def pick_edf():
    windows = webview.windows
    if not windows:
        return jsonify({"path": None})
    result = windows[0].create_file_dialog(
        webview.FileDialog.OPEN, allow_multiple=False,
        file_types=("EDF files (*.edf)", "All files (*.*)"))
    return jsonify({"path": result[0] if result else None})

#seek/load frame at the cursor, body {kind}, returns the feature image and the sh data for both panels
@app.route("/frame", methods=["POST"])
def frame():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(session.frame(kind=body.get("kind", "image")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#apply a single named control edit, body {name, value}, returns the new snapshot
@app.route("/set", methods=["POST"])
def set_argument():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    if name is None:
        return jsonify({"error": "name is required"}), 400
    try:
        session.set_argument(name, body.get("value"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(session.snapshot())

#the recording raster, built once and cached, the strip panel refetches on a new strip_version
@app.route("/strip", methods=["GET"])
def strip():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    try:
        return jsonify(session.strip())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#move the window to a start sample, body {start}, returns the snapshot so the feature panel re-renders
@app.route("/seek", methods=["POST"])
def seek():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    try:
        session.seek(body.get("start", 0))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(session.snapshot())

#persist a panel's collapsed state on the session so a reload restores the layout, body {panel, collapsed}
@app.route("/toggle_panel", methods=["POST"])
def toggle_panel():
    body = request.get_json(silent=True) or {}
    panel = body.get("panel")
    if not panel:
        return jsonify({"error": "panel is required"}), 400
    if body.get("collapsed"):
        session.collapsed.add(panel)
    else:
        session.collapsed.discard(panel)
    return jsonify({"ok": True})

#begin playback, resets the stream and locks controls, returns the snapshot
@app.route("/play_start", methods=["POST"])
def play_start():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    session.play_start(step=body.get("step"))
    return jsonify(session.snapshot())

#advance one playback frame, returns the feature image, sh data, the new cursor, and the end flag
@app.route("/play_step", methods=["POST"])
def play_step():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(session.play_advance(kind=body.get("kind", "image")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#stop playback, returns the snapshot so the panels unlock
@app.route("/play_stop", methods=["POST"])
def play_stop():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    session.play_stop()
    return jsonify(session.snapshot())

#serve the electrode selector popup page, the context picks which action buttons it shows
@app.route("/electrode_widget")
def electrode_widget_page():
    return render_template("electrode_widget.html", context=request.args.get("context", "exclude"))

#open the selector as its own pywebview window on the same server
@app.route("/open_electrode_selector", methods=["POST"])
def open_electrode_selector():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    from . import electrode_widget as ew
    ew.open_selector(host="127.0.0.1", port=5000, action_context=(request.get_json(silent=True) or {}).get("context", "exclude"))
    return jsonify({"ok": True})

#receive a committed selection from the popup, apply it, close the popup from python, return the snapshot
@app.route("/electrode_action", methods=["POST"])
def electrode_action():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    try:
        session.electrode_action(body.get("action"), body.get("channels", []))
    except Exception as e:
            return jsonify({"error_type": "generic", "message": str(e)}), 400
    from . import electrode_widget as ew
    ew.close_selector()
    return jsonify(session.snapshot())

#reset manual exclusions, restoring the dropped resolved channels
@app.route("/reset_exclusions", methods=["POST"])
def reset_exclusions():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    try:
        session.reset_exclusions()
    except Exception as e:
            return jsonify({"error_type": "generic", "message": str(e)}), 400
    return jsonify(session.snapshot())

#the harmonic basis transpose for the matrix view, fetched once per geometry_version
@app.route("/sh_basis", methods=["GET"])
def sh_basis():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    Y = session.env.get_sh_basis
    return jsonify(_jsonable({"YT": Y.T, "n_modes": int(Y.shape[0]), "n_channels": int(Y.shape[1])}))

#decode simulation on the current window, body {scale, seed}, returns the before/delta/after render
@app.route("/decode_sim", methods=["POST"])
def decode_sim():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(session.decode_simulation(scale=float(body.get("scale", 1.0)), seed=body.get("seed")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#flask in a daemon thread, the pywebview window on the main thread, no js_api bridge
def main():
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False),
        daemon=True).start()
    
    webview.create_window("eeg env diagnostic", "http://127.0.0.1:5000", width=1280, height=820)
    webview.start()


if __name__ == "__main__":
    main()