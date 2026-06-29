import threading
import webview
from flask import Flask, jsonify, render_template, request

from ..mod import MNE_MONTAGES
from .session import Session, _jsonable

app = Flask(__name__)
session = Session()

#====================================================================
#pages
#====================================================================
@app.route("/")
def index():
    return render_template("index.html")

#serve the electrode selector popup page, the context picks which action buttons it shows
@app.route("/electrode_widget")
def electrode_widget_page():
    return render_template("electrode_widget.html", context=request.args.get("context", "exclude"))

#====================================================================
#state
#====================================================================
#full snapshot for the polling frontend
@app.route("/state")
def state():
    return jsonify(session.snapshot())

#the montage list for the loading dropdown, straight from the env's montage registry
@app.route("/montages")
def montages():
    return jsonify(MNE_MONTAGES)

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

#====================================================================
#load
#====================================================================
#native edf dialog opened on the server, the frontend posts here and reads back the chosen path
#opening the dialog through a route avoids passing a js_api object into the window, which triggers the webview2 recursion
@app.route("/pick_edf", methods=["POST"])
def pick_edf():
    windows = webview.windows
    if not windows:
        return jsonify({"path": None})
    result = windows[0].create_file_dialog(
        webview.FileDialog.OPEN, allow_multiple=False,
        file_types=("EDF files (*.edf)", "All files (*.*)"))
    return jsonify({"path": result[0] if result else None})

#load attempt, the env drops unresolved channels and exposes them
#if the user did not pre-approve, surface the unresolved modal, otherwise return the snapshot
@app.route("/load", methods=["POST"])
def load():
    data = request.get_json()
    source = data.get("source")
    montage = data.get("montage")
    ref_scheme = (data.get("ref_scheme") or "average").strip()
    auto_exclude = bool(data.get("auto_exclude", False))

    try:
        session.load(source, montage, ref_scheme, auto_exclude)
    except Exception as e:
        return jsonify({"ok": False, "kind": "error", "message": str(e)})

    unresolved = session.env.auto_ex_chns
    if unresolved and not auto_exclude:
        return jsonify({"ok": False, "kind": "unresolved",
                        "unresolved": unresolved, "montage": montage})

    return jsonify({"ok": True, "snapshot": session.snapshot()})

#====================================================================
#frame, the combined vis and decode payload for both panels
#====================================================================
#read-only frame at the cursor, body {vis_mode}, returns the vis payload and the decode payload
@app.route("/frame", methods=["POST"])
def frame():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(_jsonable(session.frame(vis_mode=body.get("vis_mode", "image"))))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#the harmonic basis transpose for the system view, fetched once per geometry_version, static per geometry
@app.route("/sh_basis", methods=["GET"])
def sh_basis():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    Y = session.env.Y
    return jsonify(_jsonable({"YT": Y.T, "n_modes": int(Y.shape[0]), "n_channels": int(Y.shape[1])}))

#====================================================================
#edits
#====================================================================
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

#====================================================================
#strip and view
#====================================================================
#the recording raster over the current zoom window, cached, refetched on a new strip_version or view_version
@app.route("/strip", methods=["GET"])
def strip():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    try:
        return jsonify(session.strip())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#move the window to a start sample, body {start}, returns the snapshot so the panels re-render
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

#set the zoom window, body {start, span}, re-rasters the strip on the next fetch, returns the snapshot
@app.route("/zoom", methods=["POST"])
def zoom():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    try:
        session.set_view(body.get("start", 0), body.get("span", session.env.timepoints))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(session.snapshot())

#====================================================================
#playback
#====================================================================
#begin playback, re-primes the lag and locks controls, returns the snapshot
@app.route("/play_start", methods=["POST"])
def play_start():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    session.play_start()
    return jsonify(session.snapshot())

#advance one playback frame, body {vis_mode}, returns the vis and decode payloads, the cursor, and the end flag
@app.route("/play_step", methods=["POST"])
def play_step():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(_jsonable(session.play_advance(vis_mode=body.get("vis_mode", "image"))))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#stop playback, returns the snapshot so the panels unlock
@app.route("/play_stop", methods=["POST"])
def play_stop():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    session.play_stop()
    return jsonify(session.snapshot())

#====================================================================
#channels
#====================================================================
#open the selector as its own pywebview window on the same server
@app.route("/open_electrode_selector", methods=["POST"])
def open_electrode_selector():
    if session.env is None:
        return jsonify({"error": "no source loaded"}), 400
    from . import electrode_widget as ew
    ew.open_selector(host="127.0.0.1", port=5000,
                     action_context=(request.get_json(silent=True) or {}).get("context", "exclude"))
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

#====================================================================
#entry, flask in a daemon thread, the pywebview window on the main thread, no js_api bridge
#====================================================================
def main():
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False),
        daemon=True).start()

    webview.create_window("eeg env diagnostic", "http://127.0.0.1:5000", width=1280, height=820)
    webview.start()


if __name__ == "__main__":
    main()