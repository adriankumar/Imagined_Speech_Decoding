import json
import traceback
import numpy as np

from .session import Session, _jsonable

#true when a structure survives json.dumps, the no-numpy-leak guard
def _is_jsonable(obj):
    import json
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False

#----------------
#config
#----------------
EDF = r"C:\Users\PC\Documents\Coding_Projects\general experiential learning stuff for ISD and robot\EEG_stuff\datasets\ISD\chisco\subject_1\sub-01_ses-01_task-imagine_run-01_eeg.edf"
MONTAGE = "standard_1005"   #change if the recording resolves against a different standard
REF = "average"
AUTO_EXCLUDE = True         #the chisco recording has unresolved names, auto-exclude clears them on load

#----------------
#tiny test harness, each case isolated so one failure never masks the rest
#----------------
_passed = 0
_failed = 0

def check(name, condition, detail=""):
    global _passed, _failed
    tag = "PASS" if condition else "FAIL"
    if condition:
        _passed += 1
    else:
        _failed += 1
    print(f"  [{tag}] {name}" + (f"  --  {detail}" if detail else ""))

def section(title):
    print(f"\n=== {title} ===")

def run(title, fn, session):
    section(title)
    try:
        fn(session)
    except Exception as e:
        global _failed
        _failed += 1
        print(f"  [ERROR] {title} raised {type(e).__name__}: {e}")
        traceback.print_exc()

#----------------
#empty session: snapshot before any load is the unloaded shell, still json-safe
#----------------
section("empty session")
session = Session()
try:
    snap = session.snapshot()
    check("unloaded snapshot reports loaded false", snap.get("loaded") is False)
    check("unloaded snapshot carries state_version", "state_version" in snap)
    check("unloaded snapshot carries collapsed", "collapsed" in snap)
    json.dumps(snap)
    check("unloaded snapshot json-serialisable", True)
except Exception as e:
    check("unloaded snapshot json-serialisable", False, f"{type(e).__name__}: {e}")

#----------------
#load, kept outside run() since every later test needs the loaded session
#----------------
section("load")
try:
    session.load(source=EDF, montage=MONTAGE, ref_scheme=REF, auto_exclude=AUTO_EXCLUDE)
    env = session.env
    print(f"  loaded: {env.n_chns} channels, L={env.L}, n_modes={env.n_modes}, "
          f"sfreq={env.sfreq}, timepoints={env.timepoints}, window={env.window_size}")
    print(f"  auto-excluded {len(env.auto_ex_chns)}: {env.auto_ex_chns}")
except Exception as e:
    print(f"  [ERROR] load failed, cannot continue: {type(e).__name__}: {e}")
    traceback.print_exc()
    raise SystemExit(1)

#----------------
#snapshot completeness: every nested key the panels read must be present, and the whole thing json-safe
#----------------
#the contract the frontend binds against, nested exactly as the panels destructure it
SNAP_CONTRACT = {
    "loaded": None, "state_version": None, "locked": None, "cursor": None,
    "strip_version": None, "geometry_version": None, "view_version": None, "collapsed": None,
    "n_modes": None,
    "recording": ["source", "montage", "sfreq", "time_points", "duration_s"],
    "config": ["ref_scheme", "target_ref", "L", "L_ceiling", "recommended_L",
               "margin", "img_res", "window_size"],
    "channels": ["resolved", "n_resolved", "default_excluded", "auto_excluded", "manual_excluded"],
    "geometry": ["channel_names", "pos_2d"],
    "shapes": ["M", "Y"],
    "view": ["start", "span"],
    "features": ["toggles", "weights", "n_active"],
}

def test_snapshot_contract(session):
    snap = session.snapshot()
    check("snapshot reports loaded true", snap.get("loaded") is True)

    missing = []
    for key, sub in SNAP_CONTRACT.items():
        if key not in snap:
            missing.append(key)
        elif sub is not None:
            for s in sub:
                if s not in snap[key]:
                    missing.append(f"{key}.{s}")
    check("all contract keys present", not missing, f"missing {missing}" if missing else "")

    #json round-trip proves _jsonable left no numpy scalar or array behind
    try:
        json.dumps(snap)
        check("snapshot json-serialisable, no numpy leak", True)
    except TypeError as e:
        check("snapshot json-serialisable, no numpy leak", False, str(e))

    #spot-check the derived values land where the env reports them
    env = session.env
    check("n_modes matches env", snap["n_modes"] == env.n_modes, f"{snap['n_modes']}")
    check("n_active matches enabled names", snap["features"]["n_active"] == len(env.features.enabled_names()),
          f"{snap['features']['n_active']}")
    check("recommended_L present and <= ceiling",
          snap["config"]["recommended_L"] <= snap["config"]["L_ceiling"],
          f"rec {snap['config']['recommended_L']} ceil {snap['config']['L_ceiling']}")
    check("view span defaults to whole recording", snap["view"]["span"] == env.timepoints,
          f"{snap['view']['span']} vs {env.timepoints}")
    check("cursor mirrors env, not the session", snap["cursor"] == env.window_cursor)

run("snapshot contract", test_snapshot_contract, session)

#----------------
#seek: env owns the clamp and the cursor, at full zoom the view never moves
#----------------
def test_seek(session):
    env = session.env
    v_before = session.view_version
    pos = session.seek(50000)
    check("seek moves the cursor", env.window_cursor == 50000, f"{env.window_cursor}")
    check("seek returns the clamped position", pos == 50000, f"{pos}")
    check("seek at full zoom does not move the view", session.view_version == v_before,
          f"{session.view_version} vs {v_before}")

    #seek past the end clamps to the last whole window
    last = env.timepoints - env.window_size
    pos = session.seek(env.timepoints + 10_000)
    check("seek clamps past-end to last window", pos == last, f"{pos} vs {last}")
    check("cursor clamped in env too", env.window_cursor == last)

    #negative clamps to zero
    pos = session.seek(-500)
    check("seek clamps negative to zero", pos == 0, f"{pos}")
    session.seek(0)

run("seek", test_seek, session)

#----------------
#seek lag coherence: the env primes the lag cache from the previous window, so a jumped-to window carries real lag
#----------------
def test_seek_lag_priming(session):
    env = session.env
    #enable a lag feature so priming is observable
    env.change_feature_toggles({"median_lag": True})

    #cold seek to window 0, no previous window exists, lag must be zero there
    session.seek(0)
    stack0 = env.preview_window_features(start=0)
    names = env.features.enabled_names()
    li = names.index("median_lag")
    check("lag at window 0 is zero (no previous window)", np.allclose(stack0[:, li], 0),
          f"max |lag| = {np.abs(stack0[:, li]).max():.2e}")

    #seek to a later window, the env primes from the window before, so this lag is a real difference
    session.seek(10 * env.window_size)
    stack_n = env.preview_window_features(start=env.window_cursor)
    check("lag at a jumped-to window is non-zero (primed from previous)",
          not np.allclose(stack_n[:, li], 0), f"max |lag| = {np.abs(stack_n[:, li]).max():.2e}")

    env.change_feature_toggles({"median_lag": False})  #restore default toggles
    session.seek(0)

run("seek lag priming", test_seek_lag_priming, session)

#----------------
#zoom: set_view clamps span and start, bumps view_version, a seek outside the span recentres
#----------------
def test_zoom(session):
    env = session.env
    n = env.timepoints

    v_before = session.view_version
    session.set_view(start=0, span=20_000)
    check("view span set", session.view_span == 20_000, f"{session.view_span}")
    check("set_view bumps view_version", session.view_version == v_before + 1)

    #span clamps up to at least one window and down to the whole recording
    session.set_view(start=0, span=10)
    check("span clamps to at least one window", session.view_span == env.window_size,
          f"{session.view_span}")
    session.set_view(start=0, span=n + 1_000_000)
    check("span clamps to whole recording", session.view_span == n, f"{session.view_span}")

    #start clamps so the span never overruns the end
    session.set_view(start=n, span=20_000)
    check("start clamps so span fits", session.view_start == n - 20_000, f"{session.view_start}")

    #zoom in, then a seek outside the span recentres the view on the cursor
    session.set_view(start=0, span=20_000)
    v_pre = session.view_version
    session.seek(500_000)  #well outside [0, 20000)
    check("seek outside the span moves the view", session.view_version > v_pre,
          f"{session.view_version} vs {v_pre}")
    cur = env.window_cursor
    check("recentred view contains the cursor",
          session.view_start <= cur and cur + env.window_size <= session.view_start + session.view_span,
          f"cursor {cur}, view [{session.view_start}, {session.view_start + session.view_span})")

    session.set_view(start=0, span=n)  #restore full zoom
    session.seek(0)

run("zoom", test_zoom, session)

#----------------
#argument edits: geometry_version bumps on L only, margin and img_res rebuild M without bumping it
#----------------
def test_set_argument(session):
    env = session.env

    g0 = session.geometry_version
    s0 = session.state_version
    session.set_argument("margin", 0.8)
    check("margin applied", env.margin == 0.8, f"{env.margin}")
    check("margin bumps state_version", session.state_version == s0 + 1)
    check("margin leaves geometry_version", session.geometry_version == g0)
    session.set_argument("margin", 0.9)

    session.set_argument("img_res", [32, 32])
    check("img_res applied", tuple(env.img_res) == (32, 32), f"{env.img_res}")
    check("img_res leaves geometry_version", session.geometry_version == g0)
    session.set_argument("img_res", [64, 64])

    g1 = session.geometry_version
    session.set_argument("L_degree", 3)
    check("L applied", env.L == 3, f"{env.L}")
    check("L bumps geometry_version", session.geometry_version == g1 + 1)
    session.set_argument("L_degree", 4)

    #feature edits route to the stack by name
    session.set_argument("feature_weight", {"iqr": 2.0})
    check("weight applied by name", env.features.feature_weights["iqr"] == 2.0)
    session.set_argument("feature_weight", {"iqr": 1.0})

    session.set_argument("feature_toggle", {"complexity": False})
    check("toggle applied by name", env.features.feature_toggles["complexity"] is False)
    session.set_argument("feature_toggle", {"complexity": True})

run("argument edits", test_set_argument, session)

#----------------
#window size: timepoints path direct, seconds path converts through sfreq, both reset the cursor
#----------------
def test_window_size(session):
    env = session.env
    session.seek(10 * env.window_size)  #move the cursor so the reset is observable

    session.set_argument("window_size", {"value": 250, "unit": "timepoints"})
    check("window size set in timepoints", env.window_size == 250, f"{env.window_size}")
    check("window change reset the cursor", env.window_cursor == 0)

    session.seek(10 * env.window_size)
    session.set_argument("window_size", {"value": 1.0, "unit": "seconds"})
    check("window size set in seconds via sfreq", env.window_size == int(round(1.0 * env.sfreq)),
          f"{env.window_size} vs {int(round(env.sfreq))}")
    check("seconds change reset the cursor", env.window_cursor == 0)

    session.set_argument("window_size", {"value": 500, "unit": "timepoints"})  #restore

run("window size", test_window_size, session)

#----------------
#channels: exclude accumulates, n_resolved drops, reset restores, the env re-derives the rest
#----------------
def test_channels(session):
    env = session.env
    n0 = env.n_chns
    resolved = list(env.chns_list)
    victims = resolved[:2]  #two known-resolved names to drop

    session.electrode_action("exclude", victims)
    check("manual_excluded accumulates", set(session.manual_excluded) == set(victims),
          f"{session.manual_excluded}")
    check("n_resolved drops by two", env.n_chns == n0 - 2, f"{env.n_chns} vs {n0}")
    check("excluded names gone from resolved", all(v not in env.chns_list for v in victims))
    check("exclude resets the cursor", env.window_cursor == 0)

    #exclude is additive, a second call unions
    more = [c for c in resolved if c not in victims][:1]
    session.electrode_action("exclude", more)
    check("second exclude unions", env.n_chns == n0 - 3, f"{env.n_chns}")

    session.reset_exclusions()
    check("reset clears manual_excluded", session.manual_excluded == [])
    check("reset restores n_resolved", env.n_chns == n0, f"{env.n_chns} vs {n0}")

run("channels", test_channels, session)

#----------------
#reference: a channel list sets target_ref, average restores it, excluding a referenced channel is refused
#----------------
def test_reference(session):
    env = session.env
    resolved = list(env.chns_list)
    ref_chs = resolved[:3]

    session.electrode_action("reference", ref_chs)
    check("target_ref set to channel list", list(env.target_ref) == ref_chs, f"{env.target_ref}")

    #excluding a current reference channel is refused with a clear error
    refused = False
    try:
        session.electrode_action("exclude", [ref_chs[0]])
    except ValueError:
        refused = True
    check("excluding a referenced channel is refused", refused)

    session.electrode_action("average", [])
    check("average restores the reference", env.target_ref == "average", f"{env.target_ref}")
    session.reset_exclusions()

run("reference", test_reference, session)

#----------------
#lock: playback locks edits and seeks, stop unlocks
#----------------
def test_lock(session):
    session.play_start()
    check("play_start locks", session.locked is True)

    blocked_set = False
    try:
        session.set_argument("margin", 0.7)
    except RuntimeError:
        blocked_set = True
    check("set_argument refused while locked", blocked_set)

    blocked_seek = False
    try:
        session.seek(1000)
    except RuntimeError:
        blocked_seek = True
    check("seek refused while locked", blocked_seek)

    session.play_stop()
    check("play_stop unlocks", session.locked is False)
    check("edits work again after stop", (session.set_argument("margin", 0.85) or env.margin == 0.85))
    session.set_argument("margin", 0.9)

run("lock", test_lock, session)

#----------------
#collapsed panels: view-only, survives in the snapshot
#----------------
def test_collapsed(session):
    session.collapsed.add("panel-controls")
    snap = session.snapshot()
    check("collapsed reflected in snapshot", "panel-controls" in snap["collapsed"])
    session.collapsed.discard("panel-controls")
    snap = session.snapshot()
    check("uncollapse reflected in snapshot", "panel-controls" not in snap["collapsed"])

run("collapsed panels", test_collapsed, session)


#----------------
#rendering: pure env -> base64 png, decoded and dimension-checked headless, no flask
#----------------
import io
import base64

#decode a data-url png to (height, width), proving it is a real non-trivial image
def _png_dims(data_url):
    from PIL import Image  #pillow ships with matplotlib's deps, used only in the test
    prefix = "data:image/png;base64,"
    assert data_url.startswith(prefix), "not a png data url"
    raw = base64.b64decode(data_url[len(prefix):])
    img = Image.open(io.BytesIO(raw))
    return img.height, img.width

#write a data-url png next to the test so a failure can be eyeballed, returns the path
def _dump_png(data_url, name):
    import os
    raw = base64.b64decode(data_url[len("data:image/png;base64,"):])
    path = os.path.join(os.path.dirname(__file__), f"_render_{name}.png")
    with open(path, "wb") as f:
        f.write(raw)
    return path

def test_render_feature_image(session):
    from .rendering import render_stack_image
    env = session.env
    env.reset_stream()
    stack = env.advance_window_features()
    names = env.features.enabled_names()

    url = render_stack_image(env, stack, names, kind="image")
    h, w = _png_dims(url)
    check("feature image render is a valid png", h > 0 and w > 0, f"{h}x{w}")
    check("feature image render non-trivial size", h > 100 and w > 100, f"{h}x{w}")
    _dump_png(url, "feature_image")

    url_s = render_stack_image(env, stack, names, kind="stack")
    h, w = _png_dims(url_s)
    check("feature stack render is a valid png", h > 0 and w > 0, f"{h}x{w}")
    _dump_png(url_s, "feature_stack")

run("render feature image", test_render_feature_image, session)

def test_render_strip(session):
    from .rendering import build_strip_png
    env = session.env

    #full-recording strip, the default view on load
    out = build_strip_png(env, 0, env.timepoints, width=1000)
    url, n_ch, width = out if isinstance(out, tuple) else (out, None, None)
    h, w = _png_dims(url)
    check("strip render is a valid png", h > 0 and w > 0, f"{h}x{w}")
    check("strip width tracks requested bins", w in (1000, env.timepoints) or w <= 1000, f"{w}")
    check("strip height tracks channel count", h == env.n_chns, f"{h} vs {env.n_chns}")
    _dump_png(url, "strip_full")

    #a zoomed span, the same primitive reads an arbitrary window of the timeline
    out_z = build_strip_png(env, 0, 20_000, width=1000)
    url_z = out_z[0] if isinstance(out_z, tuple) else out_z
    hz, wz = _png_dims(url_z)
    check("zoomed strip render is a valid png", hz == env.n_chns, f"{hz}x{wz}")
    _dump_png(url_z, "strip_zoom")

run("render strip", test_render_strip, session)

def test_render_decode(session):
    from .rendering import render_decode_png
    env = session.env
    env.reset_stream()
    env.advance_window_features()  #cursor at window 1, a future window exists

    preview = env.preview_decode_at_cursor(delta_coeffs=None)
    names = env.features.enabled_names()
    url = render_decode_png(env, preview, names)
    h, w = _png_dims(url)
    check("decode render is a valid png", h > 0 and w > 0, f"{h}x{w}")
    check("decode render non-trivial size", h > 100 and w > 100, f"{h}x{w}")
    _dump_png(url, "decode")

    #at stream end the forward slots are blank, the renderer must still produce a valid image
    env.seek(env.timepoints - env.window_size)
    preview_end = env.preview_decode_at_cursor()
    url_end = render_decode_png(env, preview_end, names)
    he, we = _png_dims(url_end)
    check("decode render valid at stream end (blank forward slots)", he > 0 and we > 0, f"{he}x{we}")
    _dump_png(url_end, "decode_end")
    env.reset_stream()

run("render decode", test_render_decode, session)

#----------------
#frame assembly: vis + decode payloads, computed together, the decode panel always populated
#----------------
def test_frame_image(session):
    session.env.reset_stream()
    f = session.frame(vis_mode="image")
    check("frame has vis and decode", "vis" in f and "decode" in f)
    check("image vis carries a render url", f["vis"]["mode"] == "image" and "render_url" in f["vis"])
    check("decode carries a render url", "render_url" in f["decode"])
    check("decode residuals present mid-stream", f["decode"]["electrode_residual"] is not None)
    check("decode not at end mid-stream", f["decode"]["at_end"] is False)

def test_frame_stack(session):
    f = session.frame(vis_mode="stack")
    check("stack vis carries a render url", f["vis"]["mode"] == "stack" and "render_url" in f["vis"])

def test_frame_harmonic(session):
    env = session.env
    f = session.frame(vis_mode="harmonic")
    v = f["vis"]
    F = v["stack"].shape[1] if hasattr(v["stack"], "shape") else len(v["stack"][0])
    check("harmonic vis carries coeffs", v["mode"] == "harmonic" and "coeffs" in v)
    #coeffs are (n_modes, F), residual is per-feature, names match the active set
    coeffs = np.asarray(v["coeffs"])
    check("harmonic coeffs shape (n_modes, F)", coeffs.shape == (env.n_modes, len(v["names"])),
          f"{coeffs.shape}")
    check("harmonic residual is per-feature", len(np.asarray(v["residual"])) == len(v["names"]))
    check("harmonic vis json-serialisable", _is_jsonable(_jsonable(v)))

run("frame image", test_frame_image, session)
run("frame stack", test_frame_stack, session)
run("frame harmonic", test_frame_harmonic, session)

#----------------
#frame at stream end: decode forward fields blank, vis still renders the current window
#----------------
def test_frame_at_end(session):
    env = session.env
    env.seek(env.timepoints - env.window_size)
    f = session.frame(vis_mode="image")
    check("vis renders at stream end", "render_url" in f["vis"])
    check("decode reports at_end", f["decode"]["at_end"] is True)
    check("decode residuals blank at end", f["decode"]["electrode_residual"] is None
          and f["decode"]["image_residual"] is None)
    env.reset_stream()

run("frame at stream end", test_frame_at_end, session)

#----------------
#playback frame: builds at the current window then advances, cursor steps by one window, lag stays coherent
#----------------
def test_play_advance(session):
    env = session.env
    session.seek(10 * env.window_size)
    session.play_start()
    check("play_start locks", session.locked is True)
    start = env.window_cursor

    d = session.play_advance(vis_mode="image")
    check("play frame has vis decode cursor at_end", all(k in d for k in ("vis", "decode", "cursor", "at_end")))
    check("cursor advanced by one window", d["cursor"] == start + env.window_size,
          f"{d['cursor']} vs {start + env.window_size}")
    check("play frame vis renders", "render_url" in d["vis"])
    check("play frame decode renders", "render_url" in d["decode"])

    #a second advance steps again from the new cursor
    d2 = session.play_advance(vis_mode="image")
    check("second advance steps again", d2["cursor"] == d["cursor"] + env.window_size,
          f"{d2['cursor']}")

    session.play_stop()
    check("play_stop unlocks", session.locked is False)
    check("play_stop leaves cursor where it paused", env.window_cursor == d2["cursor"],
          f"{env.window_cursor} vs {d2['cursor']}")

run("play advance", test_play_advance, session)

#----------------
#playback to the end: the last window builds with a blank decode, at_end unlocks
#----------------
def test_play_to_end(session):
    env = session.env
    #park on the last window, one advance should hit the end
    env.seek(env.timepoints - env.window_size)
    session.play_start()
    d = session.play_advance(vis_mode="image")
    check("advance at last window flags at_end", d["at_end"] is True)
    check("at_end unlocked the session", session.locked is False)
    env.reset_stream()

run("play to end", test_play_to_end, session)

#----------------
#strip over the zoom window: cached, rebuilt only when the view moves
#----------------
def test_strip_payload(session):
    session.set_view(start=0, span=session.env.timepoints)  #full zoom
    s1 = session.strip()
    check("strip carries raster and bounds", all(k in s1 for k in
          ("raster_url", "n", "width", "view_start", "view_span")))
    check("strip height is channel count", s1["n"] == session.env.n_chns, f"{s1['n']}")
    check("strip raster is a png data url", s1["raster_url"].startswith("data:image/png;base64,"))

    s2 = session.strip()
    check("strip is cached between calls", s2 is s1)

    #zooming invalidates the cache and re-rasters a different span
    session.set_view(start=0, span=20_000)
    s3 = session.strip()
    check("zoom rebuilds the strip", s3 is not s1)
    check("zoomed strip records the new span", s3["view_span"] == 20_000, f"{s3['view_span']}")
    session.set_view(start=0, span=session.env.timepoints)

run("strip payload", test_strip_payload, session)

#----------------
#summary
#----------------
section("summary")
print(f"  {_passed} passed, {_failed} failed")