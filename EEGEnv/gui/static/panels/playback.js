//recording strip with playback: a raster of the current zoom window, a draggable playline, a play loop
//the raster covers the view span not the whole recording, so all overlay positions are relative to the view
//zoom and the window-size control re-raster while idle, playback freezes them and only moves the playline
(function(){
  let lastStripV = -1, lastViewV = -1;
  let raster = null;                 //{raster_url, n, width, view_start, view_span}
  let nSamples = 0, segment = 0;     //recording length, window size
  let viewStart = 0, viewSpan = 0;   //the zoom window the raster covers
  let lastCursor = 0, sfreq = 1;
  let locked = false;
  let playing = false, playTimer = null;

  function el(id){ return document.getElementById(id); }

  async function fetchRaster(){ const res = await fetch("/strip"); const d = await res.json(); return d.error ? null : d; }

  //build the strip dom and the controls once per raster, attach the drag handler
  function buildDom(){
    el("pb-body").innerHTML = `
      <div class="pb-controls">
        <button id="pb-play" class="seg">play</button>
        <span class="pb-zoom">
          <button id="pb-zin" class="seg">zoom in</button>
          <button id="pb-zout" class="seg">zoom out</button>
          <button id="pb-fit" class="seg">fit</button>
        </span>
        <span class="pb-ws-ctrl">
          <span class="pb-label">window_size</span>
          <input type="number" id="pb-ws" min="1" step="1" value="${segment}">
          <select id="pb-ws-unit">
            <option value="timepoints">samples</option>
            <option value="seconds">seconds</option>
          </select>
          <button id="pb-ws-set" class="seg">set</button>
          <span id="pb-ws-status" class="pb-ws-status"></span>
        </span>
      </div>
      <div class="strip-wrap">
        <img class="strip-raster" src="${raster.raster_url}" alt="recording">
        <div class="strip-window"></div>
        <div class="strip-line"></div>
      </div>
      <div class="strip-meta"><span id="strip-pos"></span><span id="strip-span"></span></div>`;
    attachDrag(el("pb-body").querySelector(".strip-wrap"));
    el("pb-play").addEventListener("click", togglePlay);
    el("pb-zin").addEventListener("click", zoomIn);
    el("pb-zout").addEventListener("click", zoomOut);
    el("pb-fit").addEventListener("click", fit);
    el("pb-ws-set").addEventListener("click", applyWindowSize);
  }

  //position the window rectangle and playline from a start sample, relative to the current view span
  //during playback the start can exceed the view, the overflow clips it, no re-raster
  function positionOverlay(start){
    const wrap = el("pb-body").querySelector(".strip-wrap");
    if(!wrap) return;
    const leftFrac = (start - viewStart) / viewSpan;
    const winFrac = segment / viewSpan;
    wrap.querySelector(".strip-window").style.left = (leftFrac * 100) + "%";
    wrap.querySelector(".strip-window").style.width = "max(2px, " + (winFrac * 100) + "%)";
    wrap.querySelector(".strip-line").style.left = (leftFrac * 100) + "%";
    el("strip-pos").textContent = "start " + start;
    el("strip-span").textContent = "window " + segment + " \u00b7 view " + viewSpan;
  }

  //drag moves the window start within the view, the overlay tracks live, the seek commits on release
  function attachDrag(wrap){
    let dragging = false, pending = 0;
    const startFromEvent = e => {
      const rect = wrap.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const sample = viewStart + frac * viewSpan;            //map the pixel into the view, not the whole recording
      return Math.max(0, Math.min(nSamples - segment, Math.round(sample)));
    };
    const move = e => { if(!dragging) return; pending = startFromEvent(e); positionOverlay(pending); };
    const up = () => {
      if(!dragging) return;
      dragging = false;
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
      commitSeek(pending);
    };
    wrap.addEventListener("mousedown", e => {
      if(locked) return;
      dragging = true; pending = startFromEvent(e); positionOverlay(pending);
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
  }

  //commit a seek, the returned snapshot re-renders the panels and may recentre the view
  async function commitSeek(start){
    const res = await fetch("/seek", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({start})
    });
    const snap = await res.json();
    if(!snap.error) GUI.apply(snap);
  }

  //=====
  //zoom, centres the new span on the cursor, the server clamps and re-rasters
  //====
  async function zoomTo(span){
    const start = Math.round(lastCursor - span / 2);
    const res = await fetch("/zoom", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({start, span})
    });
    const snap = await res.json();
    if(!snap.error) GUI.apply(snap);
  }
  function zoomIn(){ zoomTo(Math.max(segment, Math.floor(viewSpan / 2))); }
  function zoomOut(){ zoomTo(Math.min(nSamples, viewSpan * 2)); }
  function fit(){ zoomTo(nSamples); }

  //=====
  //window size, text plus unit, validated before commit, the seconds path converts through sfreq in the env
  //====
  async function applyWindowSize(){
    const raw = parseFloat(el("pb-ws").value);
    const unit = el("pb-ws-unit").value;
    const status = el("pb-ws-status");

    if(unit === "timepoints"){
      if(!Number.isInteger(raw) || raw < 1 || raw > nSamples){
        status.textContent = `1 to ${nSamples} samples`; return;
      }
    } else {
      const maxSec = nSamples / sfreq;
      if(!(raw > 0 && raw <= maxSec)){ status.textContent = `0 to ${maxSec.toFixed(1)} s`; return; }
    }
    status.textContent = "";

    const res = await fetch("/set", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({name:"window_size", value:{value: raw, unit}})
    });
    const snap = await res.json();
    if(snap.error){ status.textContent = snap.error; return; }
    GUI.apply(snap);
  }

  //=====
  //playback, the loop advances server-side and moves the playline, both panels receive the frame
  //====
  function reflectButton(){
    const b = el("pb-play");
    if(b){ b.textContent = playing ? "pause" : "play"; b.classList.toggle("on", playing); }
  }

  //freeze the strip drag and the zoom/window controls during playback, the play button stays live to pause
  function lockControls(){
    const wrap = el("pb-body").querySelector(".strip-wrap");
    if(wrap) wrap.classList.toggle("locked", locked);
    ["pb-zoom", "pb-ws-ctrl"].forEach(c => {
      const e = el("pb-body").querySelector("." + c);
      if(e) e.classList.toggle("locked", locked);
    });
  }

  function togglePlay(){ playing ? stopPlay() : startPlay(); }

  async function startPlay(){
    const res = await fetch("/play_start", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
    const snap = await res.json();
    if(snap.error) return;
    playing = true;
    GUI.apply(snap);   //locks controls across all panels
    playLoop();
  }

  //one frame: advance and project server-side, hand the frame to both panels, move the playline
  async function playLoop(){
    if(!playing) return;
    const res = await fetch("/play_step", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        vis_mode: window.FeatureView ? FeatureView.mode() : "image",
        delta_mode: window.DecodeView ? DecodeView.deltaMode() : "field"
      })
    });
    const data = await res.json();
    if(data.error){ stopLocal(); GUI.refresh(); return; }
    if(window.Frame) Frame.applyFrame(data);
    lastCursor = data.cursor;
    positionOverlay(data.cursor);
    if(data.at_end){ stopLocal(); GUI.refresh(); return; }
    playTimer = setTimeout(playLoop, 120);
  }

  async function stopPlay(){
    stopLocal();
    const res = await fetch("/play_stop", {method:"POST"});
    const snap = await res.json();
    if(!snap.error) GUI.apply(snap);
  }

  function stopLocal(){ playing = false; if(playTimer){ clearTimeout(playTimer); playTimer = null; } reflectButton(); }

  //snapshot-driven: refetch the raster on a new strip or view version, position the overlay, reflect lock and button
  async function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("pb-body").innerHTML = `<div class="pb-blank">no recording loaded</div>`;
      lastStripV = -1; lastViewV = -1; stopLocal();
      return;
    }
    nSamples = snap.recording.time_points;
    sfreq = snap.recording.sfreq;
    segment = snap.config.window_size;
    viewStart = snap.view.start;
    viewSpan = snap.view.span;
    lastCursor = snap.cursor;
    locked = !!snap.locked;

    if(snap.strip_version !== lastStripV || snap.view_version !== lastViewV){
        raster = await fetchRaster();
        if(!raster) return;
        lastStripV = snap.strip_version;
        lastViewV = snap.view_version;
        if(!el("pb-play")) buildDom();              //build the controls and strip once
        else el("pb-body").querySelector(".strip-raster").src = raster.raster_url;  //otherwise just swap the image
        }

    if(!locked) playing = false;   //a snapshot that unlocks means playback has ended
    lockControls();
    reflectButton();
    positionOverlay(snap.cursor);
  }

  GUI.register("playback", onState);
})();