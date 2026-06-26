//recording strip with playback: a raster of the whole recording, a draggable playline, and a play loop
//the raster is fetched once per strip_version, playback drives both panels through the frame coordinator
(function(){
  let stripVersion = -1;
  let raster = null;      //{raster_url, n, width}
  let nSamples = 0, segment = 0;
  let locked = false;
  let playing = false, playTimer = null;

  function el(id){ return document.getElementById(id); }

  async function fetchRaster(){ const res = await fetch("/strip"); const d = await res.json(); return d.error ? null : d; }

  //build the strip dom and the play control once, attach the drag handler
  function buildDom(){
    el("pb-body").innerHTML = `
      <div class="pb-controls"><button id="pb-play" class="seg">play</button></div>
      <div class="strip-wrap">
        <img class="strip-raster" src="${raster.raster_url}" alt="recording">
        <div class="strip-window"></div>
        <div class="strip-line"></div>
      </div>
      <div class="strip-meta"><span id="strip-pos"></span><span id="strip-span"></span></div>`;
    attachDrag(el("pb-body").querySelector(".strip-wrap"));
    el("pb-play").addEventListener("click", togglePlay);
  }

  //position the window rectangle and playline from a start sample
  function positionOverlay(start){
    const wrap = el("pb-body").querySelector(".strip-wrap");
    if(!wrap) return;
    const startFrac = start / nSamples, winFrac = segment / nSamples;
    wrap.querySelector(".strip-window").style.left = (startFrac * 100) + "%";
    wrap.querySelector(".strip-window").style.width = "max(2px, " + (winFrac * 100) + "%)";
    wrap.querySelector(".strip-line").style.left = (startFrac * 100) + "%";
    el("strip-pos").textContent = "start " + start;
    el("strip-span").textContent = "window " + segment;
  }

  //drag moves the window start, the overlay tracks live, the seek commits on release, blocked while locked
  function attachDrag(wrap){
    let dragging = false, pending = 0;
    const startFromEvent = e => {
      const rect = wrap.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      return Math.max(0, Math.min(nSamples - segment, Math.round(frac * nSamples)));
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

  //commit a seek, the returned snapshot re-renders both panels through the frame coordinator
  async function commitSeek(start){
    const res = await fetch("/seek", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({start})
    });
    const snap = await res.json();
    if(!snap.error) GUI.apply(snap);
  }

  function reflectButton(){
    const b = el("pb-play");
    if(b){ b.textContent = playing ? "pause" : "play"; b.classList.toggle("on", playing); }
  }

  function lockStrip(){ const wrap = el("pb-body").querySelector(".strip-wrap"); if(wrap) wrap.classList.toggle("locked", locked); }

  function togglePlay(){ playing ? stopPlay() : startPlay(); }

  //start playback, the returned snapshot locks the controls immediately, then the loop runs
  async function startPlay(){
    const res = await fetch("/play_start", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
    const snap = await res.json();
    if(snap.error) return;
    playing = true;
    GUI.apply(snap);   //locks controls and the strip across all panels
    playLoop();
  }

  //one frame: advance and project server-side, apply the whole frame to both panels, move the playline
  async function playLoop(){
    if(!playing) return;
    const res = await fetch("/play_step", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({kind: window.FeatureView ? FeatureView.kind() : "image"})
    });
    const data = await res.json();
    if(data.error){ stopLocal(); GUI.refresh(); return; }
    if(window.Frame) Frame.applyFrame(data);
    positionOverlay(data.cursor);
    if(data.at_end){ stopLocal(); GUI.refresh(); return; }
    playTimer = setTimeout(playLoop, 120);
  }

  //stop playback, the returned snapshot unlocks the controls
  async function stopPlay(){
    stopLocal();
    const res = await fetch("/play_stop", {method:"POST"});
    const snap = await res.json();
    if(!snap.error) GUI.apply(snap);
  }

  function stopLocal(){ playing = false; if(playTimer){ clearTimeout(playTimer); playTimer = null; } reflectButton(); }

  //snapshot-driven: fetch the raster on a new strip_version, position the overlay, reflect lock and button
  async function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("pb-body").innerHTML = `<div class="pb-blank">no recording loaded</div>`;
      stripVersion = -1; stopLocal();
      return;
    }
    nSamples = snap.recording.time_points;
    segment = snap.config.window_size;
    locked = !!snap.locked;

    if(snap.strip_version !== stripVersion){
      raster = await fetchRaster();
      if(!raster) return;
      stripVersion = snap.strip_version;
      buildDom();
    }
    if(!locked) playing = false;  //a snapshot that unlocks means playback has ended
    lockStrip();
    reflectButton();
    positionOverlay(snap.cursor);
  }

  GUI.register("playback", onState);
})();