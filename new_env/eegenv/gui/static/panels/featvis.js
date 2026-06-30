//feature visualisation panel: five mutually exclusive modes, each pulls only its own computation path
//image, stack, raw, operator render server-side to a png, harmonic draws the Y^T c = b system on the canvas
//it never fetches its own window, the frame coordinator pushes vis payloads through setVis
(function(){
  let mode = "image";
  let locked = false;
  let basis = null;          //{YT, n_modes, n_channels}, refetched on a geometry change for the harmonic system
  let geomVersion = -1;
  let lastVis = null;        //the last vis payload, redrawn on resize and on a late basis arrival
  const PNG_MODES = new Set(["image", "stack", "raw", "operator"]);

  function el(id){ return document.getElementById(id); }

  function setActive(){
    document.querySelectorAll("#panel-featviz [data-mode]").forEach(b =>
      b.classList.toggle("on", b.dataset.mode === mode));
  }

  //build the img and canvas holders once, both hidden, the active mode reveals one
  function ensureContent(){
    if(el("fv-img") || el("fv-canvas")) return;
    el("fv-body").innerHTML =
      `<img id="fv-img" style="display:none" alt="feature visualisation">` +
      `<canvas id="fv-canvas" style="display:none"></canvas>`;
  }

  //draw the harmonic system, needs both the per-window payload and the static basis
  function drawHarmonic(vis){
    const cv = el("fv-canvas");
    if(!cv || !basis) return;
    const w = Math.max(560, el("fv-body").clientWidth - 4), h = 480;
    const g = setupPlot(cv, w, h);
    drawHarmonicSystem(g, w, h, {YT: basis.YT, c: vis.coeffs, b: vis.stack, residual: vis.residual, names: vis.names});
  }

  //exposed for the frame coordinator and the playback loop to push frames and read the active mode
  window.FeatureView = {
    mode(){ return mode; },
    setVis(vis){
      lastVis = vis;
      ensureContent();
      const img = el("fv-img"), cv = el("fv-canvas");
      if(PNG_MODES.has(vis.mode)){
        cv.style.display = "none"; img.style.display = "";
        img.src = vis.render_url;
      } else if(vis.mode === "harmonic"){
        img.style.display = "none"; cv.style.display = "";
        drawHarmonic(vis);
      }
    }
  };

  //change mode and ask the coordinator for a fresh frame, unless playback is driving frames
  function setMode(m){
    mode = m; setActive();
    if(!locked && window.Frame) Frame.refresh();
  }

  async function fetchBasis(){
    const res = await fetch("/sh_basis");
    const d = await res.json();
    if(!d.error) basis = d;
  }

  //snapshot-driven: keep content mounted while loaded, refetch the basis when the geometry changes
  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("fv-body").innerHTML = `<div class="fv-blank">no recording loaded</div>`;
      basis = null; geomVersion = -1; lastVis = null;
      return;
    }
    locked = !!snap.locked;
    ensureContent();
    if(snap.geometry_version !== geomVersion){
      geomVersion = snap.geometry_version;
      fetchBasis().then(() => { if(mode === "harmonic" && lastVis) drawHarmonic(lastVis); });
    }
  }

  document.addEventListener("DOMContentLoaded", ()=>{
    document.querySelectorAll("#panel-featviz [data-mode]").forEach(b =>
      b.addEventListener("click", ()=> setMode(b.dataset.mode)));
  });

  window.addEventListener("resize", ()=>{ if(mode === "harmonic" && lastVis) drawHarmonic(lastVis); });

  GUI.register("featvis", onState);
})();