//spherical harmonics panel: three flat views over the current window's least squares projection
//coefficients heatmap, reconstruction residual, and the Y c = b basis system, all from the shared frame
(function(){
  let view = "coeff";         //coeff | residual | basis
  let data = null;            //last sh frame {coeffs, residual, stack, names}
  let basis = null;           //static {YT, n_modes, n_channels}, refetched on geometry change
  let geomVersion = -1;
  let selFeature = 0;         //which feature column the basis view draws

  function el(id){ return document.getElementById(id); }
  function column(mat, j){ return mat.map(row => row[j]); }

  //size the canvas to the body width and return a ready context
  function ctx(){
    const canvas = el("sh-canvas"), body = el("sh-body");
    const w = Math.max(220, body.clientWidth - 4), h = 320;
    return {g: setupPlot(canvas, w, h), w, h};
  }

  //coefficient block as a modes-by-features heatmap, self-normalised
  function drawCoeffs(g, w, h){
    g.clearRect(0, 0, w, h);
    const coeffs = data.coeffs, names = data.names;
    const modes = coeffs.length, F = names.length;
    if(!modes || !F) return;
    let amax = 1e-9;
    for(const row of coeffs) for(const v of row) amax = Math.max(amax, Math.abs(v));

    const padL = 34, padT = 44, padR = 10, padB = 10;
    const cw = Math.max(4, Math.min(46, (w - padL - padR) / F));
    const ch = Math.max(1, Math.min(16, (h - padT - padB) / modes));
    const x0 = padL, y0 = padT;

    for(let i=0;i<modes;i++) for(let j=0;j<F;j++){
      g.fillStyle = cellColor(coeffs[i][j], amax);
      g.fillRect(x0 + j*cw, y0 + i*ch, Math.max(1, cw-0.5), Math.max(1, ch-0.5));
    }
    g.fillStyle = "#6b6a66"; g.font = "9px sans-serif"; g.textBaseline = "alphabetic";
    for(let j=0;j<F;j++){
      g.save();
      g.translate(x0 + j*cw + cw/2, y0 - 6); g.rotate(-Math.PI/4);
      g.textAlign = "left"; g.fillText(names[j], 0, 0);
      g.restore();
    }
    g.fillStyle = "#9a978f"; g.textAlign = "right"; g.textBaseline = "middle";
    const step = Math.max(1, Math.round(modes/12));
    for(let i=0;i<modes;i+=step) g.fillText(String(i), x0 - 4, y0 + i*ch + ch/2);
    g.fillStyle = "#6b6a66"; g.font = "10px sans-serif"; g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.fillText(`coefficients  (L+1)^2=${modes} x F=${F}`, x0, 16);
  }

  //per-feature reconstruction residual as labelled bars in [0, 1]
  function drawResidual(g, w, h){
    g.clearRect(0, 0, w, h);
    const residual = data.residual, names = data.names, F = names.length;
    if(!F) return;
    const padL = 96, padT = 26, padR = 44, padB = 10;
    const bw = w - padL - padR;
    const rh = Math.min(34, (h - padT - padB) / F);
    const y0 = padT;
    g.font = "11px sans-serif"; g.textBaseline = "middle";
    for(let i=0;i<F;i++){
      const v = residual[i], frac = Math.max(0, Math.min(1, v)), y = y0 + i*rh + rh/2;
      g.fillStyle = "#4a4843"; g.textAlign = "right"; g.fillText(names[i], padL - 8, y);
      g.fillStyle = "#eceae5"; g.fillRect(padL, y - 6, bw, 12);
      g.fillStyle = cellColor(frac, 1); g.fillRect(padL, y - 6, bw*frac, 12);
      g.fillStyle = "#6b6a66"; g.textAlign = "left"; g.fillText(v.toFixed(2), padL + bw + 6, y);
    }
    g.fillStyle = "#6b6a66"; g.font = "10px sans-serif"; g.textAlign = "left";
    g.fillText("reconstruction residual = |b - Y^T c| / |b|", padL, 14);
  }

  //the Y c = b system for one feature, drawn with the sandbox helper
  function drawBasis(g, w, h){
    if(!basis || !data){ g.clearRect(0, 0, w, h); return; }
    const f = Math.min(selFeature, data.names.length - 1);
    drawLinearSystem(g, w, h, {
      M: basis.n_modes, n: basis.n_channels, Y: basis.YT,
      b: column(data.stack, f), c: column(data.coeffs, f),
    });
  }

  function draw(){
    if(!el("sh-canvas")) return;
    const {g, w, h} = ctx();
    if(view === "coeff" && data) drawCoeffs(g, w, h);
    else if(view === "residual" && data) drawResidual(g, w, h);
    else if(view === "basis") drawBasis(g, w, h);
    else g.clearRect(0, 0, w, h);
  }

  //feature buttons for the basis view, choosing which column the system draws
  function buildFeatSel(){
    if(!data) return;
    el("sh-featsel").innerHTML = data.names.map((n, i) =>
      `<button class="seg sh-feat ${i === selFeature ? "on" : ""}" data-feat="${i}">${n}</button>`).join("");
    el("sh-featsel").querySelectorAll("[data-feat]").forEach(b =>
      b.addEventListener("click", () => { selFeature = +b.dataset.feat; buildFeatSel(); draw(); }));
  }

  function setView(v){
    view = v;
    el("sh-body").querySelectorAll("[data-view]").forEach(b => b.classList.toggle("on", b.dataset.view === v));
    el("sh-featsel").style.display = (v === "basis") ? "flex" : "none";
    if(v === "basis") buildFeatSel();
    draw();
  }

  function buildDom(){
    el("sh-body").innerHTML = `
      <div class="sh-toolbar">
        <button class="seg on" data-view="coeff">coefficients</button>
        <button class="seg" data-view="residual">residual</button>
        <button class="seg" data-view="basis">basis</button>
      </div>
      <div id="sh-featsel" class="sh-featsel" style="display:none"></div>
      <canvas id="sh-canvas"></canvas>`;
    el("sh-body").querySelectorAll("[data-view]").forEach(b =>
      b.addEventListener("click", () => setView(b.dataset.view)));
  }

  async function fetchBasis(){
    const res = await fetch("/sh_basis");
    const d = await res.json();
    if(!d.error) basis = d;
  }

  //exposed for the frame coordinator and the playback loop to push the projection data
  window.SHView = {
    setData(sh){
      data = sh;
      if(!data.names || selFeature >= data.names.length) selFeature = 0;
      if(view === "basis") buildFeatSel();
      draw();
    }
  };

  //snapshot-driven: build the panel on load, refetch the basis when the geometry changes
  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("sh-body").innerHTML = `<div class="sh-blank">no recording loaded</div>`;
      geomVersion = -1; data = null; basis = null;
      return;
    }
    if(!el("sh-canvas")) buildDom();
    if(snap.geometry_version !== geomVersion){
      geomVersion = snap.geometry_version;
      fetchBasis().then(() => { if(view === "basis") draw(); });
    }
  }

  window.addEventListener("resize", draw);
  GUI.register("sh", onState);
})();