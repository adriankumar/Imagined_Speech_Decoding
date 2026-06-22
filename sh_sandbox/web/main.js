//sandbox state and control wiring

const MAX_L = 4;
const BASIS_NTHETA = 96, BASIS_NPHI = 192;
const SYNTH_NTHETA = 64, SYNTH_NPHI = 128;
const PLOT_W = 220, PLOT_H = 84;
const CROSS_W = 220, CROSS_H = 150;
const SYS_W = 372, SYS_H = 300;

//single source of truth for what the view shows
const state = {
  view: "basis",
  l: 2, m: 0,
  mode: "blob",
  nodes: false,
  maxL: 3,
  coeffs: new Map(),
  n: 32,
  slice: true,
  bedit: false,        //round-trip: edit b and solve c, rather than set c
  bvec: [],            //the editable measurement vector
  animate: false,
  animSpeed: 1.0
};

let ctx = null;
let drawn = [];
let el = {};
let synthFields = null, synthFieldsL = -1;
let lastRtError = 0;
let animDirs = null;
let animMesh = null, animMeshRecon = null, animCoeffMap = null;
let animAbsmax = 1, animAbsmaxRecon = 1, animTime = 0;
let animNodeObjs = [];
let resCache = {key:null, R:null, modes:null};
let slicePhi = 0, sliceTheta = Math.PI/2;
let inspectMesh = null;
let marker = null;
let raycaster = null;

//dispose a mesh/group and its descendants, including sprite/texture maps
function disposeObject(o){
  if(o.geometry) o.geometry.dispose();
  if(o.material){ if(o.material.map) o.material.map.dispose(); o.material.dispose(); }
  if(o.children) for(const c of o.children) disposeObject(c);
}

//remove the per-frame animated node objects
function clearAnimNodes(){
  for(const o of animNodeObjs){ ctx.scene.remove(o); disposeObject(o); }
  animNodeObjs = [];
}

//remove whatever we drew last
function clearDrawn(){
  clearAnimNodes();
  for(const o of drawn){ ctx.scene.remove(o); disposeObject(o); }
  drawn = [];
}

//precompute the synthesis basis grids when the level changes
function ensureSynthFields(){
  if(synthFieldsL !== state.maxL){
    synthFields = precomputeBasis(state.maxL, SYNTH_NTHETA, SYNTH_NPHI);
    synthFieldsL = state.maxL;
  }
}

//precompute the resolution operator when n or the level changes
function ensureResolution(){
  const key = `${state.n}_${state.maxL}`;
  if(resCache.key !== key){
    const res = resolutionMatrix(state.n, state.maxL);
    resCache = {key, R:res.R, modes:res.modes};
  }
}

//coeff map of the field currently shown (single mode in basis, the sum otherwise)
function currentFieldMap(){
  if(state.view === "basis") return new Map([[`${state.l},${state.m}`, 1]]);
  return state.coeffs;
}

//orient the slice through the field's strongest direction (never a node)
function setSliceDir(){
  const d = dominantDirection(currentFieldMap(), state.maxL);
  slicePhi = d.phi; sliceTheta = d.theta;
}

//add the translucent slice plane at the current slice azimuth, if enabled
function addSlicePlane(){
  if(!state.slice) return;
  const p = makeSlicePlane(slicePhi);
  if(state.view === "roundtrip" && !state.bedit){ p.scale.setScalar(0.8); p.position.x = -0.95; }
  ctx.scene.add(p); drawn.push(p);
}

//show the field's peak magnitude on the colour bar
function updateColourBar(grid){
  const absmax = Math.max(Math.abs(grid.min), Math.abs(grid.max)) || 0;
  el.cbarMag.textContent = `±${absmax.toFixed(2)}`;
}

//rebuild the per-frame moving nodal contours from the current grids
function updateAnimNodes(specs){
  clearAnimNodes();
  for(const [grid, xoff] of specs){
    const nl = buildZeroContour(grid, 1.004);
    if(xoff !== 0){ nl.scale.setScalar(0.8); nl.position.x = xoff; }
    ctx.scene.add(nl); animNodeObjs.push(nl);
  }
}

//build the animatable mesh(es): one field for basis/synthesis, a pair for round-trip
function buildAnimated(){
  ensureSynthFields();
  animMeshRecon = null;

  if(state.view === "roundtrip"){
    ensureResolution();
    animCoeffMap = new Map(state.coeffs);
    const origBase = combineFields(synthFields, animCoeffMap, SYNTH_NTHETA, SYNTH_NPHI);
    animAbsmax = Math.max(Math.abs(origBase.min), Math.abs(origBase.max)) || 1;
    const recBase = combineFields(synthFields, applyResolution(resCache.R, resCache.modes, animCoeffMap), SYNTH_NTHETA, SYNTH_NPHI);
    animAbsmaxRecon = Math.max(Math.abs(recBase.min), Math.abs(recBase.max)) || 1;

    const om = state.mode==="blob" ? buildBlobMesh(origBase) : buildSphereMesh(origBase);
    om.scale.setScalar(0.8); om.position.x = -0.95;
    const rm = state.mode==="blob" ? buildBlobMesh(recBase) : buildSphereMesh(recBase);
    rm.scale.setScalar(0.8); rm.position.x = 0.95;
    animMesh = om; animMeshRecon = rm;
    inspectMesh = om;
    ctx.scene.add(om, rm); drawn.push(om, rm);

    const dots = buildSamplePoints(state.n, animCoeffMap, state.maxL, animAbsmax);
    dots.scale.setScalar(0.8); dots.position.x = -0.95;
    ctx.scene.add(dots); drawn.push(dots);

    lastRtError = fieldRmsError(origBase, recBase, animAbsmax);
    setSliceDir(); addSlicePlane();
    updateColourBar(origBase);
  } else {
    animCoeffMap = state.view === "basis"
      ? new Map([[`${state.l},${state.m}`, 1]])
      : new Map(state.coeffs);
    const base = combineFields(synthFields, animCoeffMap, SYNTH_NTHETA, SYNTH_NPHI);
    animAbsmax = Math.max(Math.abs(base.min), Math.abs(base.max)) || 1;
    const mesh = state.mode==="blob" ? buildBlobMesh(base) : buildSphereMesh(base);
    animMesh = mesh; ctx.scene.add(mesh); drawn.push(mesh);
    inspectMesh = mesh;
    if(state.view === "basis" && state.nodes){
      const lines = buildNodalLines(state.l, state.m);
      ctx.scene.add(lines); drawn.push(lines);
    }
    setSliceDir(); addSlicePlane();
    updateColourBar(base);
  }
  animTime = 0;
}

//per-frame update: each degree l oscillates at sqrt(l(l+1)); superposition nodes move
function animTick(dt){
  if(!state.animate || !animMesh) return;
  animTime += dt * state.animSpeed;

  const modMap = new Map();
  for(const [key, c] of animCoeffMap){
    if(c === 0) continue;
    const l = parseInt(key.split(",")[0]);
    modMap.set(key, c * Math.cos(Math.sqrt(l*(l+1)) * animTime));
  }

  if(state.view === "roundtrip"){
    const og = combineFields(synthFields, modMap, SYNTH_NTHETA, SYNTH_NPHI);
    updateMeshFromGrid(animMesh, og, state.mode, animAbsmax, animDirs);
    const rg = combineFields(synthFields, applyResolution(resCache.R, resCache.modes, modMap), SYNTH_NTHETA, SYNTH_NPHI);
    updateMeshFromGrid(animMeshRecon, rg, state.mode, animAbsmaxRecon, animDirs);
    if(state.nodes) updateAnimNodes([[og, -0.95], [rg, 0.95]]);
  } else {
    const grid = combineFields(synthFields, modMap, SYNTH_NTHETA, SYNTH_NPHI);
    updateMeshFromGrid(animMesh, grid, state.mode, animAbsmax, animDirs);
    if(state.nodes && state.view === "synthesis") updateAnimNodes([[grid, 0]]);
  }

  renderPanels(animTime);
}

//rebuild the 3d view from the current state
function rebuild(){
  clearDrawn();
  animMesh = null;

  if(state.animate && !state.bedit){
    buildAnimated();
    return;
  }

  if(state.view === "synthesis"){
    ensureSynthFields();
    const grid = combineFields(synthFields, state.coeffs, SYNTH_NTHETA, SYNTH_NPHI);
    const mesh = state.mode==="blob" ? buildBlobMesh(grid) : buildSphereMesh(grid);
    ctx.scene.add(mesh); drawn.push(mesh);
    inspectMesh = mesh;
    if(state.nodes){ const nl = buildZeroContour(grid, 1.004); ctx.scene.add(nl); drawn.push(nl); }
    setSliceDir(); addSlicePlane();
    updateColourBar(grid);
  } else if(state.view === "roundtrip" && state.bedit){
    ensureSynthFields();
    const grid = combineFields(synthFields, state.coeffs, SYNTH_NTHETA, SYNTH_NPHI);
    const absmax = Math.max(Math.abs(grid.min), Math.abs(grid.max)) || 1;
    const mesh = state.mode==="blob" ? buildBlobMesh(grid) : buildSphereMesh(grid);
    ctx.scene.add(mesh); drawn.push(mesh);
    inspectMesh = mesh;
    if(state.nodes){ const nl = buildZeroContour(grid, 1.004); ctx.scene.add(nl); drawn.push(nl); }
    const dots = buildSamplePointsFromB(state.n, state.bvec, absmax);
    ctx.scene.add(dots); drawn.push(dots);
    setSliceDir(); addSlicePlane();
    updateColourBar(grid);
  } else if(state.view === "roundtrip"){
    ensureSynthFields();
    ensureResolution();
    const origGrid = combineFields(synthFields, state.coeffs, SYNTH_NTHETA, SYNTH_NPHI);
    const absmax = Math.max(Math.abs(origGrid.min), Math.abs(origGrid.max)) || 1;
    const recMap = applyResolution(resCache.R, resCache.modes, state.coeffs);
    const reconGrid = combineFields(synthFields, recMap, SYNTH_NTHETA, SYNTH_NPHI);
    lastRtError = fieldRmsError(origGrid, reconGrid, absmax);

    const origMesh = state.mode==="blob" ? buildBlobMesh(origGrid) : buildSphereMesh(origGrid);
    origMesh.scale.setScalar(0.8); origMesh.position.x = -0.95;
    const reconMesh = state.mode==="blob" ? buildBlobMesh(reconGrid) : buildSphereMesh(reconGrid);
    reconMesh.scale.setScalar(0.8); reconMesh.position.x = 0.95;
    ctx.scene.add(origMesh, reconMesh); drawn.push(origMesh, reconMesh);
    inspectMesh = origMesh;

    if(state.nodes){
      const nlo = buildZeroContour(origGrid, 1.004); nlo.scale.setScalar(0.8); nlo.position.x = -0.95;
      const nlr = buildZeroContour(reconGrid, 1.004); nlr.scale.setScalar(0.8); nlr.position.x = 0.95;
      ctx.scene.add(nlo, nlr); drawn.push(nlo, nlr);
    }
    const dots = buildSamplePoints(state.n, state.coeffs, state.maxL, absmax);
    dots.scale.setScalar(0.8); dots.position.x = -0.95;
    ctx.scene.add(dots); drawn.push(dots);
    setSliceDir(); addSlicePlane();
    updateColourBar(origGrid);
  } else {
    const grid = evalGrid(state.l, state.m, BASIS_NTHETA, BASIS_NPHI);
    const mesh = state.mode==="blob" ? buildBlobMesh(grid) : buildSphereMesh(grid);
    ctx.scene.add(mesh); drawn.push(mesh);
    inspectMesh = mesh;
    if(state.nodes){ const lines = buildNodalLines(state.l, state.m); ctx.scene.add(lines); drawn.push(lines); }
    setSliceDir(); addSlicePlane();
    updateColourBar(grid);
  }
}

//jump to a specific mode (grid buttons)
function setMode(l, m){
  state.l = l; state.m = m;
  rebuild(); refreshControls();
}

//populate the L dropdown 0..MAX_L
function initLSelect(){
  el.selL.innerHTML = "";
  for(let v=0;v<=MAX_L;v++){
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v;
    el.selL.appendChild(opt);
  }
}

//build the pyramid of (l, m) buttons, one row per level up to maxL
function buildGrid(){
  el.navGrid.innerHTML = "";
  for(let l=0;l<=state.maxL;l++){
    const row = document.createElement("div");
    row.className = "grid-row";
    for(let m=-l;m<=l;m++){
      const b = document.createElement("button");
      b.className = "cell";
      b.textContent = `${l},${m}`;
      b.onclick = ()=>setMode(l, m);
      if(l===state.l && m===state.m) b.classList.add("on");
      row.appendChild(b);
    }
    el.navGrid.appendChild(row);
  }
}

//build one slider per (l, m) up to maxL, reading persistent coeffs
function buildSliders(){
  el.sidebar.innerHTML = "";

  const head = document.createElement("div");
  head.className = "side-head";
  const title = document.createElement("span");
  title.textContent = "coefficients";
  const reset = document.createElement("button");
  reset.className = "seg"; reset.textContent = "reset";
  reset.onclick = resetCoeffs;
  head.appendChild(title); head.appendChild(reset);
  el.sidebar.appendChild(head);

  for(let l=0;l<=state.maxL;l++){
    for(let m=-l;m<=l;m++){
      const key = `${l},${m}`;
      const c0 = state.coeffs.get(key) || 0;

      const row = document.createElement("div");
      row.className = "slider-row";

      const lab = document.createElement("span");
      lab.className = "slider-label";
      lab.textContent = key;

      const input = document.createElement("input");
      input.type = "range";
      input.min = "-1"; input.max = "1"; input.step = "0.05";
      input.value = String(c0);

      const val = document.createElement("span");
      val.className = "slider-val";
      val.textContent = c0.toFixed(2);

      input.oninput = ()=>{
        const c = parseFloat(input.value);
        state.coeffs.set(key, c);
        val.textContent = c.toFixed(2);
        rebuild();
        refreshDynamicReadout();
      };

      row.appendChild(lab); row.appendChild(input); row.appendChild(val);
      el.sidebar.appendChild(row);
    }
  }
}

//build one slider per sample point for editing b, plus the read-only recovered c
function buildBSliders(){
  el.sidebar.innerHTML = "";

  const head = document.createElement("div");
  head.className = "side-head";
  const title = document.createElement("span");
  title.textContent = "measurements (b)";
  const clear = document.createElement("button");
  clear.className = "seg"; clear.textContent = "clear";
  clear.onclick = clearB;
  head.appendChild(title); head.appendChild(clear);
  el.sidebar.appendChild(head);

  for(let i=0;i<state.n;i++){
    const v0 = state.bvec[i] || 0;

    const row = document.createElement("div");
    row.className = "slider-row";

    const lab = document.createElement("span");
    lab.className = "slider-label";
    lab.textContent = `b${i}`;

    const input = document.createElement("input");
    input.type = "range";
    input.min = "-2.5"; input.max = "2.5"; input.step = "0.05";
    input.value = String(v0);

    const val = document.createElement("span");
    val.className = "slider-val";
    val.textContent = v0.toFixed(2);

    input.oninput = ()=>{
      const v = parseFloat(input.value);
      state.bvec[i] = v;
      val.textContent = v.toFixed(2);
      solveBToCoeffs();
      rebuild();
      updateCReadout();
      refreshDynamicReadout();
    };

    row.appendChild(lab); row.appendChild(input); row.appendChild(val);
    el.sidebar.appendChild(row);
  }

  const chead = document.createElement("div");
  chead.className = "side-head c-head";
  const ctitle = document.createElement("span");
  ctitle.textContent = "recovered c";
  chead.appendChild(ctitle);
  el.sidebar.appendChild(chead);

  const cbox = document.createElement("div");
  cbox.id = "c-readout";
  cbox.className = "c-readout";
  el.sidebar.appendChild(cbox);
  updateCReadout();
}

//refresh the read-only recovered c display without rebuilding the b sliders
function updateCReadout(){
  const box = document.getElementById("c-readout");
  if(!box) return;
  let html = "";
  for(let l=0;l<=state.maxL;l++) for(let m=-l;m<=l;m++){
    const c = state.coeffs.get(`${l},${m}`) || 0;
    const strong = Math.abs(c) >= 0.2 ? " creadout-strong" : "";
    html += `<div class="creadout-row${strong}"><span>${l},${m}</span><span>${c.toFixed(3)}</span></div>`;
  }
  box.innerHTML = html;
}
//solve the coefficients that best fit the current b vector
function solveBToCoeffs(){
  state.coeffs = solveCoeffsFromB(state.n, state.bvec, state.maxL);
}

//zero every measurement and resolve
function clearB(){
  state.bvec = new Array(state.n).fill(0);
  solveBToCoeffs();
  buildBSliders();
  rebuild();
  refreshDynamicReadout();
}

//enter b-edit: seed b from the current field's samples, keep the field as is until edited
function enterBEdit(){
  state.bedit = true;
  state.animate = false;
  state.bvec = sampleBVector(state.n, state.coeffs, state.maxL);
  rebuild(); refreshControls();
}

//leave b-edit, keeping the recovered coefficients
function exitBEdit(){
  state.bedit = false;
  rebuild(); refreshControls();
}

//zero every coefficient and redraw
function resetCoeffs(){
  state.coeffs.clear();
  buildSliders();
  rebuild();
  refreshDynamicReadout();
}

//time-modulated coeffs for animating the panels
function modulatedCoeffs(phase){
  const map = new Map();
  for(const [key, c] of state.coeffs){
    if(!c) continue;
    const l = parseInt(key.split(",")[0]);
    map.set(key, c * Math.cos(Math.sqrt(l*(l+1)) * phase));
  }
  return map;
}

//draw all three 2d panels; phase=null is the static full-amplitude view
function renderPanels(phase){
  if(state.view === "basis"){
    const tFac = phase===null ? 1 : Math.cos(Math.sqrt(state.l*(state.l+1)) * phase);
    const ls = legendreSamples(state.l, state.m, 140);
    const refLeg = Math.max(...ls.map(s=>Math.abs(s.v)), 1e-9);
    const lz = sampleZeros(ls);
    const lsS = ls.map(s=>({t:s.t, v:s.v*tFac}));
    drawCrossSection(el.gCross, CROSS_W, CROSS_H, lsS, lsS, refLeg, lz);
    drawCurve(el.gLeg, PLOT_W, PLOT_H, lsS, lz, refLeg);
    el.capLeg.textContent = `${lz.length} latitude ring${lz.length===1?"":"s"}`;

    const as = azimuthSamples(state.m, 140);
    const refAz = Math.max(...as.map(s=>Math.abs(s.v)), 1e-9);
    const az = sampleZeros(as);
    const asS = as.map(s=>({t:s.t, v:s.v*tFac}));
    drawCurve(el.gPhi, PLOT_W, PLOT_H, asS, az, refAz);
    const mer = Math.abs(state.m);
    el.capPhi.textContent = `${mer} meridian${mer===1?"":"s"}`;
  } else {
    const coeffs = phase===null ? state.coeffs : modulatedCoeffs(phase);
    const refMerS = meridianSamples(state.coeffs, state.maxL, slicePhi, 140);
    const refMer = Math.max(...refMerS.map(s=>Math.abs(s.v)), 1e-9);
    const refParS = parallelSamples(state.coeffs, state.maxL, sliceTheta, 140);
    const refPar = Math.max(...refParS.map(s=>Math.abs(s.v)), 1e-9);

    const merR = meridianSamples(coeffs, state.maxL, slicePhi, 140);
    const merL = meridianSamples(coeffs, state.maxL, slicePhi + Math.PI, 140);
    const mz = sampleZeros(merR);
    drawCrossSection(el.gCross, CROSS_W, CROSS_H, merR, merL, refMer, mz);
    drawCurve(el.gLeg, PLOT_W, PLOT_H, merR, mz, refMer);
    el.capLeg.textContent = `meridian slice · φ = ${(slicePhi*180/Math.PI).toFixed(0)}°`;

    const par = parallelSamples(coeffs, state.maxL, sliceTheta, 140);
    const pz = sampleZeros(par);
    drawCurve(el.gPhi, PLOT_W, PLOT_H, par, pz, refPar);
    el.capPhi.textContent = `parallel slice · θ = ${(sliceTheta*180/Math.PI).toFixed(0)}°`;
  }
}

//flag plus reconstruction error for the round-trip view
function roundtripReadout(){
  const needed = (state.maxL+1)*(state.maxL+1);
  const ok = state.n >= needed;
  const pct = lastRtError*100;
  el.readout.innerHTML =
    `round-trip   ·   n = ${state.n} vs (L+1)² = ${needed}   ·   ` +
    `<span class="flag ${ok?'ok':'bad'}">${ok ? 'resolved' : 'under-resolved'}</span>` +
    `   ·   reconstruction error ${pct.toFixed(pct<10?2:1)}%`;
}

//regime flag plus fit residual for the b-edit view
function bEditReadout(){
  const M = (state.maxL+1)*(state.maxL+1);
  const pct = bResidual(state.n, state.bvec, state.coeffs, state.maxL)*100;
  let regime, cls, note;
  if(state.n < M){ regime = "underdetermined"; cls = "warn"; note = "any b fits exactly — c is the flattest solution"; }
  else if(state.n === M){ regime = "exactly determined"; cls = "ok"; note = "a unique c fits b"; }
  else { regime = "overdetermined"; cls = (pct < 1) ? "ok" : "bad"; note = (pct < 1) ? "b lies within the basis" : "b exceeds what M modes can build"; }
  el.readout.innerHTML =
    `edit b   ·   n = ${state.n} vs M = ${M}   ·   ` +
    `<span class="flag ${cls}">${regime}</span>   ·   ${note}   ·   fit residual ${pct.toFixed(pct<10?2:1)}%`;
}

//draw the round-trip linear system panel (using the edited b when in b-edit)
function renderSystem(){
  const data = systemData(state.n, state.coeffs, state.maxL, state.bedit ? state.bvec : null);
  drawLinearSystem(el.gSystem, SYS_W, SYS_H, data);
}

//update the live readout (and system panel) without rebuilding the sliders
function refreshDynamicReadout(){
  if(state.view === "roundtrip"){
    if(state.bedit) bEditReadout(); else roundtripReadout();
    renderSystem();
  }
}

//term-by-term Y breakdown for a single basis harmonic at (theta, phi)
function inspectorBasisHTML(l, m, theta, phi){
  const am = Math.abs(m);
  const N = shNorm(l, m);
  const P = assocLegendre(l, am, Math.cos(theta));
  const Y = ylm(l, m, theta, phi);
  let azim, azLabel;
  if(m > 0){ azim = Math.cos(am*phi); azLabel = `cos(${am}φ)`; }
  else if(m < 0){ azim = Math.sin(am*phi); azLabel = `sin(${am}φ)`; }
  else { azim = 1; azLabel = "1"; }
  let html = `<div class="insp-coord">θ = ${(theta*180/Math.PI).toFixed(1)}°   φ = ${(phi*180/Math.PI).toFixed(1)}°</div>`;
  html += `<table class="insp-tab">`;
  html += `<tr><td>N<sub>${l},${m}</sub></td><td>${N.toFixed(4)}</td></tr>`;
  html += `<tr><td>P<sub>${l}</sub><sup>${am}</sup>(cos θ)</td><td>${P.toFixed(4)}</td></tr>`;
  if(m !== 0) html += `<tr><td>√2</td><td>${Math.SQRT2.toFixed(4)}</td></tr>`;
  html += `<tr><td>${azLabel}</td><td>${azim.toFixed(4)}</td></tr>`;
  html += `<tr class="insp-tot"><td>Y<sub>${l}</sub><sup>${m}</sup></td><td>${Y.toFixed(4)}</td></tr>`;
  html += `</table>`;
  html += `<div class="insp-prod">${m!==0?"√2 · ":""}N · P · ${azLabel} = Y</div>`;
  return html;
}

//per-mode contribution table for a superposition at (theta, phi)
function inspectorSumHTML(theta, phi){
  let html = `<div class="insp-coord">θ = ${(theta*180/Math.PI).toFixed(1)}°   φ = ${(phi*180/Math.PI).toFixed(1)}°</div>`;
  html += `<table class="insp-tab"><tr><th>mode</th><th>c</th><th>Y</th><th>c·Y</th></tr>`;
  let total = 0, rows = "";
  for(let l=0;l<=state.maxL;l++) for(let m=-l;m<=l;m++){
    const c = state.coeffs.get(`${l},${m}`) || 0;
    if(Math.abs(c) < 1e-9) continue;
    const Y = ylm(l, m, theta, phi);
    const cy = c*Y; total += cy;
    rows += `<tr><td>${l},${m}</td><td>${c.toFixed(2)}</td><td>${Y.toFixed(3)}</td><td>${cy.toFixed(3)}</td></tr>`;
  }
  html += rows || `<tr><td colspan="4">no active modes</td></tr>`;
  html += `<tr class="insp-tot"><td colspan="3">field = Σ c·Y</td><td>${total.toFixed(3)}</td></tr></table>`;
  return html;
}

//fill the inspector panel for the hovered direction
function showInspector(theta, phi){
  el.inspector.innerHTML = state.view === "basis"
    ? inspectorBasisHTML(state.l, state.m, theta, phi)
    : inspectorSumHTML(theta, phi);
  el.inspector.style.display = "block";
}

//raycast the hovered point to (theta, phi); always live, including during animation
function onInspectMove(ev){
  if(!inspectMesh || !marker) return;
  const rect = ctx.renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((ev.clientX - rect.left)/rect.width)*2 - 1,
    -((ev.clientY - rect.top)/rect.height)*2 + 1);
  raycaster.setFromCamera(ndc, ctx.camera);
  const hits = raycaster.intersectObject(inspectMesh, false);
  if(hits.length === 0){ marker.visible = false; el.inspector.style.display = "none"; return; }
  marker.position.copy(hits[0].point);
  marker.visible = true;
  const lp = inspectMesh.worldToLocal(hits[0].point.clone());
  if(lp.lengthSq() < 1e-10){ marker.visible = false; el.inspector.style.display = "none"; return; }
  lp.normalize();
  const theta = Math.acos(Math.max(-1, Math.min(1, lp.y)));
  const phi = Math.atan2(lp.z, lp.x);
  showInspector(theta, phi);
}

//hide the inspector when the cursor leaves the canvas
function onInspectLeave(){
  if(marker) marker.visible = false;
  el.inspector.style.display = "none";
}

//redraw highlights, readout, and the active panels
function refreshControls(){
  const isBasis = state.view === "basis";
  const isSynth = state.view === "synthesis";
  const isRoundtrip = state.view === "roundtrip";

  if(marker) marker.visible = false;
  el.inspector.style.display = "none";

  el.btnBasis.classList.toggle("on", isBasis);
  el.btnSynthesis.classList.toggle("on", isSynth);
  el.btnRoundtrip.classList.toggle("on", isRoundtrip);
  el.btnBlob.classList.toggle("on", state.mode==="blob");
  el.btnSphere.classList.toggle("on", state.mode==="sphere");
  el.btnNodes.classList.toggle("on", state.nodes);
  el.btnSlice.classList.toggle("on", state.slice);
  el.btnBedit.classList.toggle("on", state.bedit);
  el.btnAnim.classList.toggle("on", state.animate);
  el.btnAnim.textContent = state.animate ? "pause" : "play";
  el.animSpeed.value = state.animSpeed;
  el.selL.value = state.maxL;

  el.groupAnim.style.display = state.bedit ? "none" : "flex";
  el.btnBedit.style.display = isRoundtrip ? "" : "none";
  el.navGrid.style.display = isBasis ? "flex" : "none";
  el.groupSamples.style.display = isRoundtrip ? "flex" : "none";
  el.sidebar.style.display = (isSynth || isRoundtrip) ? "block" : "none";
  el.plots.style.display = isRoundtrip ? "none" : "flex";
  el.system.style.display = isRoundtrip ? "block" : "none";
  el.rtcaption.style.display = isRoundtrip ? "block" : "none";
  ctx.controls.autoRotate = !isRoundtrip && !state.animate;

  if(isBasis){
    buildGrid();
    const rings = state.l - Math.abs(state.m);
    const merid = Math.abs(state.m);
    el.readout.textContent =
      `l = ${state.l}   m = ${state.m}   ·   ${state.l} nodal lines (${rings} latitude rings, ${merid} longitude meridians)   ·   ${2*state.l+1} modes at this level`;
  } else if(isSynth){
    buildSliders();
    let active = 0;
    for(const v of state.coeffs.values()) if(Math.abs(v) > 1e-9) active++;
    const total = (state.maxL+1)*(state.maxL+1);
    el.readout.textContent = `synthesis   ·   field = Σ c · Y   ·   ${active} of ${total} modes active (L = ${state.maxL})`;
  } else if(state.bedit){
    buildBSliders();
    el.nRange.value = state.n;
    el.nVal.textContent = state.n;
    bEditReadout();
    renderSystem();
    el.rtcaption.textContent = "edit the measurements b — coefficients c are solved to fit them";
  } else {
    buildSliders();
    el.nRange.value = state.n;
    el.nVal.textContent = state.n;
    roundtripReadout();
    renderSystem();
    el.rtcaption.textContent = "left: original field   ·   right: reconstructed from the samples";
  }

  renderPanels(null);
  if(ctx) ctx.resize();
}

//boot
window.addEventListener("DOMContentLoaded", ()=>{
  const stage = document.getElementById("stage");
  if(typeof THREE === "undefined"){
    stage.innerHTML = "<p style='padding:24px;color:#b4332a'>three.js did not load — check web/vendor/three.min.js</p>";
    return;
  }

  ctx = createScene(stage);
  ctx.onFrame = animTick;
  animDirs = computeDirs(SYNTH_NTHETA, SYNTH_NPHI);
  raycaster = new THREE.Raycaster();
  marker = makeInspectMarker();
  ctx.scene.add(marker);

  el = {
    btnBasis: document.getElementById("btn-basis"),
    btnSynthesis: document.getElementById("btn-synthesis"),
    btnRoundtrip: document.getElementById("btn-roundtrip"),
    btnBlob: document.getElementById("btn-blob"),
    btnSphere: document.getElementById("btn-sphere"),
    btnNodes: document.getElementById("btn-nodes"),
    btnSlice: document.getElementById("btn-slice"),
    btnBedit: document.getElementById("btn-bedit"),
    btnAnim: document.getElementById("btn-anim"),
    animSpeed: document.getElementById("anim-speed"),
    groupAnim: document.getElementById("group-anim"),
    groupSamples: document.getElementById("group-samples"),
    selL: document.getElementById("sel-l"),
    nRange: document.getElementById("n-range"),
    nVal: document.getElementById("n-val"),
    navGrid: document.getElementById("nav-grid"),
    sidebar: document.getElementById("sidebar"),
    readout: document.getElementById("readout"),
    plots: document.getElementById("plots"),
    capLeg: document.getElementById("cap-legendre"),
    capPhi: document.getElementById("cap-phi"),
    cbarMag: document.getElementById("cbar-mag"),
    rtcaption: document.getElementById("rtcaption"),
    inspector: document.getElementById("inspector"),
    system: document.getElementById("system")
  };
  el.gLeg = setupPlot(document.getElementById("plot-legendre"), PLOT_W, PLOT_H);
  el.gPhi = setupPlot(document.getElementById("plot-phi"), PLOT_W, PLOT_H);
  el.gCross = setupPlot(document.getElementById("plot-cross"), CROSS_W, CROSS_H);
  el.gSystem = setupPlot(document.getElementById("system-canvas"), SYS_W, SYS_H);

  el.btnBasis.onclick = ()=>{ state.bedit=false; state.view="basis"; rebuild(); refreshControls(); };
  el.btnSynthesis.onclick = ()=>{
    state.bedit=false; state.view="synthesis";
    if(state.coeffs.size===0) state.coeffs.set("1,0", 1.0);
    rebuild(); refreshControls();
  };
  el.btnRoundtrip.onclick = ()=>{
    state.view="roundtrip";
    if(state.coeffs.size===0) state.coeffs.set("1,0", 1.0);
    rebuild(); refreshControls();
  };
  el.btnBlob.onclick = ()=>{ state.mode="blob"; rebuild(); refreshControls(); };
  el.btnSphere.onclick = ()=>{ state.mode="sphere"; rebuild(); refreshControls(); };
  el.btnNodes.onclick = ()=>{ state.nodes=!state.nodes; rebuild(); refreshControls(); };
  el.btnSlice.onclick = ()=>{ state.slice=!state.slice; rebuild(); refreshControls(); };
  el.btnBedit.onclick = ()=>{ if(state.bedit) exitBEdit(); else enterBEdit(); };
  el.btnAnim.onclick = ()=>{ state.animate=!state.animate; rebuild(); refreshControls(); };
  el.animSpeed.oninput = ()=>{ state.animSpeed = parseFloat(el.animSpeed.value); };
  el.selL.onchange = ()=>{
    state.maxL = parseInt(el.selL.value);
    if(state.l > state.maxL){ state.l = state.maxL; state.m = 0; }
    if(state.bedit) state.bvec = sampleBVector(state.n, state.coeffs, state.maxL);
    rebuild(); refreshControls();
  };
  el.nRange.oninput = ()=>{
    state.n = parseInt(el.nRange.value);
    el.nVal.textContent = state.n;
    if(state.bedit){ state.bvec = sampleBVector(state.n, state.coeffs, state.maxL); buildBSliders(); }
    rebuild();
    refreshDynamicReadout();
  };

  ctx.renderer.domElement.addEventListener("mousemove", onInspectMove);
  ctx.renderer.domElement.addEventListener("mouseleave", onInspectLeave);

  initLSelect();
  rebuild();
  refreshControls();
});