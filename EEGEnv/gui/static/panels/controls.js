//feature stack controls: base toggles and weights, ema accumulation, 
//every edit posts to /set, the panel rebuilds from the snapshot, locks itself during playback
(function(){
  //the eight base features the constructor exposes, in panel order
  const BASE = ["raw", "median", "iqr", "mobility", "complexity", "raw_lag", "median_lag", "iqr_lag"];
  //the five base features that support ema accumulation
  const ACCUM = ["raw", "median", "iqr", "mobility", "complexity"];

  function el(id){ return document.getElementById(id); }

  //post a single named edit, the response is the new snapshot which redraws every panel
  async function setArg(name, value){
    const res = await fetch("/set", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({name, value})
    });
    const snap = await res.json();
    if(snap.error){ return; }
    GUI.apply(snap);  //push the fresh snapshot through the registry without an extra fetch
  }

  //a checkbox row with a numeric weight field, used for the base features
  function toggleWeightRow(name, on, weight){
    return `<div class="ctrl-row">
      <label><input type="checkbox" data-toggle="${name}" ${on ? "checked" : ""}> ${name}</label>
      <input type="number" step="0.1" min="0" data-weight="${name}" value="${weight.toFixed(4)}">
    </div>`;
  }

  //a checkbox row with an alpha slider, used for the ema accumulation features
  function accumRow(name, on, alpha){
    return `<div class="ctrl-row">
      <label><input type="checkbox" data-accum="${name}" ${on ? "checked" : ""}> ${name}</label>
      <input type="range" min="0.01" max="1" step="0.01" data-alpha="${name}" value="${alpha}">
      <span class="ctrl-val" data-alphaval="${name}">${(+alpha).toFixed(2)}</span>
    </div>`;
  }

  //build the whole panel from the snapshot's feature state
  function render(snap){
    const f = snap.features
    const base = BASE.map(n => toggleWeightRow(n, f.toggle[n], f.weight[n])).join("");
    const accum = ACCUM.map(n => accumRow(n, f.accum[n], f.alpha[n])).join("");

    el("ctrl-body").innerHTML = `
      <div class="ctrl-section">
        <div class="ctrl-head">base features</div>${base}
      </div>
      <div class="ctrl-section">
        <div class="ctrl-head">ema accumulation</div>${accum}
      </div>`;

    el("ctrl-body").classList.toggle("locked", !!snap.locked);
    wire(snap);
  }

  //attach handlers after each rebuild, reading the current dom state into the right /set payload
  function wire(snap){
    const body = el("ctrl-body");

    body.querySelectorAll("[data-toggle]").forEach(cb =>
      cb.addEventListener("change", () => setArg("feature_toggle", {[cb.dataset.toggle]: cb.checked})));

    body.querySelectorAll("[data-weight]").forEach(inp =>
      inp.addEventListener("change", () => setArg("feature_weight", {[inp.dataset.weight]: parseFloat(inp.value)})));

    //an ema toggle and its alpha commit together so change_feature_accum gets a coherent pair
    body.querySelectorAll("[data-accum]").forEach(cb =>
      cb.addEventListener("change", () => {
        const name = cb.dataset.accum;
        const alpha = parseFloat(body.querySelector(`[data-alpha="${name}"]`).value);
        setArg("feature_accum", {accum: {[name]: cb.checked}, alpha: {[name]: alpha}});
      }));
    body.querySelectorAll("[data-alpha]").forEach(rng =>
      rng.addEventListener("input", () => body.querySelector(`[data-alphaval="${rng.dataset.alpha}"]`).textContent = (+rng.value).toFixed(2)));
    body.querySelectorAll("[data-alpha]").forEach(rng =>
      rng.addEventListener("change", () => {
        const name = rng.dataset.alpha;
        const on = body.querySelector(`[data-accum="${name}"]`).checked;
        setArg("feature_accum", {accum: {[name]: on}, alpha: {[name]: parseFloat(rng.value)}});
      }));
  }

  //snapshot-driven: build when loaded, blank when not loaded or stale
  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("ctrl-body").innerHTML = `<div class="ctrl-blank">no recording loaded</div>`;
      return;
    }
    render(snap);
  }

  GUI.register("controls", onState);
})();