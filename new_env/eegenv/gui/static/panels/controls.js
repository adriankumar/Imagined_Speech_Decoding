//feature stack controls: the eight features as base-above-lag columns, a weight under each name, every edit posts to /set
//a lag depends on its base: the panel disables a lag whose base is off, and turning a base off cascades its lag off
(function(){
  //base features in column order, each paired with the lag that sits directly beneath it
  const BASES = ["median", "iqr", "mobility", "complexity"];
  const LAG = {median: "median_lag", iqr: "iqr_lag", mobility: "mobility_lag", complexity: "complexity_lag"};

  function el(id){ return document.getElementById(id); }

  //post a single named edit, the response snapshot redraws every panel
  //on a rejected edit resync from the env so an optimistic checkbox never lies about the real state
  async function setArg(name, value){
    const res = await fetch("/set", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({name, value})
    });
    const snap = await res.json();
    if(snap.error){ GUI.refresh(); return; }
    GUI.apply(snap);
  }

  //a feature unit, the name with its toggle and the weight directly beneath, weight shown to two decimals
  function featUnit(name, on, weight, disabled){
    return `<div class="ctrl-feat${disabled ? " ctrl-disabled" : ""}">
      <label><input type="checkbox" data-toggle="${name}" ${on ? "checked" : ""} ${disabled ? "disabled" : ""}> ${name}</label>
      <input type="number" step="0.01" min="0" data-weight="${name}" value="${(+weight).toFixed(2)}" ${disabled ? "disabled" : ""}>
    </div>`;
  }

  //a column, the base unit above its lag unit, the lag greyed and disabled while the base is off
  function column(base, t, w){
    const lag = LAG[base];
    return `<div class="ctrl-col">
      ${featUnit(base, t[base], w[base], false)}
      ${featUnit(lag, t[lag], w[lag], !t[base])}
    </div>`;
  }

  function render(snap){
    const t = snap.features.toggles, w = snap.features.weights;
    el("ctrl-body").innerHTML = `<div class="ctrl-grid">${BASES.map(b => column(b, t, w)).join("")}</div>`;
    el("ctrl-body").classList.toggle("locked", !!snap.locked);
    wire();
  }

  //attach handlers after each rebuild
  function wire(){
    const body = el("ctrl-body");
    body.querySelectorAll("[data-toggle]").forEach(cb =>
      cb.addEventListener("change", () => onToggle(cb.dataset.toggle, cb.checked)));
    body.querySelectorAll("[data-weight]").forEach(inp =>
      inp.addEventListener("change", () => setArg("feature_weight", {[inp.dataset.weight]: parseFloat(inp.value)})));
  }

  //turning a base off would orphan its lag, so cascade the lag off in the same commit to satisfy the dependency
  function onToggle(name, checked){
    if(name in LAG && !checked){
      setArg("feature_toggle", {[name]: false, [LAG[name]]: false});
    } else {
      setArg("feature_toggle", {[name]: checked});
    }
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