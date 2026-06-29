//channels panel: exclude or re-reference resolved channels through the electrode popup, reset restores them
//the env owns the re-resolve, this panel only opens the popup and summarises the current channel state
(function(){
  function el(id){ return document.getElementById(id); }

  //open the electrode selector popup in the given context (exclude or reference)
  async function openSelector(context){
    await fetch("/open_electrode_selector", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({context})
    });
  }

  //a labelled summary line
  function line(key, value){ return `<div><span class="ch-key">${key}</span> ${value}</div>`; }

  function render(snap){
    const ch = snap.channels;
    const ref = Array.isArray(snap.config.target_ref) ? snap.config.target_ref.join(", ") : snap.config.target_ref;
    el("ch-body").innerHTML = `
      <div class="ch-actions">
        <button id="ch-exclude" class="seg">exclude...</button>
        <button id="ch-ref" class="seg">set target re-reference...</button>
      </div>
      <div class="ch-summary">
        ${line("resolved", ch.n_resolved)}
        ${line("manual-excluded", ch.manual_excluded.length ? ch.manual_excluded.join(", ") : "none")}
        ${line("target re-reference", ref)}
      </div>`;

    el("ch-body").classList.toggle("locked", !!snap.locked);
    el("ch-exclude").addEventListener("click", ()=> openSelector("exclude"));
    el("ch-ref").addEventListener("click", ()=> openSelector("reference"));
  }

  //snapshot-driven: build when loaded, blank when not loaded or stale
  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("ch-body").innerHTML = `<div class="ch-blank">no recording loaded</div>`;
      return;
    }
    render(snap);
  }

  GUI.register("channels", onState);
})();