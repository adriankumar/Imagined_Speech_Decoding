//channels panel: exclude or reference resolved channels through the electrode popup, reset to restore them
//exclusion and reference re-resolve through the env's change_source and change_target_ref, the env owns it
(function(){
  function el(id){ return document.getElementById(id); }

  //open the electrode selector popup in the given context (exclude or reference)
  async function openSelector(context){
    await fetch("/open_electrode_selector", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({context})
    });
  }

  //average reference is a direct action, no selection needed
  async function setAverage(){
    const res = await fetch("/electrode_action", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({action:"average", channels:[]})
    });
    const snap = await res.json();
    if(!snap.error) GUI.apply(snap);
  }

  //reset manual exclusions, restoring the dropped resolved channels
  async function resetExclusions(){
    const res = await fetch("/reset_exclusions", {method:"POST"});
    const snap = await res.json();
    if(!snap.error) GUI.apply(snap);
  }

  //a labelled summary line
  function line(key, value){ return `<div><span class="ch-key">${key}</span> ${value}</div>`; }

  function render(snap){
    const ch = snap.channels;
    const ref = Array.isArray(snap.config.target_ref) ? snap.config.target_ref.join(", ") : snap.config.target_ref;
    el("ch-body").innerHTML = `
      <div class="ch-actions">
        <button id="ch-exclude" class="seg">exclude...</button>
        <button id="ch-ref" class="seg">set reference...</button>
      </div>
      <div class="ch-summary">
        ${line("resolved", ch.n_resolved)}
        ${line("manual-excluded", ch.manual_excluded.length ? ch.manual_excluded.join(", ") : "none")}
        ${line("reference", ref)}
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