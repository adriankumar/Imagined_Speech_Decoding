//geometry controls: harmonic degree L as a dropdown over the resolvable range, and the interpolation margin
//L rebuilds the basis Y, margin rebuilds the operator M, both committed through the env on /set
(function(){
  function el(id){ return document.getElementById(id); }

  async function setArg(name, value){
    const res = await fetch("/set", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({name, value})
    });
    const snap = await res.json();
    if(snap.error){ GUI.refresh(); return; }
    GUI.apply(snap);
  }

  function render(snap){
    const c = snap.config, n = snap.channels.n_resolved;
    const L = c.L, ceil = c.L_ceiling, rec = c.recommended_L, modes = (L + 1) * (L + 1);

    //the dropdown offers every resolvable degree 0..ceiling, all valid by construction so no option is disabled
    const opts = [];
    for(let d = 0; d <= ceil; d++) opts.push(`<option value="${d}" ${d === L ? "selected" : ""}>${d}</option>`);

    el("geo-body").innerHTML = `
      <div class="geo-row">
        <span class="geo-label">harmonic degree L</span>
        <select id="geo-l">${opts.join("")}</select>
      </div>
      <div class="geo-sub">(L+1)\u00b2 = ${modes} modes \u00b7 max L ${ceil} for this montage</div>
      <div class="geo-sub">${n} electrodes \u00b7 recommended L = ${rec}</div>
      <div class="geo-row">
        <span class="geo-label">interpolation margin</span>
        <input type="range" id="geo-margin" min="0.1" max="1.0" step="0.05" value="${c.margin}">
        <span class="geo-mval" id="geo-mval">${(+c.margin).toFixed(2)}</span>
      </div>`;

    el("geo-body").classList.toggle("locked", !!snap.locked);
    el("geo-l").addEventListener("change", () => setArg("L_degree", parseInt(el("geo-l").value)));
    //the slider tracks live, the rebuild commits on release
    el("geo-margin").addEventListener("input", () => el("geo-mval").textContent = (+el("geo-margin").value).toFixed(2));
    el("geo-margin").addEventListener("change", () => setArg("margin", parseFloat(el("geo-margin").value)));
  }

  //snapshot-driven: build when loaded, blank when not loaded or stale
  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("geo-body").innerHTML = `<div class="geo-blank">no recording loaded</div>`;
      return;
    }
    render(snap);
  }

  GUI.register("geometry", onState);
})();