//geometry controls: the harmonic degree L as a stepper clamped to the resolvable ceiling, and the margin slider
//L rebuilds the basis Y, margin rebuilds the interpolation operator M, both through the env on /set
(function(){
  //slider bounds for the interpolation margin, adjust to the env's actual margin range
  const MARGIN_MIN = 0.1, MARGIN_MAX = 1.0;

  function el(id){ return document.getElementById(id); }

  //post a geometry edit and push the returned snapshot through the registry
  async function setArg(name, value){
    const res = await fetch("/set", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({name, value})
    });
    const snap = await res.json();
    if(!snap.error) GUI.apply(snap);
  }

  function render(snap){
    const L = snap.config.L, ceil = snap.config.L_ceiling, modes = (L + 1) * (L + 1);
    const margin = snap.config.margin;

    el("geo-body").innerHTML = `
      <div class="geo-row">
        <span class="geo-label">harmonic degree L</span>
        <div class="geo-stepper">
          <button id="geo-ldec" class="seg">\u2212</button>
          <span class="geo-lval">${L}</span>
          <button id="geo-linc" class="seg">+</button>
        </div>
      </div>
      <div class="geo-sub">(L+1)\u00b2 = ${modes} modes \u00b7 max ${ceil} for this montage</div>
      <div class="geo-row">
        <span class="geo-label">interpolation margin</span>
        <input type="range" id="geo-margin" min="${MARGIN_MIN}" max="${MARGIN_MAX}" step="0.05" value="${margin}">
        <span class="geo-mval" id="geo-mval">${(+margin).toFixed(2)}</span>
      </div>`;

    el("geo-body").classList.toggle("locked", !!snap.locked);

    el("geo-ldec").disabled = L <= 0;
    el("geo-linc").disabled = L >= ceil;
    el("geo-ldec").addEventListener("click", () => { if(L > 0) setArg("L_degree", L - 1); });
    el("geo-linc").addEventListener("click", () => { if(L < ceil) setArg("L_degree", L + 1); });

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