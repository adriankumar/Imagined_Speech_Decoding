//decoding simulation: inject a random coefficient delta of model shape and show before, delta, after
//the decode pathway runs in the env, Y^T then M onto a copy of the current window, this panel only triggers it
(function(){
  let seed = null, scale = 1.0, active = false, locked = false;

  function el(id){ return document.getElementById(id); }

  //run the simulation at the current window with the held seed and scale
  async function run(){
    if(seed === null) return;
    const res = await fetch("/decode_sim", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({scale, seed})
    });
    const d = await res.json();
    if(!d.error) el("dec-img-wrap").innerHTML = `<img src="${d.render_url}" alt="decode simulation">`;
  }

  function buildDom(){
    el("dec-body").innerHTML = `
      <div class="dec-hint">a random coefficient delta of model shape, decoded through Y^T then M onto a copy of the current window</div>
      <div class="dec-controls">
        <button id="dec-rand" class="seg">randomise</button>
        <span class="dec-scale">scale
          <input type="range" id="dec-scale" min="0" max="3" step="0.05" value="1">
          <span class="dec-scaleval" id="dec-scaleval">1.00</span>
        </span>
      </div>
      <div id="dec-img-wrap"></div>`;
    //randomise draws a new delta direction, scale grows or shrinks the same draw
    el("dec-rand").addEventListener("click", () => { seed = Math.floor(Math.random() * 1e9); active = true; run(); });
    el("dec-scale").addEventListener("input", () => {
      scale = parseFloat(el("dec-scale").value);
      el("dec-scaleval").textContent = scale.toFixed(2);
    });
    el("dec-scale").addEventListener("change", () => { if(active) run(); });
  }

  //snapshot-driven: follow the window like the other panels once activated, idle until randomised
  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("dec-body").innerHTML = `<div class="dec-blank">no recording loaded</div>`;
      seed = null; active = false; locked = false;
      return;
    }
    if(!el("dec-img-wrap")) buildDom();
    locked = !!snap.locked;
    if(active && !locked) run();
  }

  GUI.register("decode", onState);
})();