//decoding simulation: the true delta decode at the current window, current, delta, decoded, true next
//field form only, the delta-as-matrix view lives with the harmonic system where c is that matrix
//it never fetches its own window, the frame coordinator pushes decode payloads through setDecode
(function(){
  function el(id){ return document.getElementById(id); }

  function ensureContent(){
    if(el("dec-img")) return;
    el("dec-body").innerHTML = `<img id="dec-img" alt="decode simulation">`;
  }

  window.DecodeView = {
    deltaMode(){ return "field"; },          //decode always renders the four-column field form now
    setDecode(d){ ensureContent(); el("dec-img").src = d.render_url; }
  };

  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("dec-body").innerHTML = `<div class="dec-blank">no recording loaded</div>`;
      return;
    }
    ensureContent();
  }

  GUI.register("decode", onState);
})();