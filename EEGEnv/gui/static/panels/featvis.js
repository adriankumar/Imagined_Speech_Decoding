//feature visualisation panel: renders the current window as a server-side png, image or stack
//while locked (playback), the playback loop owns the image and pushes frames through FeatureView
(function(){
  let kind = "image";
  let locked = false;

  function el(id){ return document.getElementById(id); }

  //reflect the active kind on the toggle buttons
  function setActive(){
    el("fv-image").classList.toggle("on", kind === "image");
    el("fv-stack").classList.toggle("on", kind === "stack");
  }

  //request a seek/peek render of the current window and swap the image source
  async function render(){
    const body = el("fv-body");
    const res = await fetch("/render", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({kind})
    });
    const data = await res.json();
    if(data.error){ body.innerHTML = `<div class="fv-blank">render error</div>`; return; }
    body.innerHTML = `<img id="fv-img" src="${data.render_url}" alt="feature window">`;
  }

  //change view kind, re-render only when not locked, otherwise the next playback frame picks it up
  function setKind(k){ kind = k; setActive(); if(!locked) render(); }

  //exposed for the playback loop to push live frames and to read the current view kind
  window.FeatureView = {
    setImage(url){ el("fv-body").innerHTML = `<img id="fv-img" src="${url}" alt="feature window">`; },
    kind(){ return kind; }
  };

  //snapshot-driven: render when loaded and unlocked, blank when not loaded or stale
  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("fv-body").innerHTML = `<div class="fv-blank">no recording loaded</div>`;
      locked = false;
      return;
    }
    locked = !!snap.locked;
    if(!locked) render();
  }

  document.addEventListener("DOMContentLoaded", ()=>{
    el("fv-image").addEventListener("click", ()=> setKind("image"));
    el("fv-stack").addEventListener("click", ()=> setKind("stack"));
  });

  GUI.register("featviz", onState);
})();