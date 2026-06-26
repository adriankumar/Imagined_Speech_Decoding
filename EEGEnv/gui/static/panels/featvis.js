//feature visualisation panel: displays the feature image, image or stack kind
//it does not fetch its own window, the frame coordinator pushes images through setImage
(function(){
  let kind = "image";
  let locked = false;

  function el(id){ return document.getElementById(id); }

  function setActive(){
    el("fv-image").classList.toggle("on", kind === "image");
    el("fv-stack").classList.toggle("on", kind === "stack");
  }

  //exposed for the frame coordinator and the playback loop to push frames and read the current kind
  window.FeatureView = {
    setImage(url){ el("fv-body").innerHTML = `<img id="fv-img" src="${url}" alt="feature window">`; },
    kind(){ return kind; }
  };

  //change kind and ask the coordinator for a fresh frame, unless playback is driving frames
  function setKind(k){
    kind = k; setActive();
    if(!locked && window.Frame) Frame.refresh();
  }

  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)){
      el("fv-body").innerHTML = `<div class="fv-blank">no recording loaded</div>`;
      locked = false;
      return;
    }
    locked = !!snap.locked;
  }

  document.addEventListener("DOMContentLoaded", ()=>{
    el("fv-image").addEventListener("click", ()=> setKind("image"));
    el("fv-stack").addEventListener("click", ()=> setKind("stack"));
  });

  GUI.register("featviz", onState);
})();