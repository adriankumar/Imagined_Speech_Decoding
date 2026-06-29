//loading panel: native edf pick, montage dropdown, source ref, the load state machine, unresolved modal, info button
//the load button is the safe gate: disabled until a file is chosen, locks on a successful load, reopens on any edit
(function(){
  let pickedFile = null;
  let dirty = false;     //the selection differs from what is loaded, load is enabled only when dirty
  let loaded = false;    //a successful load has happened, info is enabled from then on
  let locked = false;    //playback freezes the loading controls

  function el(id){ return document.getElementById(id); }


//load needs a picked file and a pending change, info needs a clean loaded state, all frozen during playback
  function updateButtons(){
    el("ld-load").disabled = locked || !(pickedFile && dirty);
    el("ld-info").disabled = locked || !loaded || dirty;
    el("ld-pick").disabled = locked;
    el("ld-montage").disabled = locked;
    el("ld-ref").disabled = locked;
  }

  //a change to the selection means a reload is pending, the load button reopens and downstream blanks
  function markDirty(){
    dirty = true;
    el("ld-status").textContent = "click load to apply";
    updateButtons();
    GUI.markStale();
  }

  //populate the montage dropdown from the env's montage registry
  async function loadMontages(){
    const res = await fetch("/montages");
    const list = await res.json();
    const sel = el("ld-montage");
    sel.innerHTML = "";
    for(const m of list){
      const opt = document.createElement("option");
      opt.value = m; opt.textContent = m;
      sel.appendChild(opt);
    }
  }

  //native file dialog opened server-side, falls back to a typed path in a plain browser
  async function pickFile(){
    let path = null;
    if(window.pywebview){
      const res = await fetch("/pick_edf", {method:"POST"});
      const data = await res.json();
      path = data.path;
    } else {
      path = prompt("edf file path");
    }
    if(path){
      pickedFile = path;
      el("ld-file").textContent = path.split(/[\\/]/).pop();
      markDirty();
    }
  }

  //post a load attempt, auto_exclude drives whether unresolved channels are dropped without a prompt
  async function doLoad(autoExclude){
    if(!pickedFile){ el("ld-status").textContent = "pick an edf file first"; return; }
    const montage = el("ld-montage").value;
    const refScheme = el("ld-ref").value.trim() || "average";
    el("ld-status").textContent = "loading...";

    const res = await fetch("/load", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({source:pickedFile, montage, ref_scheme:refScheme, auto_exclude:autoExclude})
    });
    const data = await res.json();

    if(data.ok){
      hideModal();
      loaded = true; dirty = false;
      el("ld-status").textContent = "loaded";
      el("ld-file").textContent = "";   //the path now lives in the info panel as source
      updateButtons();
      GUI.apply(data.snapshot);
      return;
    }
    if(data.kind === "unresolved"){
      showModal(data.unresolved, data.montage);
      el("ld-status").textContent = "";
      return;
    }
    el("ld-status").textContent = "error: " + data.message;
  }

  //unresolved modal: list the names, offer auto-exclude or change montage
  function showModal(unresolved, montage){
    el("ld-modal-list").textContent = unresolved.join(", ");
    el("ld-modal-montage").textContent = montage;
    el("ld-modal").style.display = "flex";
  }
  function hideModal(){ el("ld-modal").style.display = "none"; }

  //lock-only snapshot handling, the load flow itself is user-driven not snapshot-driven
  function onState(snap){
    locked = !!(snap && snap.locked);
    if(snap && snap.loaded) loaded = true;
    updateButtons();
  }

  document.addEventListener("DOMContentLoaded", ()=>{
    loadMontages();
    el("ld-pick").addEventListener("click", pickFile);
    el("ld-load").addEventListener("click", ()=> doLoad(false));
    el("ld-info").addEventListener("click", ()=> window.InfoModal && InfoModal.open());
    el("ld-montage").addEventListener("change", ()=>{ if(pickedFile) markDirty(); });
    el("ld-ref").addEventListener("input", ()=>{ if(pickedFile) markDirty(); });
    el("ld-modal-auto").addEventListener("click", ()=> doLoad(true));
    el("ld-modal-change").addEventListener("click", ()=>{
      hideModal();
      el("ld-status").textContent = "pick another montage and load";
      markDirty();
    });
    updateButtons();
  });

  GUI.register("loading", onState);
})();