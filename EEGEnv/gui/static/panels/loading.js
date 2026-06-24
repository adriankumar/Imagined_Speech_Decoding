//loading panel: native edf pick, montage dropdown, source ref, load button, unresolved modal
(function(){
  let pickedFile = null;

  //short id helper
  function el(id){ return document.getElementById(id); }

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


//native file dialog opened server-side over a route, falls back to a typed path in a plain browser
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

  //changing the file or montage means the loaded config is stale until load is pressed again
  function markDirty(){
    el("ld-status").textContent = "click load to apply";
    GUI.markStale();
  }

  //post a load attempt, auto_exclude drives whether unresolved channels are dropped
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

    if(data.ok){ hideModal(); el("ld-status").textContent = "loaded"; GUI.refresh(); return; }
    if(data.kind === "unresolved"){ showModal(data.unresolved, data.montage); el("ld-status").textContent = ""; return; }
    el("ld-status").textContent = "error: " + data.message;
  }

  //unresolved modal: list the names, offer auto-exclude or change montage
  function showModal(unresolved, montage){
    el("ld-modal-list").textContent = unresolved.join(", ");
    el("ld-modal-montage").textContent = montage;
    el("ld-modal").style.display = "flex";
  }
  function hideModal(){ el("ld-modal").style.display = "none"; }

  document.addEventListener("DOMContentLoaded", ()=>{
    loadMontages();
    el("ld-pick").addEventListener("click", pickFile);
    el("ld-load").addEventListener("click", ()=> doLoad(false));
    el("ld-montage").addEventListener("change", markDirty);
    el("ld-ref").addEventListener("input", markDirty);
    el("ld-modal-auto").addEventListener("click", ()=> doLoad(true));
    el("ld-modal-change").addEventListener("click", ()=>{
      hideModal();
      el("ld-status").textContent = "pick another montage and load";
    });
  });
})();