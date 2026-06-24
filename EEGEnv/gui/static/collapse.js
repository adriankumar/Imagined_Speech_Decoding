//collapsible panel chrome: a caret in the title hides the body, the grid reclaims the space
//collapsed state is persisted on the session so a reload restores the layout
(function(){
  //inject a caret into each collapsible panel's title and wire the toggle, once
  function init(){
    document.querySelectorAll("[data-collapsible]").forEach(section => {
      const title = section.querySelector(".panel-title");
      if(!title || title.querySelector(".collapse-caret")) return;
      const caret = document.createElement("button");
      caret.className = "collapse-caret";
      caret.textContent = section.classList.contains("collapsed") ? "\u25B8" : "\u25BE";
      caret.addEventListener("click", () => toggle(section));
      title.prepend(caret);
    });
  }

  //flip the collapsed class, update the caret, persist to the session
  function toggle(section){
    const collapsed = section.classList.toggle("collapsed");
    const caret = section.querySelector(".collapse-caret");
    if(caret) caret.textContent = collapsed ? "\u25B8" : "\u25BE";
    fetch("/toggle_panel", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({panel: section.id, collapsed})
    });
  }

  //restore collapsed state from the snapshot on each dispatch, so a reload keeps the layout
  function restore(snap){
    if(!snap || !snap.collapsed) return;
    const set = new Set(snap.collapsed);
    document.querySelectorAll("[data-collapsible]").forEach(section => {
      const collapsed = set.has(section.id);
      section.classList.toggle("collapsed", collapsed);
      const caret = section.querySelector(".collapse-caret");
      if(caret) caret.textContent = collapsed ? "\u25B8" : "\u25BE";
    });
  }

  document.addEventListener("DOMContentLoaded", init);
  GUI.register("collapse", restore);
})();