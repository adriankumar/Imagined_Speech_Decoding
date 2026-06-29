//thin orchestration: panels register a render(snapshot, opts) callback, main polls state and dispatches
window.GUI = (function(){
  const panels = {};
  let lastVersion = -1;
  let stale = false;

  //register a panel renderer by name
  function register(name, renderFn){ panels[name] = renderFn; }

  //dispatch a snapshot to every registered panel
  function dispatch(snap, opts){
    for(const name in panels) panels[name](snap, opts);
  }

  //fetch the current snapshot and render every panel fresh
  async function refresh(){
    const res = await fetch("/state");
    const snap = await res.json();
    lastVersion = snap.state_version;
    stale = false;
    dispatch(snap, {stale:false});
  }

  //blank downstream displays so the user sees a reload is needed
  function markStale(){
    stale = true;
    dispatch(null, {stale:true});
  }

  //poll for external state changes, e.g. a committed channel edit from a popup
  async function poll(){
    try{
      const res = await fetch("/state");
      const snap = await res.json();
      if(snap.state_version !== lastVersion && !stale){
        lastVersion = snap.state_version;
        dispatch(snap, {stale:false});
      }
    }catch(e){}
  }

//push an already-fetched snapshot through the registry, used after a /set or /load response
  function apply(snap){
    lastVersion = snap.state_version;
    stale = false;
    dispatch(snap, {stale:false});
  }

  return {register, refresh, markStale, apply, start(){ setInterval(poll, 600); }};
})();

//pywebview lifecycle, with a plain-browser fallback for quick checks outside the app
window.addEventListener("pywebviewready", ()=> GUI.refresh().then(()=> GUI.start()));
if(!window.pywebview){ setTimeout(()=> GUI.refresh().then(()=> GUI.start()), 200); }