//frame coordinator: one /frame fetch produces the feature image and the sh data, handed to both panels
//the feature panel and the sh panel never fetch their own window, they receive it from here
(function(){
  //distribute one frame to the panels that render it
  function applyFrame(f){
    if(window.FeatureView) FeatureView.setImage(f.render_url);
    if(window.SHView) SHView.setData(f.sh);
  }

  //fetch a read-only frame at the cursor and distribute it, skipped while locked (playback drives instead)
  async function refresh(){
    const res = await fetch("/frame", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({kind: window.FeatureView ? FeatureView.kind() : "image"})
    });
    const f = await res.json();
    if(!f.error) applyFrame(f);
  }

  window.Frame = {refresh, applyFrame};

  //snapshot-driven: refresh the frame on load, seek, or an edit, when not locked
  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)) return;
    if(!snap.locked) refresh();
  }

  GUI.register("frame", onState);
})();