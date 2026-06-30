//frame coordinator: one /frame fetch produces the vis payload and the decode payload, handed to both panels
//the feature panel and the decode panel never fetch their own window, they receive it from here
(function(){
  //inert until a consumer exists, so earlier stages never trigger renders that nothing displays
  function hasConsumers(){ return window.FeatureView || window.DecodeView; }

  //distribute one frame to the panels that render it
  function applyFrame(f){
    if(window.FeatureView && f.vis) FeatureView.setVis(f.vis);
    if(window.DecodeView && f.decode) DecodeView.setDecode(f.decode);
  }

  //fetch a read-only frame at the cursor and distribute it, the active vis mode picks what the server computes
  async function refresh(){
      if(!hasConsumers()) return;
      const res = await fetch("/frame", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          vis_mode: window.FeatureView ? FeatureView.mode() : "image",
          delta_mode: window.DecodeView ? DecodeView.deltaMode() : "field"
        })
      });
      const f = await res.json();
      if(!f.error) applyFrame(f);
    }

  window.Frame = {refresh, applyFrame};

  //snapshot-driven: refresh the frame on load, seek, or an edit, when not locked (playback drives instead)
  function onState(snap, opts){
    if(!snap || !snap.loaded || (opts && opts.stale)) return;
    if(!snap.locked) refresh();
  }

  GUI.register("frame", onState);
})();