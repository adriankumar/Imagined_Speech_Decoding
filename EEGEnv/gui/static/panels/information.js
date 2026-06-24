//information panel: read-only render of the snapshot, blanks when nothing is loaded or stale
(function(){
  //labelled row markup
  function row(label, value){
    return `<div class="info-row"><span>${label}</span><span>${value}</span></div>`;
  }

  //a list value falls back to none when empty
  function listOrNone(arr){ return arr && arr.length ? arr.join(", ") : "none"; }

  function render(snap, opts){
    const body = document.getElementById("info-body");
    if(!snap || !snap.loaded || (opts && opts.stale)){
      body.innerHTML = `<div class="info-blank">no recording loaded</div>`;
      return;
    }

    const r = snap.recording, c = snap.config, ch = snap.channels, s = snap.shapes;
    const ref = Array.isArray(c.target_ref) ? c.target_ref.join(", ") : c.target_ref;

    body.innerHTML = [
      row("source", r.source.split(/[\\/]/).pop()),
      row("montage", r.montage),
      row("sampling rate", r.sfreq.toFixed(1) + " hz"),
      row("samples", r.time_points),
      row("duration", r.duration_s.toFixed(1) + " s"),
      row("target ref", ref),
      row("source ref", c.ref_scheme),
      row("L", c.L),
      row("window size", c.window_size + " samples"),
      row("image", c.img_res[0] + " x " + c.img_res[1]),
      row("resolved channels", ch.n_resolved),
      row("auto-excluded", listOrNone(ch.auto_excluded)),
      row("default-excluded", listOrNone(ch.default_excluded)),
      row("manual-excluded", listOrNone(ch.manual_excluded)),
      row("M shape", s.M.join(" x ")),
      row("Y shape", s.Y.join(" x ")),
    ].join("");
  }

  GUI.register("information", render);
})();