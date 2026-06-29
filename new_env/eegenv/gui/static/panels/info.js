//read-only info modal, opened from the loading panel, renders the snapshot meta, stays live while open
(function(){
  let open = false;
  let last = null;     //the most recent snapshot, so opening renders immediately without a fetch

  function el(id){ return document.getElementById(id); }

  //hh|mm|ss breakdown alongside the raw seconds, formatted frontend-side from duration_s
  function formatDuration(s){
    const t = Math.floor(s);
    const hr = String(Math.floor(t / 3600)).padStart(2, "0");
    const min = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
    const sec = t % 60;
    return `${hr} hr | ${min} min | ${sec}s  (${s.toFixed(1)} s)`;
  }

  function row(label, value){
    return `<div class="info-row"><span>${label}</span><span>${value}</span></div>`;
  }
  function listOrNone(a){ return a && a.length ? a.join(", ") : "none"; }

  function render(snap){
    const body = el("info-body");
    if(!snap || !snap.loaded){ body.innerHTML = `<div class="info-blank">no recording loaded</div>`; return; }

    const r = snap.recording, c = snap.config, ch = snap.channels, s = snap.shapes, f = snap.features;
    const ref = Array.isArray(c.target_ref) ? c.target_ref.join(", ") : c.target_ref;

    body.innerHTML = [
      row("source", r.source.split(/[\\/]/).pop()),
      row("montage", r.montage),
      row("sampling rate", r.sfreq.toFixed(1) + " hz"),
      row("samples", r.time_points),
      row("duration", formatDuration(r.duration_s)),
      row("target re-reference", ref),
      row("source ref", c.ref_scheme),
      row("L degree", c.L),
      row("total coefficients", snap.n_modes),
      row("active features", f.n_active),
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

  window.InfoModal = {
    open(){ open = true; el("info-modal").style.display = "flex"; render(last); },
    close(){ open = false; el("info-modal").style.display = "none"; }
  };

  //snapshot-driven: keep the modal current while it is open, ignore stale dispatches
  function onState(snap, opts){
    last = snap;
    if(open && !(opts && opts.stale)) render(snap);
  }

  document.addEventListener("DOMContentLoaded", ()=>{
    const c = el("info-close");
    if(c) c.addEventListener("click", ()=> InfoModal.close());
  });

  GUI.register("info", onState);
})();