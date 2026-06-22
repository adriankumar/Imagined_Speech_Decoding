//2d mini-plots: cross-section, latitude slice, longitude slice

//prepare a canvas for crisp drawing at css size w x h, returns its 2d context
function setupPlot(canvas, w, h){
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  canvas.width = Math.round(w*dpr);
  canvas.height = Math.round(h*dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return g;
}

//line plot with sign-coloured fill; fixedAmp keeps the vertical scale steady for the pulse
function drawCurve(g, w, h, samples, zeros, fixedAmp){
  g.clearRect(0, 0, w, h);
  const pad = 8;
  const x0 = pad, x1 = w - pad, yMid = h/2;

  let amp = fixedAmp;
  if(!(amp > 0)){ amp = 0; for(const s of samples) amp = Math.max(amp, Math.abs(s.v)); amp = amp || 1; }
  const yScale = (h/2 - pad) / amp;
  const px = t => x0 + (x1 - x0)*t;
  const py = v => yMid - Math.max(-(h/2-pad), Math.min(h/2-pad, v*yScale));

  //sign-coloured fill, one quad per interval
  for(let i=1;i<samples.length;i++){
    const a = samples[i-1], b = samples[i];
    const mid = (a.v + b.v)/2;
    g.fillStyle = mid >= 0 ? "rgba(59,111,208,0.18)" : "rgba(208,74,59,0.18)";
    g.beginPath();
    g.moveTo(px(a.t), yMid); g.lineTo(px(a.t), py(a.v));
    g.lineTo(px(b.t), py(b.v)); g.lineTo(px(b.t), yMid);
    g.closePath(); g.fill();
  }

  g.strokeStyle = "#e2ded7"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(x0, yMid); g.lineTo(x1, yMid); g.stroke();

  g.strokeStyle = "#6b6a66"; g.lineWidth = 1.4;
  g.beginPath();
  samples.forEach((s, i)=>{ const x = px(s.t), y = py(s.v); if(i===0) g.moveTo(x,y); else g.lineTo(x,y); });
  g.stroke();

  g.fillStyle = "#2b2926";
  for(const t of zeros){ const x = px(t); g.beginPath(); g.arc(x, yMid, 2.4, 0, 2*Math.PI); g.fill(); }
}

//deformed meridian cross-section: right side at phi0, left at phi0+pi, sign-coloured, with z-axis
function drawCrossSection(g, w, h, right, left, fixedAmp, zeros){
  g.clearRect(0, 0, w, h);
  const cx = w/2, cy = h/2;
  const S = Math.min(w, h)/2 - 12;

  let amp = fixedAmp;
  if(!(amp > 0)){
    amp = 0;
    for(const s of right) amp = Math.max(amp, Math.abs(s.v));
    for(const s of left) amp = Math.max(amp, Math.abs(s.v));
    amp = amp || 1;
  }

  //reference circle and vertical z-axis
  g.strokeStyle = "#eceae5"; g.lineWidth = 1;
  g.beginPath(); g.arc(cx, cy, S, 0, 2*Math.PI); g.stroke();
  g.beginPath(); g.moveTo(cx, cy - S - 4); g.lineTo(cx, cy + S + 4); g.stroke();
  g.fillStyle = "#9a978f"; g.font = "10px sans-serif"; g.textAlign = "center";
  g.fillText("z", cx, cy - S - 7);

  const ptsR = right.map(s=>{ const th = s.t*Math.PI, r = Math.abs(s.v)/amp; return {x: cx + S*r*Math.sin(th), y: cy - S*r*Math.cos(th), v:s.v}; });
  const ptsL = left.map(s=>{ const th = s.t*Math.PI, r = Math.abs(s.v)/amp; return {x: cx - S*r*Math.sin(th), y: cy - S*r*Math.cos(th), v:s.v}; });

  const drawSide = pts=>{
    for(let i=1;i<pts.length;i++){
      const mid = (pts[i-1].v + pts[i].v)/2;
      g.fillStyle = mid >= 0 ? "rgba(59,111,208,0.22)" : "rgba(208,74,59,0.22)";
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(pts[i-1].x, pts[i-1].y); g.lineTo(pts[i].x, pts[i].y); g.closePath(); g.fill();
    }
  };
  drawSide(ptsR); drawSide(ptsL);

  g.strokeStyle = "#6b6a66"; g.lineWidth = 1.4;
  g.beginPath();
  ptsR.forEach((p,i)=> i===0 ? g.moveTo(p.x,p.y) : g.lineTo(p.x,p.y));
  for(let i=ptsL.length-1;i>=0;i--) g.lineTo(ptsL[i].x, ptsL[i].y);
  g.closePath(); g.stroke();

  //nodes marked on the reference circle, showing the directions where the field is zero
  g.fillStyle = "#2b2926";
  for(const t of zeros){
    const th = t*Math.PI;
    g.beginPath(); g.arc(cx + S*Math.sin(th), cy - S*Math.cos(th), 2.4, 0, 2*Math.PI); g.fill();
  }
}

//sample the legendre part P_l^|m|(cos theta) across theta in [0, pi]
function legendreSamples(l, m, n){
  const am = Math.abs(m);
  const out = [];
  for(let i=0;i<=n;i++){ const t = i/n; out.push({t, v: assocLegendre(l, am, Math.cos(Math.PI*t))}); }
  return out;
}

//sample the azimuthal factor across phi in [0, 2pi]
function azimuthSamples(m, n){
  const am = Math.abs(m);
  const out = [];
  for(let i=0;i<=n;i++){
    const t = i/n, phi = 2*Math.PI*t;
    let v;
    if(m > 0) v = Math.cos(am*phi);
    else if(m < 0) v = Math.sin(am*phi);
    else v = 1;
    out.push({t, v});
  }
  return out;
}

//field value along a meridian (fixed phi), theta 0..pi
function meridianSamples(coeffs, maxL, phi0, n){
  const out = [];
  for(let i=0;i<=n;i++){
    const t = i/n, theta = Math.PI*t;
    let v = 0;
    for(let l=0;l<=maxL;l++) for(let m=-l;m<=l;m++){ const c = coeffs.get(`${l},${m}`)||0; if(c) v += c*ylm(l, m, theta, phi0); }
    out.push({t, v});
  }
  return out;
}

//field value along a parallel (fixed theta), phi 0..2pi
function parallelSamples(coeffs, maxL, theta0, n){
  const out = [];
  for(let i=0;i<=n;i++){
    const t = i/n, phi = 2*Math.PI*t;
    let v = 0;
    for(let l=0;l<=maxL;l++) for(let m=-l;m<=l;m++){ const c = coeffs.get(`${l},${m}`)||0; if(c) v += c*ylm(l, m, theta0, phi); }
    out.push({t, v});
  }
  return out;
}

//normalised positions (0..1) of sign changes in a sample list
function sampleZeros(samples){
  const zeros = [];
  for(let i=1;i<samples.length;i++){
    const a = samples[i-1].v, b = samples[i].v;
    if((a<0&&b>0)||(a>0&&b<0)){
      const f = a/(a-b);
      zeros.push(samples[i-1].t + (samples[i].t - samples[i-1].t)*f);
    }
  }
  return zeros;
}

//colour a signed value with the field's diverging map
function cellColor(v, absmax){
  const c = signColor(v, absmax);
  return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;
}

//draw the round-trip linear system Y c = b as coloured cells: n sample rows, M mode columns
function drawLinearSystem(g, w, h, data){
  g.clearRect(0, 0, w, h);
  const {M, n, Y, b, c} = data;

  let ymax = 1e-9, bmax = 1e-9, cmax = 1e-9;
  for(const row of Y) for(const v of row) ymax = Math.max(ymax, Math.abs(v));
  for(const v of b) bmax = Math.max(bmax, Math.abs(v));
  for(const v of c) cmax = Math.max(cmax, Math.abs(v));

  const padL = 8, padT = 18, padB = 10, gap = 16, vw = 16;
  const rowsMax = Math.max(n, M);
  const ch = Math.max(1, Math.min(13, (h - padT - padB)/rowsMax));
  const cw = Math.max(3, Math.min(13, (w - padL - 2*gap - 2*vw - 8)/M));
  const matW = cw*M, matH = ch*n;

  const xY = padL, yTop = padT;
  for(let i=0;i<n;i++) for(let k=0;k<M;k++){
    g.fillStyle = cellColor(Y[i][k], ymax);
    g.fillRect(xY + k*cw, yTop + i*ch, Math.max(1, cw-0.4), Math.max(1, ch-0.4));
  }

  const xC = xY + matW + gap;
  for(let k=0;k<M;k++){
    g.fillStyle = cellColor(c[k], cmax);
    g.fillRect(xC, yTop + k*ch, vw, Math.max(1, ch-0.4));
  }

  const xB = xC + vw + gap;
  for(let i=0;i<n;i++){
    g.fillStyle = cellColor(b[i], bmax);
    g.fillRect(xB, yTop + i*ch, vw, Math.max(1, ch-0.4));
  }

  g.fillStyle = "#6b6a66"; g.font = "10px sans-serif"; g.textAlign = "center"; g.textBaseline = "alphabetic";
  g.fillText(`Y · n=${n} x M=${M}`, xY + matW/2, yTop - 6);
  g.fillText("c", xC + vw/2, yTop - 6);
  g.fillText("b", xB + vw/2, yTop - 6);
  g.fillStyle = "#9a978f"; g.font = "13px sans-serif"; g.textBaseline = "middle";
  g.fillText("x", xC - gap/2, yTop + matH/2);
  g.fillText("=", xB - gap/2, yTop + matH/2);
}