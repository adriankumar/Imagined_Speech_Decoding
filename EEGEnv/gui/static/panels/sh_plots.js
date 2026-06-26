//canvas drawing helpers copied unmodified from the sh sandbox: setup, sign colour, cell colour, linear system

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

//diverging map: pale -> blue for positive, pale -> red for negative, scaled by absmax
function signColor(v, absmax){
  const c = Math.max(-1, Math.min(1, absmax>0 ? v/absmax : 0));
  if(c >= 0) return [0.95-0.78*c, 0.95-0.55*c, 0.95];   //pale -> blue (positive)
  const a = -c;
  return [0.95, 0.95-0.55*a, 0.95-0.78*a];               //pale -> red (negative)
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
  g.fillText(`Y . n=${n} x M=${M}`, xY + matW/2, yTop - 6);
  g.fillText("c", xC + vw/2, yTop - 6);
  g.fillText("b", xB + vw/2, yTop - 6);
  g.fillStyle = "#9a978f"; g.font = "13px sans-serif"; g.textBaseline = "middle";
  g.fillText("x", xC - gap/2, yTop + matH/2);
  g.fillText("=", xB - gap/2, yTop + matH/2);
}