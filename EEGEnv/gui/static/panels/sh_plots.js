//canvas drawing helpers: crisp setup, a dark-centred luminous diverging colour, and the harmonic system Y^T c = b
//on the dark theme zero blends into the panel and strong values glow outward, positive warm, negative cool

//palette matched to app.css so canvas text and cells read against the dark panel
const SH_PANEL = [0.137, 0.137, 0.153];   //panel rgb, the zero/centre of the diverging map (#232327)
const SH_LABEL = "#c77dff";               //luminous magenta block labels, drawn with a glow
const SH_LIGHT = "#d8d4e0";               //light text for glyphs and residual
const SH_DIM   = "#9a96a2";               //dim text for tick numbers

//prepare a canvas for crisp drawing at css size w x h, returns its 2d context
function setupPlot(canvas, w, h){
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return g;
}

//dark-centred diverging map: at zero the cell equals the panel colour, growing to a luminous orange for
//positive and a luminous cyan for negative, so the matrix glows away from zero against the dark background
function signColor(v, absmax){
  const c = Math.max(-1, Math.min(1, absmax > 0 ? v / absmax : 0));
  const a = Math.abs(c);
  const [r0, g0, b0] = SH_PANEL;
  if(c >= 0){
    //panel -> warm orange (~1.0, 0.62, 0.30)
    return [r0 + (1.00 - r0) * a, g0 + (0.62 - g0) * a, b0 + (0.30 - b0) * a];
  }
  //panel -> cool cyan (~0.30, 0.72, 0.98)
  return [r0 + (0.30 - r0) * a, g0 + (0.72 - g0) * a, b0 + (0.98 - b0) * a];
}

//colour a signed value with the field's diverging map
function cellColor(v, absmax){
  const c = signColor(v, absmax);
  return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;
}

//draw glowing text, used for the block labels so they read as illuminated against the dark background
function _glowText(g, text, x, y, color, blur){
  g.save();
  g.shadowColor = color; g.shadowBlur = blur;
  g.fillStyle = color;
  g.fillText(text, x, y);
  g.restore();
}

//draw the harmonic system Y^T c = b with c and b as matrices over all features, three heatmaps side by side
//Y^T is channels by modes, c is modes by features, b is channels by features, each scaled by its own absmax
//the per-feature compression residual sits under the b block, the current-window fidelity ceiling on decode reach
function drawHarmonicSystem(g, w, h, data){
  g.clearRect(0, 0, w, h);
  const {YT, c, b, residual, names} = data;
  const n = YT.length, M = YT[0].length, F = c[0].length;

  let ymax = 1e-9, cmax = 1e-9, bmax = 1e-9;
  for(const row of YT) for(const v of row) ymax = Math.max(ymax, Math.abs(v));
  for(const row of c)  for(const v of row) cmax = Math.max(cmax, Math.abs(v));
  for(const row of b)  for(const v of row) bmax = Math.max(bmax, Math.abs(v));

  const padL = 10, padT = 24, padB = 34, gap = 30;
  const ch  = Math.max(1, Math.min(7, (h - padT - padB) / n));        //cell height from the tallest block, channels
  const cwY = Math.max(2, Math.min(14, (w * 0.40) / M));              //Y^T columns, the modes
  const cwF = Math.max(12, Math.min(40, (w * 0.20) / Math.max(F, 1)));//c and b feature columns

  const yX = padL, yTop = padT;
  const matWY = cwY * M, matWF = cwF * F;

  //Y^T, n rows by M cols
  for(let i = 0; i < n; i++) for(let k = 0; k < M; k++){
    g.fillStyle = cellColor(YT[i][k], ymax);
    g.fillRect(yX + k * cwY, yTop + i * ch, Math.max(1, cwY - 0.4), Math.max(1, ch - 0.4));
  }
  //c, M rows by F cols
  const cX = yX + matWY + gap;
  for(let k = 0; k < M; k++) for(let j = 0; j < F; j++){
    g.fillStyle = cellColor(c[k][j], cmax);
    g.fillRect(cX + j * cwF, yTop + k * ch, Math.max(1, cwF - 0.5), Math.max(1, ch - 0.4));
  }
  //b, n rows by F cols
  const bX = cX + matWF + gap;
  for(let i = 0; i < n; i++) for(let j = 0; j < F; j++){
    g.fillStyle = cellColor(b[i][j], bmax);
    g.fillRect(bX + j * cwF, yTop + i * ch, Math.max(1, cwF - 0.5), Math.max(1, ch - 0.4));
  }

  //block labels, glowing magenta
  g.font = "10px sans-serif"; g.textAlign = "center"; g.textBaseline = "alphabetic";
  _glowText(g, `Y^T  n=${n} x M=${M}`, yX + matWY / 2, yTop - 8, SH_LABEL, 6);
  _glowText(g, `c  M=${M} x F=${F}`, cX + matWF / 2, yTop - 8, SH_LABEL, 6);
  _glowText(g, `b  n=${n} x F=${F}`, bX + matWF / 2, yTop - 8, SH_LABEL, 6);

  //operator glyphs between the blocks
  g.fillStyle = SH_LIGHT; g.font = "14px sans-serif"; g.textBaseline = "middle";
  g.fillText("\u00d7", cX - gap / 2, yTop + (M * ch) / 2);
  g.fillText("=", bX - gap / 2, yTop + (n * ch) / 2);

  //per-feature compression residual under the b block
  if(residual){
    g.font = "9px sans-serif"; g.textAlign = "center"; g.textBaseline = "top";
    const ry = yTop + n * ch + 6;
    for(let j = 0; j < F; j++){
      g.fillStyle = SH_LIGHT; g.fillText(names[j], bX + j * cwF + cwF / 2, ry);
      g.fillStyle = SH_DIM;   g.fillText("r=" + residual[j].toFixed(2), bX + j * cwF + cwF / 2, ry + 11);
    }
  }
}

//draw a modes-by-features coefficient matrix as a heatmap with feature labels, self-normalised
function drawCoeffMatrix(g, w, h, data){
  g.clearRect(0, 0, w, h);
  const {coeffs, names, title} = data;
  const modes = coeffs.length, F = names.length;
  if(!modes || !F) return;
  let amax = 1e-9;
  for(const row of coeffs) for(const v of row) amax = Math.max(amax, Math.abs(v));

  const padL = 34, padT = 30, padR = 10, padB = 26;
  const cw = Math.max(6, Math.min(48, (w - padL - padR) / F));
  const ch = Math.max(1, Math.min(12, (h - padT - padB) / modes));
  const x0 = padL, y0 = padT;

  for(let i = 0; i < modes; i++) for(let j = 0; j < F; j++){
    g.fillStyle = cellColor(coeffs[i][j], amax);
    g.fillRect(x0 + j * cw, y0 + i * ch, Math.max(1, cw - 0.5), Math.max(1, ch - 0.5));
  }
  g.fillStyle = SH_LIGHT; g.font = "9px sans-serif"; g.textAlign = "center"; g.textBaseline = "top";
  for(let j = 0; j < F; j++) g.fillText(names[j], x0 + j * cw + cw / 2, y0 + modes * ch + 4);
  g.fillStyle = SH_DIM; g.textAlign = "right"; g.textBaseline = "middle";
  const step = Math.max(1, Math.round(modes / 10));
  for(let i = 0; i < modes; i += step) g.fillText(String(i), x0 - 4, y0 + i * ch + ch / 2);
  g.font = "10px sans-serif"; g.textAlign = "left"; g.textBaseline = "alphabetic";
  if(title) _glowText(g, title, x0, 16, SH_LABEL, 6);
}