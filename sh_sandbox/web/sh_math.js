//spherical harmonics maths, written from scratch for the sandbox

//factorial for the small integers the normalisation needs
function factorial(n){
  let f = 1;
  for(let i=2; i<=n; i++) f *= i;
  return f;
}

//associated legendre P_l^m(x) by upward recurrence, m assumed >= 0 here
function assocLegendre(l, m, x){
  let pmm = 1.0;
  if(m > 0){
    const somx2 = Math.sqrt((1.0 - x) * (1.0 + x));
    let fact = 1.0;
    for(let i=1; i<=m; i++){ pmm *= -fact * somx2; fact += 2.0; }
  }
  if(l === m) return pmm;
  let pmmp1 = x * (2*m + 1) * pmm;
  if(l === m + 1) return pmmp1;
  let pll = 0.0;
  for(let ll = m + 2; ll <= l; ll++){
    pll = ((2*ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

//orthonormal real-form normalisation factor for a given l, m
function shNorm(l, m){
  const am = Math.abs(m);
  return Math.sqrt((2*l + 1) / (4*Math.PI) * factorial(l - am) / factorial(l + am));
}

//real spherical harmonic Y_l^m evaluated at one (theta, phi)
function ylm(l, m, theta, phi){
  const am = Math.abs(m);
  const p = assocLegendre(l, am, Math.cos(theta));
  if(m > 0) return Math.SQRT2 * shNorm(l, m) * p * Math.cos(am * phi);
  if(m < 0) return Math.SQRT2 * shNorm(l, m) * p * Math.sin(am * phi);
  return shNorm(l, m) * p;
}

//evaluate one harmonic over a theta/phi grid
function evalGrid(l, m, nTheta, nPhi){
  const values = new Float32Array(nTheta * nPhi);
  let min = Infinity, max = -Infinity;
  for(let i=0; i<nTheta; i++){
    const theta = Math.PI * i / (nTheta - 1);
    for(let j=0; j<nPhi; j++){
      const phi = 2*Math.PI * j / (nPhi - 1);
      const v = ylm(l, m, theta, phi);
      values[i*nPhi + j] = v;
      if(v < min) min = v;
      if(v > max) max = v;
    }
  }
  return {values, nTheta, nPhi, min, max};
}

//precompute every harmonic's grid up to maxL, so the sum is just a fast weighted combine
function precomputeBasis(maxL, nTheta, nPhi){
  const fields = [];
  for(let l=0; l<=maxL; l++){
    for(let m=-l; m<=l; m++){
      fields.push({l, m, values: evalGrid(l, m, nTheta, nPhi).values});
    }
  }
  return fields;
}

//combine precomputed fields by their coefficients (a Map keyed "l,m") into one grid
function combineFields(fields, coeffs, nTheta, nPhi){
  const values = new Float32Array(nTheta * nPhi);
  for(const f of fields){
    const c = coeffs.get(`${f.l},${f.m}`) || 0;
    if(c === 0) continue;
    const fv = f.values;
    for(let p=0; p<values.length; p++) values[p] += c * fv[p];
  }
  let min = Infinity, max = -Infinity;
  for(let p=0; p<values.length; p++){ const v = values[p]; if(v<min) min=v; if(v>max) max=v; }
  return {values, nTheta, nPhi, min, max};
}