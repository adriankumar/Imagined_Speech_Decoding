//three.js scene scaffolding and harmonic mesh builders

//build a renderer, camera, lights, and orbit controls inside the given container
function createScene(container){
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0xf7f6f4);
  container.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(2.9, 2.0, 2.9);

  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 0.6);
  key.position.set(3, 5, 2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.25);
  fill.position.set(-3, -2, -3);
  scene.add(fill);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;        //gentle camera spin; set false for a static view
  controls.autoRotateSpeed = 0.8;

  const ctx = {scene, camera, renderer, controls, onFrame:null};

  //match the canvas to the container size
  function resize(){
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || (window.innerHeight - 92);
    camera.aspect = w/h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener("resize", resize);
  ctx.resize = resize;
  resize();

  //single animation loop
  let last = performance.now();
  function loop(now){
    const dt = (now - last)/1000; last = now;
    controls.update();
    if(ctx.onFrame) ctx.onFrame(dt);
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  return ctx;
}

//diverging colour for a signed value: blue positive, pale centre, red negative
function signColor(v, absmax){
  const c = Math.max(-1, Math.min(1, absmax>0 ? v/absmax : 0));
  if(c >= 0) return [0.95-0.78*c, 0.95-0.55*c, 0.95];   //pale -> blue (positive)
  const a = -c;
  return [0.95, 0.95-0.55*a, 0.95-0.78*a];               //pale -> red (negative)
}

//two triangles per grid cell across the (theta, phi) lattice
function gridIndices(nTheta, nPhi){
  const indices = [];
  for(let i=0;i<nTheta-1;i++){
    for(let j=0;j<nPhi-1;j++){
      const a = i*nPhi + j;
      const b = a + nPhi;
      indices.push(a, b, a+1);
      indices.push(b, b+1, a+1);
    }
  }
  return indices;
}

//assemble a vertex-coloured mesh from flat arrays
function meshFrom(positions, colors, indices, roughness){
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({vertexColors:true, roughness:roughness, metalness:0.0, side:THREE.DoubleSide});
  return new THREE.Mesh(geometry, material);
}

//deformed blob: radius = |value| normalised to 1, colour by sign
function buildBlobMesh(grid){
  const {values, nTheta, nPhi, min, max} = grid;
  const absmax = Math.max(Math.abs(min), Math.abs(max)) || 1;
  const positions = [], colors = [];
  for(let i=0;i<nTheta;i++){
    const theta = Math.PI*i/(nTheta-1);
    for(let j=0;j<nPhi;j++){
      const phi = 2*Math.PI*j/(nPhi-1);
      const v = values[i*nPhi + j];
      const r = Math.abs(v)/absmax;
      positions.push(r*Math.sin(theta)*Math.cos(phi), r*Math.cos(theta), r*Math.sin(theta)*Math.sin(phi));
      const col = signColor(v, absmax);
      colors.push(col[0], col[1], col[2]);
    }
  }
  return meshFrom(positions, colors, gridIndices(nTheta, nPhi), 0.55);
}

//fixed unit sphere coloured by the signed value
function buildSphereMesh(grid){
  const {values, nTheta, nPhi, min, max} = grid;
  const absmax = Math.max(Math.abs(min), Math.abs(max)) || 1;
  const positions = [], colors = [];
  for(let i=0;i<nTheta;i++){
    const theta = Math.PI*i/(nTheta-1);
    for(let j=0;j<nPhi;j++){
      const phi = 2*Math.PI*j/(nPhi-1);
      positions.push(Math.sin(theta)*Math.cos(phi), Math.cos(theta), Math.sin(theta)*Math.sin(phi));
      const col = signColor(values[i*nPhi + j], absmax);
      colors.push(col[0], col[1], col[2]);
    }
  }
  return meshFrom(positions, colors, gridIndices(nTheta, nPhi), 0.7);
}

//theta values in (0, pi) where the legendre part of Y_l^m is zero (latitude rings)
function legendreZeros(l, m){
  const am = Math.abs(m);
  const zeros = [];
  const n = 600;
  let prevT = Math.PI/n;
  let prev = assocLegendre(l, am, Math.cos(prevT));
  for(let i=2;i<n;i++){
    const theta = Math.PI*i/n;
    const cur = assocLegendre(l, am, Math.cos(theta));
    if((prev<0&&cur>0)||(prev>0&&cur<0)){
      zeros.push(prevT + (theta-prevT)*(0-prev)/(cur-prev));
    }
    prevT = theta; prev = cur;
  }
  return zeros;
}

//phi values where the azimuthal part of Y_l^m is zero (longitude meridians)
function meridianAngles(m){
  const am = Math.abs(m);
  const phis = [];
  if(am === 0) return phis;
  for(let k=0;k<2*am;k++){
    phis.push(m>0 ? (Math.PI/2 + k*Math.PI)/am : k*Math.PI/am);
  }
  return phis;
}

//nodal lines of Y_l^m as a group of lines on the unit sphere
function buildNodalLines(l, m){
  const group = new THREE.Group();
  const R = 1.004;
  const mat = new THREE.LineBasicMaterial({color:0x2b2926});

  for(const theta of legendreZeros(l, m)){
    const pts = [];
    for(let j=0;j<=120;j++){
      const phi = 2*Math.PI*j/120;
      pts.push(new THREE.Vector3(R*Math.sin(theta)*Math.cos(phi), R*Math.cos(theta), R*Math.sin(theta)*Math.sin(phi)));
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
  }
  for(const phi of meridianAngles(m)){
    const pts = [];
    for(let i=0;i<=120;i++){
      const theta = Math.PI*i/120;
      pts.push(new THREE.Vector3(R*Math.sin(theta)*Math.cos(phi), R*Math.cos(theta), R*Math.sin(theta)*Math.sin(phi)));
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
  }
  return group;
}

//n unit vectors spread evenly over the sphere by the fibonacci spiral
function fibonacciSphere(n){
  if(n <= 1) return [[0, 1, 0]];
  const pts = [];
  const golden = Math.PI*(3 - Math.sqrt(5));
  for(let i=0;i<n;i++){
    const y = 1 - (i/(n-1))*2;
    const r = Math.sqrt(Math.max(0, 1 - y*y));
    const a = golden*i;
    pts.push([Math.cos(a)*r, y, Math.sin(a)*r]);
  }
  return pts;
}

//evaluate the synthesis field (sum of coeffs * harmonics) at one direction
function sampleFieldValue(coeffs, maxL, theta, phi){
  let v = 0;
  for(let l=0;l<=maxL;l++){
    for(let m=-l;m<=l;m++){
      const c = coeffs.get(`${l},${m}`) || 0;
      if(c !== 0) v += c * ylm(l, m, theta, phi);
    }
  }
  return v;
}

//scatter n sample beads on the sphere, each coloured by the field value it reads
function buildSamplePoints(n, coeffs, maxL, absmax){
  const group = new THREE.Group();
  const R = 1.02;
  const positions = [], colors = [];
  for(const [ux, uy, uz] of fibonacciSphere(n)){
    const theta = Math.acos(Math.max(-1, Math.min(1, uy)));
    const phi = Math.atan2(uz, ux);
    const v = sampleFieldValue(coeffs, maxL, theta, phi);
    positions.push(R*ux, R*uy, R*uz);
    const col = signColor(v, absmax);
    colors.push(col[0], col[1], col[2]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  //dark halo behind, coloured dot in front, so each bead has a clear edge
  group.add(new THREE.Points(geo, new THREE.PointsMaterial({color:0x2b2926, size:0.085, sizeAttenuation:true})));
  group.add(new THREE.Points(geo, new THREE.PointsMaterial({size:0.058, sizeAttenuation:true, vertexColors:true})));
  return group;
}

//gaussian elimination with partial pivoting for a small dense system
function solveLinear(A, g, M){
  const a = A.map((row, i)=>{ const r = Array.from(row); r.push(g[i]); return r; });
  for(let col=0;col<M;col++){
    let piv = col;
    for(let r=col+1;r<M;r++) if(Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    [a[col], a[piv]] = [a[piv], a[col]];
    const d = a[col][col] || 1e-12;
    for(let r=0;r<M;r++){
      if(r === col) continue;
      const f = a[r][col]/d;
      for(let k=col;k<=M;k++) a[r][k] -= f*a[col][k];
    }
  }
  const x = new Float64Array(M);
  for(let i=0;i<M;i++) x[i] = a[i][M]/(a[i][i] || 1e-12);
  return x;
}

//recover coeffs from n sampled values by least squares (the inverse of sampling)
function projectToCoeffs(n, trueCoeffs, maxL){
  const modes = [];
  for(let l=0;l<=maxL;l++) for(let m=-l;m<=l;m++) modes.push([l, m]);
  const M = modes.length;

  //accumulate the normal equations A = YtY, g = Ytb, one sample row at a time
  const A = Array.from({length:M}, ()=>new Float64Array(M));
  const g = new Float64Array(M);
  for(const [ux, uy, uz] of fibonacciSphere(n)){
    const theta = Math.acos(Math.max(-1, Math.min(1, uy)));
    const phi = Math.atan2(uz, ux);
    const row = new Float64Array(M);
    for(let k=0;k<M;k++) row[k] = ylm(modes[k][0], modes[k][1], theta, phi);
    let b = 0;
    for(let k=0;k<M;k++) b += (trueCoeffs.get(`${modes[k][0]},${modes[k][1]}`) || 0) * row[k];
    for(let i=0;i<M;i++){
      g[i] += row[i]*b;
      for(let j=0;j<M;j++) A[i][j] += row[i]*row[j];
    }
  }
  //small tikhonov term keeps it solvable when under-resolved (gives the min-norm fit)
  for(let i=0;i<M;i++) A[i][i] += 1e-6;

  const c = solveLinear(A, g, M);
  const map = new Map();
  for(let k=0;k<M;k++) map.set(`${modes[k][0]},${modes[k][1]}`, c[k]);
  return map;
}

//normalised rms difference between two fields, for the reconstruction error readout
function fieldRmsError(orig, recon, absmax){
  const v = orig.values, w = recon.values, N = v.length;
  let se = 0;
  for(let p=0;p<N;p++){ const d = v[p] - w[p]; se += d*d; }
  return Math.sqrt(se/N)/absmax;
}

//unit direction per vertex over the theta/phi lattice, precomputed once for animation
function computeDirs(nTheta, nPhi){
  const dirs = new Float32Array(nTheta*nPhi*3);
  let p = 0;
  for(let i=0;i<nTheta;i++){
    const theta = Math.PI*i/(nTheta-1);
    const st = Math.sin(theta), ct = Math.cos(theta);
    for(let j=0;j<nPhi;j++){
      const phi = 2*Math.PI*j/(nPhi-1);
      dirs[p] = st*Math.cos(phi); dirs[p+1] = ct; dirs[p+2] = st*Math.sin(phi);
      p += 3;
    }
  }
  return dirs;
}

//rewrite an existing mesh in place from a new grid, reusing precomputed directions
function updateMeshFromGrid(mesh, grid, mode, absmax, dirs){
  const v = grid.values;
  const pos = mesh.geometry.attributes.position.array;
  const col = mesh.geometry.attributes.color.array;
  const blob = mode === "blob";
  for(let k=0;k<v.length;k++){
    const val = v[k];
    const b = 3*k;
    const r = blob ? Math.abs(val)/absmax : 1;
    pos[b] = dirs[b]*r; pos[b+1] = dirs[b+1]*r; pos[b+2] = dirs[b+2]*r;
    const c = signColor(val, absmax);
    col[b] = c[0]; col[b+1] = c[1]; col[b+2] = c[2];
  }
  mesh.geometry.attributes.position.needsUpdate = true;
  mesh.geometry.attributes.color.needsUpdate = true;
  if(blob) mesh.geometry.computeVertexNormals();
}


//true zero-set of a field: marching the grid cells and joining sign-change crossings
function buildZeroContour(grid, radius){
  const {values, nTheta, nPhi} = grid;
  const R = radius;
  const dir = (t, p)=>[R*Math.sin(t)*Math.cos(p), R*Math.cos(t), R*Math.sin(t)*Math.sin(p)];
  const pts = [];
  //column nPhi-1 is the same meridian as column 0; read column 0's exact values for it
  //so a node sitting on the phi=0 seam (every negative-m mode) survives sin(2pi) rounding
  const col = j => (j === nPhi-1 ? 0 : j);
  for(let i=0;i<nTheta-1;i++){
    const t0 = Math.PI*i/(nTheta-1), t1 = Math.PI*(i+1)/(nTheta-1);
    for(let j=0;j<nPhi-1;j++){
      const p0 = 2*Math.PI*j/(nPhi-1), p1 = 2*Math.PI*(j+1)/(nPhi-1);
      const ja = col(j), jb = col(j+1);
      const v00 = values[i*nPhi+ja],     v01 = values[i*nPhi+jb];
      const v10 = values[(i+1)*nPhi+ja], v11 = values[(i+1)*nPhi+jb];
      const cross = [];
      if((v00<0)!==(v01<0)){ const f=v00/(v00-v01); cross.push([t0, p0+(p1-p0)*f]); }
      if((v10<0)!==(v11<0)){ const f=v10/(v10-v11); cross.push([t1, p0+(p1-p0)*f]); }
      if((v00<0)!==(v10<0)){ const f=v00/(v00-v10); cross.push([t0+(t1-t0)*f, p0]); }
      if((v01<0)!==(v11<0)){ const f=v01/(v01-v11); cross.push([t0+(t1-t0)*f, p1]); }
      for(let k=0;k+1<cross.length;k+=2){
        const a = dir(cross[k][0], cross[k][1]);
        const b = dir(cross[k+1][0], cross[k+1][1]);
        pts.push(a[0],a[1],a[2], b[0],b[1],b[2]);
      }
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({color:0x2b2926}));
}

//a small camera-facing text sprite for an axis label
function makeAxisLabel(text){
  const c = document.createElement("canvas"); c.width = 64; c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "#6b6a66"; g.font = "bold 42px sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 32, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c), depthTest:false}));
  sp.scale.set(0.22, 0.22, 0.22);
  return sp;
}

//coordinate triad: world-Y is the pole (physics z), world-X is x, world-Z is y
function makeAxes(){
  const group = new THREE.Group();
  const L = 1.45;
  const mat = new THREE.LineBasicMaterial({color:0xc4c0b8});
  const seg = (x,y,z)=>new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(x,y,z)]), mat);
  group.add(seg(L,0,0), seg(0,L,0), seg(0,0,L));
  const lz = makeAxisLabel("z"); lz.position.set(0, L+0.13, 0);
  const lx = makeAxisLabel("x"); lx.position.set(L+0.13, 0, 0);
  const ly = makeAxisLabel("y"); ly.position.set(0, 0, L+0.13);
  group.add(lz, lx, ly);
  return group;
}

//gauss-jordan inverse of a small dense matrix
function invertMatrix(A, M){
  const a = A.map((row, i)=>{ const r = Array.from(row); for(let j=0;j<M;j++) r.push(i===j?1:0); return r; });
  for(let col=0;col<M;col++){
    let piv = col;
    for(let r=col+1;r<M;r++) if(Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    [a[col], a[piv]] = [a[piv], a[col]];
    const d = a[col][col] || 1e-12;
    for(let k=col;k<2*M;k++) a[col][k] /= d;
    for(let r=0;r<M;r++){
      if(r === col) continue;
      const f = a[r][col];
      for(let k=col;k<2*M;k++) a[r][k] -= f*a[col][k];
    }
  }
  const inv = Array.from({length:M}, ()=>new Float64Array(M));
  for(let i=0;i<M;i++) for(let j=0;j<M;j++) inv[i][j] = a[i][M+j];
  return inv;
}

//resolution operator R = (YtY + lambda*I)^-1 (YtY): maps true coeffs to recovered coeffs
function resolutionMatrix(n, maxL){
  const modes = [];
  for(let l=0;l<=maxL;l++) for(let m=-l;m<=l;m++) modes.push([l, m]);
  const M = modes.length;
  const A = Array.from({length:M}, ()=>new Float64Array(M));
  for(const [ux, uy, uz] of fibonacciSphere(n)){
    const theta = Math.acos(Math.max(-1, Math.min(1, uy)));
    const phi = Math.atan2(uz, ux);
    const row = new Float64Array(M);
    for(let k=0;k<M;k++) row[k] = ylm(modes[k][0], modes[k][1], theta, phi);
    for(let i=0;i<M;i++) for(let j=0;j<M;j++) A[i][j] += row[i]*row[j];
  }
  const Areg = A.map((r, i)=>{ const rr = Array.from(r); rr[i] += 1e-6; return rr; });
  const Ainv = invertMatrix(Areg, M);
  const R = Array.from({length:M}, ()=>new Float64Array(M));
  for(let i=0;i<M;i++) for(let j=0;j<M;j++){
    let s = 0; for(let k=0;k<M;k++) s += Ainv[i][k]*A[k][j];
    R[i][j] = s;
  }
  return {R, modes};
}

//apply the resolution operator to a coeff map, returning the recovered coeff map
function applyResolution(R, modes, trueMap){
  const M = modes.length;
  const ct = new Float64Array(M);
  for(let j=0;j<M;j++) ct[j] = trueMap.get(`${modes[j][0]},${modes[j][1]}`) || 0;
  const map = new Map();
  for(let i=0;i<M;i++){
    let s = 0; for(let j=0;j<M;j++) s += R[i][j]*ct[j];
    map.set(`${modes[i][0]},${modes[i][1]}`, s);
  }
  return map;
}

//direction of the field's largest magnitude, for orienting the slice through real structure
function dominantDirection(coeffs, maxL){
  let best = -1, bestTheta = Math.PI/2, bestPhi = 0;
  const nT = 24, nP = 48;
  for(let i=1;i<nT;i++){
    const theta = Math.PI*i/nT;
    for(let j=0;j<nP;j++){
      const phi = 2*Math.PI*j/nP;
      let v = 0;
      for(let l=0;l<=maxL;l++) for(let m=-l;m<=l;m++){ const c = coeffs.get(`${l},${m}`)||0; if(c) v += c*ylm(l, m, theta, phi); }
      const a = Math.abs(v);
      if(a > best){ best = a; bestTheta = theta; bestPhi = phi; }
    }
  }
  return {theta:bestTheta, phi:bestPhi};
}

//translucent disc plus the great-circle line marking the cross-section's meridian plane
function makeSlicePlane(phiStar){
  const group = new THREE.Group();
  const hx = Math.cos(phiStar), hz = Math.sin(phiStar);
  const tint = 0x16a394;   //teal, kept distinct from the blue/red field

  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(1.0, 48),
    new THREE.MeshBasicMaterial({color:tint, transparent:true, opacity:0.16, side:THREE.DoubleSide, depthWrite:false}));
  const m = new THREE.Matrix4();
  m.makeBasis(new THREE.Vector3(hx, 0, hz), new THREE.Vector3(0, 1, 0), new THREE.Vector3(-Math.sin(phiStar), 0, Math.cos(phiStar)));
  disc.quaternion.setFromRotationMatrix(m);
  group.add(disc);

  const pts = [];
  for(let i=0;i<=96;i++){ const a = 2*Math.PI*i/96, c = Math.cos(a), s = Math.sin(a);
    pts.push(new THREE.Vector3(1.01*c*hx, 1.01*s, 1.01*c*hz)); }
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color:tint})));
  return group;
}

//small marker sphere placed at the inspected point
function makeInspectMarker(){
  const mk = new THREE.Mesh(
    new THREE.SphereGeometry(0.035, 16, 12),
    new THREE.MeshBasicMaterial({color:0xf2a93b}));
  mk.visible = false;
  return mk;
}

//the round-trip linear system Y c = b: n sample rows by M mode columns; bvec overrides b
function systemData(n, coeffs, maxL, bvec){
  const modes = [];
  for(let l=0;l<=maxL;l++) for(let m=-l;m<=l;m++) modes.push([l, m]);
  const M = modes.length;
  const dirs = fibonacciSphere(n);
  const Y = [], b = [];
  dirs.forEach(([ux, uy, uz], i)=>{
    const theta = Math.acos(Math.max(-1, Math.min(1, uy)));
    const phi = Math.atan2(uz, ux);
    const row = new Float64Array(M);
    let bi = 0;
    for(let k=0;k<M;k++){
      row[k] = ylm(modes[k][0], modes[k][1], theta, phi);
      bi += (coeffs.get(`${modes[k][0]},${modes[k][1]}`) || 0) * row[k];
    }
    Y.push(row);
    b.push(bvec ? (bvec[i] || 0) : bi);
  });
  const c = modes.map(([l, m])=> coeffs.get(`${l},${m}`) || 0);
  return {modes, M, n:dirs.length, Y, b, c};
}

//recover coeffs from an explicit measurement vector b by least squares: c = (YtY+λI)^-1 Yt b
function solveCoeffsFromB(n, bvec, maxL){
  const modes = [];
  for(let l=0;l<=maxL;l++) for(let m=-l;m<=l;m++) modes.push([l, m]);
  const M = modes.length;
  const A = Array.from({length:M}, ()=>new Float64Array(M));
  const g = new Float64Array(M);
  fibonacciSphere(n).forEach(([ux, uy, uz], i)=>{
    const theta = Math.acos(Math.max(-1, Math.min(1, uy)));
    const phi = Math.atan2(uz, ux);
    const row = new Float64Array(M);
    for(let k=0;k<M;k++) row[k] = ylm(modes[k][0], modes[k][1], theta, phi);
    const bi = bvec[i] || 0;
    for(let k=0;k<M;k++){ g[k] += row[k]*bi; for(let j=0;j<M;j++) A[k][j] += row[k]*row[j]; }
  });
  for(let i=0;i<M;i++) A[i][i] += 1e-6;
  const c = solveLinear(A, g, M);
  const map = new Map();
  for(let k=0;k<M;k++) map.set(`${modes[k][0]},${modes[k][1]}`, c[k]);
  return map;
}

//sample the current field at the n sample directions, giving a starting b vector
function sampleBVector(n, coeffs, maxL){
  return fibonacciSphere(n).map(([ux, uy, uz])=>{
    const theta = Math.acos(Math.max(-1, Math.min(1, uy)));
    const phi = Math.atan2(uz, ux);
    return sampleFieldValue(coeffs, maxL, theta, phi);
  });
}

//scatter beads coloured by an explicit b vector rather than by the field
function buildSamplePointsFromB(n, bvec, absmax){
  const group = new THREE.Group();
  const R = 1.02;
  const positions = [], colors = [];
  fibonacciSphere(n).forEach(([ux, uy, uz], i)=>{
    positions.push(R*ux, R*uy, R*uz);
    const col = signColor(bvec[i] || 0, absmax);
    colors.push(col[0], col[1], col[2]);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  group.add(new THREE.Points(geo, new THREE.PointsMaterial({color:0x2b2926, size:0.085, sizeAttenuation:true})));
  group.add(new THREE.Points(geo, new THREE.PointsMaterial({size:0.058, sizeAttenuation:true, vertexColors:true})));
  return group;
}

//normalised least-squares residual ||b - Yc|| / ||b||: how much of b the basis cannot reach
function bResidual(n, bvec, coeffs, maxL){
  let se = 0, sb = 0;
  fibonacciSphere(n).forEach(([ux, uy, uz], i)=>{
    const theta = Math.acos(Math.max(-1, Math.min(1, uy)));
    const phi = Math.atan2(uz, ux);
    const yc = sampleFieldValue(coeffs, maxL, theta, phi);
    const bi = bvec[i] || 0;
    const d = bi - yc; se += d*d; sb += bi*bi;
  });
  return sb > 0 ? Math.sqrt(se/sb) : 0;
}