//minimal basis viewer: one mode at a time on the fitted sphere, sampled at the electrodes
//all field values arrive pre-evaluated from python, this file only draws them

//diverging map, matches the colourbar in style.css; tweak here to track DELTA_CMAP
const NEG = [0.816, 0.290, 0.231];   //#d04a3b, red
const MID = [0.949, 0.941, 0.925];   //#f2f0ec
const POS = [0.231, 0.435, 0.816];   //#3b6fd0, blue

const BLOB_FLOOR = 0.15;   //blob radius at a node, keeps the surface from collapsing
const PT_LIFT = 1.03;      //electrodes sit just off the surface so they stay visible
const NODE_LIFT = 1.004;   //nodal lines ride a fixed reference shell in both render modes

//signed value in [-1, 1] to rgb
function signColor(v){
  const t = Math.max(-1, Math.min(1, v));
  const end = t >= 0 ? POS : NEG;
  const a = Math.abs(t);
  return [MID[0] + (end[0] - MID[0]) * a,
          MID[1] + (end[1] - MID[1]) * a,
          MID[2] + (end[2] - MID[2]) * a];
}

function cssColor(v){
  const c = signColor(v);
  return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;
}

let DATA = null;
let modeIdx = 0;
let blobMode = false;

let scene, camera, renderer, controls;
let surface, surfGeo, points, pointHalo, pointGeo, nodeLines;

//---------- geometry ----------
//grid vertex order is row-major over (n_theta, n_phi), matching the values array
function buildSurface(){
  const {n_theta, n_phi} = DATA;
  const nVerts = n_theta * n_phi;

  const pos = new Float32Array(nVerts * 3);
  const col = new Float32Array(nVerts * 3);
  const dir = new Float32Array(nVerts * 3); //unit direction, radius applied per mode

  for(let i = 0; i < n_theta; i++){
    const th = Math.PI * i / (n_theta - 1);
    for(let j = 0; j < n_phi; j++){
      const ph = 2 * Math.PI * j / (n_phi - 1);
      const k = (i * n_phi + j) * 3;
      dir[k]   = Math.sin(th) * Math.cos(ph);
      dir[k+1] = Math.cos(th);
      dir[k+2] = Math.sin(th) * Math.sin(ph);
    }
  }

  const idx = [];
  for(let i = 0; i < n_theta - 1; i++){
    for(let j = 0; j < n_phi - 1; j++){
      const a = i * n_phi + j, b = a + 1, c = a + n_phi, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  }

  surfGeo = new THREE.BufferGeometry();
  surfGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  surfGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  surfGeo.setIndex(idx);
  surfGeo.userData.dir = dir;

  //polygon offset pushes surface fragments back in the depth buffer, so the
  //nodal lines and electrode markers never drop through it as the camera moves
  const mat = new THREE.MeshLambertMaterial({vertexColors: true,
                                             side: THREE.DoubleSide,
                                             polygonOffset: true,
                                             polygonOffsetFactor: 1.0,
                                             polygonOffsetUnits: 1.0});
  surface = new THREE.Mesh(surfGeo, mat);
  scene.add(surface);
}

function buildPoints(){
  const n = DATA.electrodes.length;
  pointGeo = new THREE.BufferGeometry();
  pointGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  pointGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));

  //larger dark square drawn first, coloured square on top of it, giving a border
  pointHalo = new THREE.Points(pointGeo, new THREE.PointsMaterial({size: 0.075,
                                                                   color: 0x2b2926,
                                                                   sizeAttenuation: true}));
  pointHalo.renderOrder = 1;
  scene.add(pointHalo);

  //equal-depth test lets the fill land on the halo instead of being rejected by it
  points = new THREE.Points(pointGeo, new THREE.PointsMaterial({size: 0.05,
                                                                vertexColors: true,
                                                                sizeAttenuation: true,
                                                                depthFunc: THREE.LessEqualDepth,
                                                                depthWrite: false}));
  points.renderOrder = 2;
  scene.add(points);
}

function buildNodes(){
  nodeLines = new THREE.LineSegments(new THREE.BufferGeometry(),
                                     new THREE.LineBasicMaterial({color: 0x2b2926, depthTest: false}));
  nodeLines.renderOrder = 3;
  nodeLines.scale.set(NODE_LIFT, NODE_LIFT, NODE_LIFT); //fixed shell, set once
  scene.add(nodeLines);
}

//---------- per-mode update ----------
function updateMode(){
  const mode = DATA.modes[modeIdx];
  const {n_theta, n_phi} = DATA;

  //surface: radius from |v| in blob mode, unit sphere otherwise
  const pos = surfGeo.attributes.position.array;
  const col = surfGeo.attributes.color.array;
  const dir = surfGeo.userData.dir;

  for(let p = 0; p < n_theta * n_phi; p++){
    const v = mode.values[p];
    const r = blobMode ? BLOB_FLOOR + (1 - BLOB_FLOOR) * Math.abs(v) : 1.0;
    const k = p * 3;
    pos[k] = dir[k] * r; pos[k+1] = dir[k+1] * r; pos[k+2] = dir[k+2] * r;
    const c = signColor(v);
    col[k] = c[0]; col[k+1] = c[1]; col[k+2] = c[2];
  }
  surfGeo.attributes.position.needsUpdate = true;
  surfGeo.attributes.color.needsUpdate = true;
  surfGeo.computeVertexNormals();
  surfGeo.computeBoundingSphere();

  //electrodes: the row of Y itself, normalised by the same absmax as the field
  const row = DATA.Y[modeIdx];
  const scale = mode.absmax > 0 ? mode.absmax : 1;
  const ppos = pointGeo.attributes.position.array;
  const pcol = pointGeo.attributes.color.array;

  DATA.electrodes.forEach((e, i) => {
    const v = row[i] / scale;
    const r = (blobMode ? BLOB_FLOOR + (1 - BLOB_FLOOR) * Math.abs(v) : 1.0) * PT_LIFT;
    const k = i * 3;
    ppos[k] = e[0] * r; ppos[k+1] = e[1] * r; ppos[k+2] = e[2] * r;
    const c = signColor(v);
    pcol[k] = c[0]; pcol[k+1] = c[1]; pcol[k+2] = c[2];
  });
  pointGeo.attributes.position.needsUpdate = true;
  pointGeo.attributes.color.needsUpdate = true;
  pointGeo.computeBoundingSphere();

  //nodal lines mark the directions where the field is zero, so they stay on the
  //reference shell rather than following the blob into its pinch points
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(mode.nodes), 3));
  nodeLines.geometry.dispose();
  nodeLines.geometry = g;

  drawMatrix();

  const sign = mode.m > 0 ? "+" : "";
  document.getElementById("readout").innerHTML =
    `<span class="mode-tag">Y ℓ=${mode.l}, m=${sign}${mode.m}</span>` +
    `row ${modeIdx + 1} of ${DATA.modes.length} · peak |Y| on the sphere ${mode.absmax.toFixed(4)} · ` +
    `${mode.nodes.length / 6} nodal segments`;
  document.getElementById("cbar-mag").textContent = "±" + mode.absmax.toFixed(3);
}

//---------- matrix panel ----------
function drawMatrix(){
  const canvas = document.getElementById("matrix");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const rows = DATA.Y.length, cols = DATA.n_chns;
  const pad = 4;
  const ch = (h - 2 * pad) / rows;
  const cw = (w - 2 * pad) / cols;

  for(let i = 0; i < rows; i++){
    const scale = DATA.modes[i].absmax > 0 ? DATA.modes[i].absmax : 1; //per-row, so it reads like the sphere
    for(let j = 0; j < cols; j++){
      g.fillStyle = cssColor(DATA.Y[i][j] / scale);
      g.fillRect(pad + j * cw, pad + i * ch, Math.max(1, cw), Math.max(1, ch));
    }
  }

  g.strokeStyle = "#2b2926"; g.lineWidth = 1.5;
  g.strokeRect(pad - 0.5, pad + modeIdx * ch - 0.5, w - 2 * pad + 1, Math.max(2, ch) + 1);

  canvas.onclick = ev => {
    const y = ev.clientY - canvas.getBoundingClientRect().top - pad;
    const i = Math.floor(y / ch);
    if(i >= 0 && i < rows){ modeIdx = i; updateMode(); }
  };
}

//---------- scene ----------
function initScene(){
  const wrap = document.getElementById("canvas-wrap");

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  //a tight near plane costs depth precision, which is what makes thin overlays flicker
  camera = new THREE.PerspectiveCamera(42, wrap.clientWidth / wrap.clientHeight, 0.5, 40);
  camera.position.set(2.4, 1.5, 2.4);

  renderer = new THREE.WebGLRenderer({antialias: true, logarithmicDepthBuffer: false});
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  wrap.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.2;
  controls.maxDistance = 12;

  //heavy ambient so lighting gives shape without distorting the field colours
  scene.add(new THREE.AmbientLight(0xffffff, 0.78));
  const key = new THREE.DirectionalLight(0xffffff, 0.35);
  key.position.set(2, 3, 2);
  scene.add(key);

  window.addEventListener("resize", () => {
    camera.aspect = wrap.clientWidth / wrap.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(wrap.clientWidth, wrap.clientHeight);
    drawMatrix();
  });
}

function animate(){
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

//---------- controls ----------
function setRender(blob){
  blobMode = blob;
  document.getElementById("btn-blob").classList.toggle("on", blob);
  document.getElementById("btn-sphere").classList.toggle("on", !blob);
  updateMode();
}

function step(d){
  modeIdx = (modeIdx + d + DATA.modes.length) % DATA.modes.length;
  updateMode();
}

function wireControls(){
  document.getElementById("btn-blob").onclick = () => setRender(true);
  document.getElementById("btn-sphere").onclick = () => setRender(false);
  document.getElementById("btn-prev").onclick = () => step(-1);
  document.getElementById("btn-next").onclick = () => step(1);

  const nodeBtn = document.getElementById("btn-nodes");
  nodeBtn.onclick = () => {
    nodeLines.visible = !nodeLines.visible;
    nodeBtn.classList.toggle("on", nodeLines.visible);
  };

  const ptBtn = document.getElementById("btn-pts");
  ptBtn.onclick = () => {
    const on = !points.visible;
    points.visible = on; pointHalo.visible = on; //halo and fill toggle together
    ptBtn.classList.toggle("on", on);
  };

  window.addEventListener("keydown", ev => {
    if(ev.key === "ArrowRight" || ev.key === "ArrowDown") step(1);
    if(ev.key === "ArrowLeft" || ev.key === "ArrowUp") step(-1);
  });
}

//---------- boot ----------
fetch("payload.json").then(r => r.json()).then(d => {
  DATA = d;
  document.getElementById("meta").textContent =
    `L = ${d.L} · ${d.modes.length} modes × ${d.n_chns} electrodes · grid ${d.n_theta}×${d.n_phi}`;
  document.getElementById("dims").textContent = `${d.modes.length} × ${d.n_chns}`;

  initScene();
  buildSurface();
  buildPoints();
  buildNodes();
  wireControls();
  updateMode();
  animate();
});