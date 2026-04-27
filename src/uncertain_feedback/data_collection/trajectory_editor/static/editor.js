import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── SMPL-22 skeleton topology ──────────────────────────────────────────────
const SMPL_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19];
const BONE_PAIRS = [];
for (let i = 1; i < 22; i++) BONE_PAIRS.push([SMPL_PARENTS[i], i]);
const CHILDREN = Array.from({length: 22}, () => []);
for (let i = 1; i < 22; i++) CHILDREN[SMPL_PARENTS[i]].push(i);

const JOINT_NAMES = [
  'pelvis','l_hip','r_hip','spine1','l_knee','r_knee',
  'spine2','l_ankle','r_ankle','spine3','l_foot','r_foot',
  'neck','l_collar','r_collar','head','l_shoulder','r_shoulder',
  'l_elbow','r_elbow','l_wrist','r_wrist',
];
const EDITABLE = new Set([3,6,9,13,14,16,17,18,19,20,21]);

// ── Colors ─────────────────────────────────────────────────────────────────
const C_FIXED    = 0x666680;
const C_EDIT     = 0x3b82f6;
const C_SELECTED = 0xf59e0b;
const C_BONE_FX  = 0x444460;
const C_BONE_ED  = 0x1d4ed8;
const JOINT_R    = 0.018;

// ── State ──────────────────────────────────────────────────────────────────
let currentPositions = null;  // [[x,y,z] × 22]  — live edited pose
let basePositions    = null;  // [[x,y,z] × 22]  — loaded from demo.pt (kept for reset)
let boneLengths      = null;  // [float × 22]    — T-pose bone lengths
let selectedJoint    = -1;
let isDragging       = false;
let dragPlane        = null;
let keyframes        = [];    // [{frame, positions}]
let totalFrames      = 100;
let currentFrame     = 0;

// Replay state
let replayPositions  = null;  // [[x,y,z]×22][] — loaded from .npy via server
let isPlaying        = false;
let playIntervalId   = null;

// ── Three.js objects ───────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let jointMeshes = [];
let boneLines   = [];
const raycaster  = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.03;

// ── Init ───────────────────────────────────────────────────────────────────
function init() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d1a);

  const vp = document.getElementById('viewport');
  camera = new THREE.PerspectiveCamera(55, vp.clientWidth / vp.clientHeight, 0.001, 50);
  camera.position.set(0, 1.1, 2.2);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(vp.clientWidth, vp.clientHeight);
  vp.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const sun = new THREE.DirectionalLight(0xffffff, 0.9);
  sun.position.set(1, 3, 2);
  scene.add(sun);

  const grid = new THREE.GridHelper(4, 20, 0x222240, 0x181828);
  scene.add(grid);

  // Axis indicator (small)
  const axisHelper = new THREE.AxesHelper(0.2);
  axisHelper.position.set(-1.8, 0.001, -1.8);
  scene.add(axisHelper);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.9, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  buildSkeletonMeshes();

  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  renderer.domElement.addEventListener('pointermove', onPointerMove);
  renderer.domElement.addEventListener('pointerup',   onPointerUp);
  window.addEventListener('resize', onResize);

  // Show placeholder skeleton in T-pose
  showTposePlaceholder();

  animate();
}

function buildSkeletonMeshes() {
  jointMeshes.forEach(m => scene.remove(m));
  boneLines.forEach(l => scene.remove(l));
  jointMeshes = [];
  boneLines   = [];

  for (let i = 0; i < 22; i++) {
    const geo = new THREE.SphereGeometry(JOINT_R, 10, 7);
    const mat = new THREE.MeshStandardMaterial({
      color: EDITABLE.has(i) ? C_EDIT : C_FIXED,
      roughness: 0.5,
      metalness: 0.1,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.userData.jointIdx = i;
    mesh.visible = false;
    scene.add(mesh);
    jointMeshes.push(mesh);
  }

  for (const [p, c] of BONE_PAIRS) {
    const mat = new THREE.LineBasicMaterial({
      color: EDITABLE.has(c) ? C_BONE_ED : C_BONE_FX,
      linewidth: 1,
    });
    const pts = [new THREE.Vector3(), new THREE.Vector3()];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, mat);
    line.userData = { parent: p, child: c };
    line.visible = false;
    scene.add(line);
    boneLines.push(line);
  }
}

function showTposePlaceholder() {
  // Show a faint default T-pose using hardcoded rough positions (metres)
  const tpose = [
    [0,0.9,0],[0.1,0.85,0],[-0.1,0.85,0],[0,1.0,0],
    [0.1,0.5,0],[-0.1,0.5,0],[0,1.15,0],[0.1,0.1,0],
    [-0.1,0.1,0],[0,1.3,0],[0.06,0.02,0],[-0.06,0.02,0],
    [0,1.5,0],[0.12,1.4,0],[-0.12,1.4,0],[0,1.65,0],
    [0.4,1.3,0],[-0.4,1.3,0],[0.7,1.3,0],[-0.7,1.3,0],
    [1.0,1.3,0],[-1.0,1.3,0],
  ];
  updateSkeletonMeshes(tpose, false);
  for (const m of jointMeshes) m.material.opacity = 0.25;
}

function updateSkeletonMeshes(positions, fullOpacity = true) {
  for (let i = 0; i < 22; i++) {
    const [x, y, z] = positions[i];
    jointMeshes[i].position.set(x, y, z);
    jointMeshes[i].visible = true;
    jointMeshes[i].material.transparent = !fullOpacity;
    jointMeshes[i].material.opacity = fullOpacity ? 1.0 : 0.25;
  }
  for (let bi = 0; bi < BONE_PAIRS.length; bi++) {
    const [p, c] = BONE_PAIRS[bi];
    const posAttr = boneLines[bi].geometry.attributes.position;
    posAttr.setXYZ(0, ...positions[p]);
    posAttr.setXYZ(1, ...positions[c]);
    posAttr.needsUpdate = true;
    boneLines[bi].visible = true;
    boneLines[bi].material.transparent = !fullOpacity;
    boneLines[bi].material.opacity = fullOpacity ? 1.0 : 0.25;
  }
}

function setSelectedJoint(idx) {
  if (selectedJoint >= 0) {
    jointMeshes[selectedJoint].material.color.setHex(
      EDITABLE.has(selectedJoint) ? C_EDIT : C_FIXED
    );
    jointMeshes[selectedJoint].material.emissive.setHex(0x000000);
  }
  selectedJoint = idx;
  if (idx >= 0) {
    jointMeshes[idx].material.color.setHex(C_SELECTED);
    jointMeshes[idx].material.emissive.setHex(0x3d2900);
    setStatus(`Selected: ${JOINT_NAMES[idx]} (${EDITABLE.has(idx) ? 'editable' : 'fixed'})`);
    document.getElementById('jointInfo').textContent = JOINT_NAMES[idx];
  } else {
    document.getElementById('jointInfo').textContent = 'Click an editable joint to select';
  }
}

// ── Pointer interaction ────────────────────────────────────────────────────
function getNDC(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  return new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width)  *  2 - 1,
   -((event.clientY - rect.top)  / rect.height) *  2 + 1,
  );
}

function onPointerDown(event) {
  if (!currentPositions || replayPositions) return;  // no drag during replay
  if (event.button !== 0) return;  // left button only

  raycaster.setFromCamera(getNDC(event), camera);
  const editableMeshes = jointMeshes.filter((_, i) => EDITABLE.has(i));
  const hits = raycaster.intersectObjects(editableMeshes);

  if (hits.length > 0) {
    const mesh = hits[0].object;
    const ji = mesh.userData.jointIdx;
    setSelectedJoint(ji);

    // Set drag plane perpendicular to camera at the joint position
    const cameraDir = camera.getWorldDirection(new THREE.Vector3());
    dragPlane = new THREE.Plane().setFromNormalAndCoplanarPoint(
      cameraDir,
      mesh.position.clone()
    );

    controls.enabled = false;
    isDragging = true;
    renderer.domElement.setPointerCapture(event.pointerId);
    event.preventDefault();
  }
}

function onPointerMove(event) {
  if (!isDragging || selectedJoint < 0 || !currentPositions) return;

  raycaster.setFromCamera(getNDC(event), camera);
  const target = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(dragPlane, target)) return;

  // Bone-length constraint: project onto sphere around parent
  const parentIdx = SMPL_PARENTS[selectedJoint];
  let newPos;
  if (parentIdx >= 0 && boneLengths) {
    const parentPos = new THREE.Vector3(...currentPositions[parentIdx]);
    const boneLen = boneLengths[selectedJoint];
    const dir = target.clone().sub(parentPos);
    const len = dir.length();
    if (len > 1e-6) dir.multiplyScalar(boneLen / len);
    newPos = parentPos.clone().add(dir);
  } else {
    newPos = target;
  }

  // Compute delta and apply to joint + subtree
  const oldPos = new THREE.Vector3(...currentPositions[selectedJoint]);
  const delta = newPos.clone().sub(oldPos);

  currentPositions[selectedJoint] = [newPos.x, newPos.y, newPos.z];
  propagateChildren(selectedJoint, delta);
  updateSkeletonMeshes(currentPositions);
}

function propagateChildren(ji, delta) {
  for (const child of CHILDREN[ji]) {
    const p = currentPositions[child];
    currentPositions[child] = [p[0]+delta.x, p[1]+delta.y, p[2]+delta.z];
    propagateChildren(child, delta);
  }
}

function onPointerUp(event) {
  if (isDragging) {
    controls.enabled = true;
    isDragging = false;
    try { renderer.domElement.releasePointerCapture(event.pointerId); } catch(_) {}
  }
}

function onResize() {
  const vp = document.getElementById('viewport');
  camera.aspect = vp.clientWidth / vp.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(vp.clientWidth, vp.clientHeight);
}

// ── Animation loop ─────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ── Keyframe management ────────────────────────────────────────────────────
function deepCopy(arr) {
  return arr.map(p => [...p]);
}

function addKeyframe(frame) {
  if (!currentPositions) return;
  const idx = keyframes.findIndex(k => k.frame === frame);
  if (idx >= 0) keyframes.splice(idx, 1);
  keyframes.push({ frame, positions: deepCopy(currentPositions) });
  keyframes.sort((a, b) => a.frame - b.frame);
  updateTimelineUI();
  setStatus(`Keyframe ${frame} saved (${keyframes.length} total)`);
}

function deleteKeyframe(frame) {
  keyframes = keyframes.filter(k => k.frame !== frame);
  updateTimelineUI();
}

function lerp(a, b, t) { return a + t * (b - a); }

function interpolateAtFrame(frame) {
  if (!currentPositions) return null;
  if (keyframes.length === 0) return deepCopy(basePositions || currentPositions);
  if (keyframes.length === 1) return deepCopy(keyframes[0].positions);

  const before = keyframes.filter(k => k.frame <= frame);
  const after  = keyframes.filter(k => k.frame  > frame);

  if (before.length === 0) return deepCopy(after[0].positions);
  if (after.length  === 0) return deepCopy(before[before.length-1].positions);

  const kfA = before[before.length - 1];
  const kfB = after[0];
  const t = (frame - kfA.frame) / (kfB.frame - kfA.frame);

  return kfA.positions.map((pA, i) => {
    const pB = kfB.positions[i];
    return [lerp(pA[0],pB[0],t), lerp(pA[1],pB[1],t), lerp(pA[2],pB[2],t)];
  });
}

function goToFrame(frame) {
  const maxFrame = (replayPositions ? replayPositions.length : totalFrames) - 1;
  currentFrame = Math.max(0, Math.min(maxFrame, frame));

  if (replayPositions) {
    currentPositions = replayPositions[currentFrame];
  } else {
    const interp = interpolateAtFrame(currentFrame);
    if (interp) {
      currentPositions = interp;
      if (selectedJoint >= 0) {
        jointMeshes[selectedJoint].material.color.setHex(C_SELECTED);
      }
    }
  }
  if (currentPositions) updateSkeletonMeshes(currentPositions);

  document.getElementById('frameSlider').value = currentFrame;
  document.getElementById('frameNum').textContent = currentFrame;
  document.getElementById('frameNumDisp').textContent = currentFrame;
  document.getElementById('playhead').style.left =
    `${(currentFrame / Math.max(maxFrame, 1)) * 100}%`;
}

// ── Timeline UI ────────────────────────────────────────────────────────────
function updateTimelineUI() {
  document.getElementById('kfCount').textContent = keyframes.length;

  // Timeline track markers
  const track = document.getElementById('timelineTrack');
  // Remove old markers
  track.querySelectorAll('.kf-marker').forEach(el => el.remove());
  for (const kf of keyframes) {
    const pct = (kf.frame / Math.max(totalFrames - 1, 1)) * 100;
    const dot = document.createElement('div');
    dot.className = 'kf-marker' + (kf.frame === currentFrame ? ' active' : '');
    dot.style.left = `${pct}%`;
    dot.title = `Frame ${kf.frame}`;
    dot.addEventListener('click', (e) => {
      e.stopPropagation();
      goToFrame(kf.frame);
    });
    track.appendChild(dot);
  }

  // Keyframe list
  const list = document.getElementById('kfList');
  list.innerHTML = '';
  if (keyframes.length === 0) {
    list.innerHTML = '<div style="color:#475569;font-size:11px;padding:4px">No keyframes yet</div>';
    return;
  }
  for (const kf of keyframes) {
    const row = document.createElement('div');
    row.className = 'kf-item' + (kf.frame === currentFrame ? ' active' : '');
    row.innerHTML = `<span>Frame ${kf.frame}</span><span class="kf-del" title="Delete">✕</span>`;
    row.querySelector('span:first-child').addEventListener('click', () => goToFrame(kf.frame));
    row.querySelector('.kf-del').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteKeyframe(kf.frame);
    });
    list.appendChild(row);
  }
}

// ── Replay ─────────────────────────────────────────────────────────────────
async function loadTrajectory() {
  const npyPath = document.getElementById('npyPath').value.trim();
  if (!npyPath) { setStatus('Enter a .npy path'); return; }

  pauseReplay();
  setStatus('Loading trajectory…');

  try {
    const res = await fetch(`/api/load-trajectory?npy_path=${encodeURIComponent(npyPath)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Unknown error');

    replayPositions = data.positions;
    totalFrames = data.num_frames;
    document.getElementById('frameSlider').max = totalFrames - 1;

    document.getElementById('btnPlay').disabled = false;
    const info = document.getElementById('replayInfo');
    info.style.display = 'block';
    info.textContent = `${data.num_frames} frames loaded`;

    goToFrame(0);
    setStatus(`Replay: ${npyPath.split('/').pop()} (${data.num_frames} frames) — press Space or Play`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

function playReplay() {
  if (!replayPositions || isPlaying) return;
  if (currentFrame >= replayPositions.length - 1) goToFrame(0);
  isPlaying = true;
  document.getElementById('btnPlay').innerHTML = '&#9646;&#9646; Pause';
  const fps = Math.max(1, Math.min(60, parseInt(document.getElementById('replayFps').value) || 20));
  playIntervalId = setInterval(() => {
    if (currentFrame >= replayPositions.length - 1) {
      pauseReplay();
      return;
    }
    goToFrame(currentFrame + 1);
  }, 1000 / fps);
}

function pauseReplay() {
  isPlaying = false;
  clearInterval(playIntervalId);
  playIntervalId = null;
  document.getElementById('btnPlay').innerHTML = '&#9654; Play';
}

function stopReplay() {
  pauseReplay();
  replayPositions = null;
  document.getElementById('btnPlay').disabled = true;
  document.getElementById('replayInfo').style.display = 'none';
  document.getElementById('npyPath').value = '';
  // Restore edit mode: go back to keyframe interpolation
  if (basePositions) {
    totalFrames = parseInt(document.getElementById('totalFrames').value) || 100;
    document.getElementById('frameSlider').max = totalFrames - 1;
    goToFrame(Math.min(currentFrame, totalFrames - 1));
    updateTimelineUI();
  }
  setStatus('Replay stopped — edit mode');
}

// ── Load base pose from server ─────────────────────────────────────────────
async function loadBasePose() {
  const ptPath = document.getElementById('ptPath').value.trim();
  const hmlDir = document.getElementById('hmlStatsDir').value.trim();
  if (!ptPath || !hmlDir) {
    setStatus('Please fill in demo.pt path and HML stats dir');
    return;
  }

  setStatus('Loading…');
  try {
    const res = await fetch(
      `/api/base-pose?pt_path=${encodeURIComponent(ptPath)}&hml_stats_dir=${encodeURIComponent(hmlDir)}`
    );
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Unknown error');

    basePositions = data.joints;
    boneLengths   = data.bone_lengths;
    currentPositions = deepCopy(basePositions);
    keyframes = [];

    updateSkeletonMeshes(currentPositions);
    goToFrame(0);
    updateTimelineUI();
    setSelectedJoint(-1);

    document.getElementById('btnExport').disabled = false;
    setStatus(`Loaded ${ptPath.split('/').pop()} — ${data.editable.length} editable joints`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

// ── Export ─────────────────────────────────────────────────────────────────
async function exportDataset() {
  const caption   = document.getElementById('caption').value.trim();
  const outputDir = document.getElementById('outputDir').value.trim();
  const ptPath    = document.getElementById('ptPath').value.trim();
  const hmlDir    = document.getElementById('hmlStatsDir').value.trim();
  const tf        = parseInt(document.getElementById('totalFrames').value, 10);
  const nAugment  = parseInt(document.getElementById('nAugment').value, 10) || 0;
  const noiseStd  = parseFloat(document.getElementById('noiseStd').value) || 0.05;

  if (!caption)   { setStatus('Caption is required'); return; }
  if (!outputDir) { setStatus('Output directory is required'); return; }
  if (keyframes.length < 1) { setStatus('Add at least 1 keyframe'); return; }

  setStatus('Exporting…');
  document.getElementById('btnExport').disabled = true;

  try {
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        keyframes,
        caption,
        output_dir: outputDir,
        hml_stats_dir: hmlDir,
        total_frames: tf,
        pt_path: ptPath,
        n_augment: nAugment,
        noise_std: noiseStd,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Unknown error');

    const box = document.getElementById('exportResult');
    box.style.display = 'block';
    const augNote = data.n_augment > 0 ? ` + ${data.n_augment} augmented` : '';
    box.textContent = `Saved: ${data.motion_ids.join(', ')}  shape=(${data.shape.join(', ')})${augNote}`;
    setStatus(`Exported ${data.motion_ids.length} motion(s) → ${data.output_dir}`);
  } catch (err) {
    setStatus(`Export error: ${err.message}`);
  } finally {
    document.getElementById('btnExport').disabled = false;
  }
}

// ── UI wiring ──────────────────────────────────────────────────────────────
function setStatus(msg) {
  document.getElementById('status').textContent = msg;
}

function wireDOMEvents() {
  document.getElementById('btnLoad').addEventListener('click', loadBasePose);
  document.getElementById('btnLoadTraj').addEventListener('click', loadTrajectory);
  document.getElementById('btnPlay').addEventListener('click', () => {
    if (isPlaying) pauseReplay(); else playReplay();
  });
  document.getElementById('btnStopReplay').addEventListener('click', stopReplay);
  document.getElementById('btnAddKF').addEventListener('click', () => addKeyframe(currentFrame));
  document.getElementById('btnClearKF').addEventListener('click', () => {
    keyframes = [];
    updateTimelineUI();
    if (basePositions) {
      currentPositions = deepCopy(basePositions);
      updateSkeletonMeshes(currentPositions);
    }
    setStatus('Keyframes cleared');
  });
  document.getElementById('btnExport').addEventListener('click', exportDataset);

  const slider = document.getElementById('frameSlider');
  slider.addEventListener('input', () => goToFrame(parseInt(slider.value, 10)));

  document.getElementById('totalFrames').addEventListener('change', (e) => {
    totalFrames = Math.max(40, Math.min(196, parseInt(e.target.value, 10) || 100));
    e.target.value = totalFrames;
    slider.max = totalFrames - 1;
    goToFrame(Math.min(currentFrame, totalFrames - 1));
    updateTimelineUI();
  });

  // Click on timeline track to seek
  document.getElementById('timelineTrack').addEventListener('click', (e) => {
    const track = e.currentTarget;
    const rect = track.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    goToFrame(Math.round(pct * (totalFrames - 1)));
  });

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === ' ') {
      e.preventDefault();
      if (replayPositions) { if (isPlaying) pauseReplay(); else playReplay(); }
    }
    if (e.key === 'k' || e.key === 'K') addKeyframe(currentFrame);
    if (e.key === 'ArrowRight') goToFrame(currentFrame + 1);
    if (e.key === 'ArrowLeft')  goToFrame(currentFrame - 1);
    if (e.key === 'Escape')     setSelectedJoint(-1);
  });
}

// ── Start ──────────────────────────────────────────────────────────────────
init();
wireDOMEvents();

(async () => {
  try {
    const res = await fetch('/api/config');
    const cfg = await res.json();
    if (cfg.pt_path)      document.getElementById('ptPath').value      = cfg.pt_path;
    if (cfg.hml_stats_dir) document.getElementById('hmlStatsDir').value = cfg.hml_stats_dir;
    if (cfg.output_dir)   document.getElementById('outputDir').value   = cfg.output_dir;
  } catch (_) {}
})();
