"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let INIT = null;
let personas = [];
let currentPersona = null;

// key -> {data: packaged trajectory, color, label, visible}
const trajs = {};
let clusters = [];          // [{label, count, oracle_score, trajectory}]
let selectedCluster = null;
let selectedClusterSegments = null;
let showClusters = true;
let showStart = true;
let showMdmStart = true;
let showOracleBounds = true;
let showGeneratedBounds = true;
let generatedBounds = [];
let startPreview = null;    // (5,3) arm positions for the edited start pose
let baseTrigger = null;

const MDM_START_COLOR = "#7b2fbe";
const GENERATED_LIMIT_COLOR = "#3567d6";

let frame = 0;
let playing = false;
let lastTick = 0;

const TRAJ_STYLES = {
  base: { color: "#e05252", label: "base rollout" },
  full: { color: "#0e7a63", label: "full corrected path" },
  generated: { color: "#3567d6", label: "generated-cost rollout" },
};
const CLUSTER_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#9a6324", "#800000", "#469990"];
const START_COLOR = "#e8a33d";
const VIEWS = [
  { name: "Front (XY)", hi: 0, vi: 1 },
  { name: "Side (ZY)", hi: 2, vi: 1 },
  { name: "Top (XZ)", hi: 0, vi: 2 },
];
const WRIST = 4; // index into the 5-joint arm chain

// ---------------------------------------------------------------------------
// API + status
// ---------------------------------------------------------------------------

const statusEl = document.getElementById("status");
let _busyCount = 0;

function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusEl.className = kind || "";
}

// A silent rendering crash looks like "nothing happened" — surface it instead.
// The usual cause is a client/server version mismatch: restart the server and
// hard-refresh (Ctrl+Shift+R).
window.onerror = (msg) => {
  setStatus(`JS error: ${msg} — restart server + hard-refresh?`, "error");
};
window.onunhandledrejection = (e) => {
  if (e.reason && e.reason.reported) return; // api() already showed it
  setStatus(`JS error: ${e.reason} — restart server + hard-refresh?`, "error");
};

async function api(path, body, busyMsg) {
  if (busyMsg) {
    _busyCount++;
    setStatus(busyMsg + " …", "busy");
  }
  const opts = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  try {
    const resp = await fetch(path, opts);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    if (busyMsg && --_busyCount === 0) setStatus("ready");
    return data;
  } catch (err) {
    if (busyMsg) _busyCount = Math.max(0, _busyCount - 1);
    setStatus(String(err.message || err), "error");
    err.reported = true;
    throw err;
  }
}

async function apiDelete(path, busyMsg) {
  if (busyMsg) setStatus(busyMsg + " …", "busy");
  const resp = await fetch(path, { method: "DELETE" });
  const data = await resp.json();
  if (!resp.ok) {
    setStatus(data.error || resp.statusText, "error");
    const err = new Error(data.error);
    err.reported = true;
    throw err;
  }
  setStatus("ready");
  return data;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

function getGoal() {
  return [+$("goal-x").value, +$("goal-y").value, +$("goal-z").value];
}

function getScale() { return +$("scale-num").value; }

function getPersona() {
  return personas.find((p) => p.name === currentPersona);
}

function fmtMetrics(m, reach) {
  let s = `mean violation ${m.mean_violation.toFixed(3)} rad · max ${m.max_violation.toFixed(3)} · ` +
          `${(m.frac_frames_violated * 100).toFixed(0)}% frames violated`;
  if (reach) {
    s += `\ngoal ${reach.reached ? "reached" : "NOT reached"} (dist ${reach.distance.toFixed(3)} m, thr ${reach.threshold})`;
  }
  return s;
}

function clientMetrics(t) {
  const v = t.violations;
  return {
    mean_violation: v.reduce((a, b) => a + b, 0) / v.length,
    max_violation: Math.max(...v),
    frac_frames_violated: v.filter((x) => x > 0).length / v.length,
  };
}

function visibleTrajs() {
  return Object.entries(trajs).filter(([, t]) => t.visible && t.data);
}

function maxFrames() {
  let m = 1;
  for (const [, t] of visibleTrajs()) m = Math.max(m, t.data.n_frames);
  if (showClusters) for (const c of clusters) m = Math.max(m, c.full.n_frames);
  return m;
}

function setTraj(key, data) {
  trajs[key] = { data, visible: true, ...TRAJ_STYLES[key] };
  refreshLegend();
  refreshTimeline();
  renderAll();
}

function clearTraj(...keys) {
  for (const k of keys) delete trajs[k];
}

// ---------------------------------------------------------------------------
// Arm / start-pose editor
// ---------------------------------------------------------------------------

const JOINT_NAMES = ["left_shoulder (clavicle)", "left_elbow (upper arm)", "left_wrist (forearm)"];
const AXES = ["x", "y", "z"];
let armAA = null; // (3,3)

function buildArmEditor() {
  const grid = $("arm-editor");
  grid.innerHTML = "";
  JOINT_NAMES.forEach((jn, j) => {
    const name = document.createElement("div");
    name.className = "jname";
    name.textContent = jn;
    grid.appendChild(name);
    AXES.forEach((ax, a) => {
      const lab = document.createElement("span");
      lab.textContent = ax;
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = -3.14; slider.max = 3.14; slider.step = 0.01;
      const num = document.createElement("input");
      num.type = "number";
      num.step = 0.01;
      slider.id = `arm-s-${j}-${a}`;
      num.id = `arm-n-${j}-${a}`;
      slider.oninput = () => { num.value = slider.value; armAA[j][a] = +slider.value; schedulePreview(); };
      num.onchange = () => { slider.value = num.value; armAA[j][a] = +num.value; schedulePreview(); };
      grid.appendChild(lab);
      grid.appendChild(slider);
      grid.appendChild(num);
    });
  });
}

function setArmEditorValues(aa) {
  armAA = aa.map((r) => r.slice());
  for (let j = 0; j < 3; j++) for (let a = 0; a < 3; a++) {
    $(`arm-s-${j}-${a}`).value = aa[j][a].toFixed(3);
    $(`arm-n-${j}-${a}`).value = aa[j][a].toFixed(3);
  }
  schedulePreview();
}

let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    const data = await api("/api/preview_pose", { arm_aa: armAA });
    startPreview = data.arm_positions;
    $("wrist-rel").textContent = data.wrist_rel.map((v) => v.toFixed(2)).join(", ");
    refreshLegend();
    renderAll();
  }, 150);
}

// ---------------------------------------------------------------------------
// Personas
// ---------------------------------------------------------------------------

function refreshPersonaSelect() {
  const sel = $("persona-select");
  sel.innerHTML = "";
  for (const p of personas) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.name + (p.builtin ? "" : " *");
    sel.appendChild(opt);
  }
  sel.value = currentPersona;
}

function onPersonaChange(name) {
  currentPersona = name;
  const p = getPersona();
  $("persona-delete").disabled = p.builtin;
  if (p.feedback_text) $("prompt").value = p.feedback_text;
  const goals = INIT.persona_goals[name];
  const goal = goals && goals.cartesian.length ? goals.cartesian[0] : INIT.default_goal;
  [$("goal-x").value, $("goal-y").value, $("goal-z").value] = goal.map((v) => v.toFixed(2));
  buildCoupledGraphs();
  renderAll();
}

// --- persona modal ---

let editingBuiltin = false;

function boundRow(b) {
  const row = document.createElement("div");
  row.className = "bound-row";
  const feats = INIT.feature_names;
  const mkSel = (opts, val) => {
    const s = document.createElement("select");
    for (const o of opts) {
      const opt = document.createElement("option");
      opt.value = o; opt.textContent = o;
      s.appendChild(opt);
    }
    s.value = val;
    return s;
  };
  const mkNum = (val, ph) => {
    const n = document.createElement("input");
    n.type = "number"; n.step = 0.05; n.placeholder = ph;
    if (val !== null && val !== undefined) n.value = val;
    return n;
  };
  const kind = mkSel(["hidden", "coupled"], b.kind || "hidden");
  const feature = mkSel(feats, b.feature || feats[0]);
  const btHidden = mkSel(["upper_bound", "lower_bound", "avoid_band"], b.bound_type || "upper_bound");
  const btCoupled = mkSel(["upper_bound", "lower_bound"], b.bound_type || "upper_bound");
  const low = mkNum(b.low, "low");
  const high = mkNum(b.high, "high");
  const cond = mkSel(feats, b.cond_feature || feats[0]);
  const intercept = mkNum(b.intercept, "intercept");
  const slope = mkNum(b.slope, "slope");
  const del = document.createElement("button");
  del.textContent = "✕";
  del.onclick = () => row.remove();

  function layout() {
    row.innerHTML = "";
    row.appendChild(kind);
    row.appendChild(feature);
    if (kind.value === "hidden") {
      row.appendChild(btHidden);
      row.appendChild(low);
      row.appendChild(high);
    } else {
      row.appendChild(btCoupled);
      const l1 = document.createElement("span"); l1.textContent = "vs";
      row.appendChild(l1);
      row.appendChild(cond);
      row.appendChild(intercept);
      row.appendChild(slope);
      const l2 = document.createElement("span"); l2.textContent = "thr = intercept + slope·cond";
      row.appendChild(l2);
    }
    row.appendChild(del);
  }
  kind.onchange = layout;
  layout();

  row.toJSON = () => {
    if (kind.value === "hidden") {
      return {
        kind: "hidden", feature: feature.value, bound_type: btHidden.value,
        low: low.value === "" ? null : +low.value,
        high: high.value === "" ? null : +high.value,
      };
    }
    return {
      kind: "coupled", feature: feature.value, bound_type: btCoupled.value,
      cond_feature: cond.value, intercept: +intercept.value || 0, slope: +slope.value || 0,
    };
  };
  return row;
}

function openPersonaModal(p) {
  editingBuiltin = p ? p.builtin : false;
  $("modal-title").textContent = p ? `Edit persona: ${p.name}` : "New persona";
  $("p-name").value = p ? p.name : "";
  $("p-name").disabled = !!p;
  $("p-feedback").value = p ? p.feedback_text : "";
  $("p-description").value = p ? p.description : "";
  const rows = $("bound-rows");
  rows.innerHTML = "";
  for (const b of (p ? p.bounds : [])) rows.appendChild(boundRow(b));
  $("modal-backdrop").classList.add("open");
}

async function savePersona() {
  const name = $("p-name").value.trim();
  if (!name) { setStatus("persona name required", "error"); return; }
  const bounds = [...$("bound-rows").children].map((r) => r.toJSON());
  const payload = {
    name,
    feedback_text: $("p-feedback").value,
    description: $("p-description").value,
    bounds,
  };
  const data = await api("/api/personas", payload, "saving persona");
  personas = data.personas;
  currentPersona = name;
  refreshPersonaSelect();
  onPersonaChange(name);
  $("modal-backdrop").classList.remove("open");
  if (editingBuiltin) setStatus("built-in persona edited in memory only (until restart)");
}

// ---------------------------------------------------------------------------
// Pipeline actions
// ---------------------------------------------------------------------------

async function runBase() {
  const data = await api("/api/base_rollout", {
    arm_aa: armAA, goal: getGoal(), persona: currentPersona,
  }, "running base MPC rollout (may take a while)");
  setTraj("base", data.trajectory);
  baseTrigger = data.trigger_step;
  showStart = false;
  clearTraj("correction", "full", "generated");
  generatedBounds = [];
  clusters = [];
  selectedCluster = null;
  selectedClusterSegments = null;
  renderClusterList();
  $("generate").disabled = false;
  $("recluster").disabled = true;
  $("generate-cost").disabled = true;
  const trig = data.trigger_step === null
    ? "never violates — no feedback would be given (pick a different goal/persona)"
    : `user interrupts at frame ${data.trigger_step}`;
  $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) + "\n" + trig;
  $("mdm-start-hint").textContent = data.trigger_step === null
    ? "MDM will generate from the start pose (base never violates)."
    : `MDM will generate from the feedback-trigger pose at frame ${data.trigger_step} (purple dashed arm).`;
  refreshLegend(); refreshTimeline(); renderAll();
}

function applyClusterPayload(data) {
  clusters = data.clusters;
  selectedCluster = null;
  selectedClusterSegments = null;
  clearTraj("correction", "full", "generated");
  generatedBounds = [];
  $("generate-cost").disabled = true;
  $("correction-metrics").textContent = "";
  renderClusterList();
  refreshLegend(); refreshTimeline(); renderAll();
}

async function generate() {
  const data = await api("/api/generate", {
    prompt: $("prompt").value,
    n_samples: +$("n-samples").value,
    n_clusters: +$("n-clusters").value,
    scale: getScale(),
  }, "generating MDM samples + assembling cluster paths (this can take minutes)");
  $("recluster").disabled = false;
  applyClusterPayload(data);
}

async function recluster() {
  const data = await api("/api/recluster", {
    n_clusters: +$("n-clusters").value, scale: getScale(),
  }, "re-clustering + assembling cluster paths");
  applyClusterPayload(data);
}

async function pickCluster(label) {
  await api("/api/pick_cluster", { label }, "selecting cluster");
  selectedCluster = label;
  const c = clusters.find((x) => x.label === label);
  selectedClusterSegments = c.full_segments;
  setTraj("full", c.full);
  $("correction-metrics").textContent =
    "full path: " + fmtMetrics(c.full_metrics, c.full_goal_reach);
  $("generate-cost").disabled = false;
  renderClusterList();
}

async function generateCost() {
  const backend = $("cost-backend").value;
  const data = await api("/api/generate_cost", { backend },
    `generating ${backend} cost + evaluation rollout — this can take several minutes`);
  generatedBounds = data.generated_bounds || [];
  setTraj("generated", data.trajectory);
  const out = $("cost-output");
  out.innerHTML = "";
  const desc = document.createElement("div");
  desc.innerHTML = `<b>${data.description || "(no description)"}</b><br>` +
    fmtMetrics(data.metrics, data.goal_reach).replace("\n", "<br>") +
    `<br>artifacts: ${data.artifact_dir}`;
  const pre = document.createElement("pre");
  pre.textContent = data.code;
  out.appendChild(desc);
  out.appendChild(pre);
}

// ---------------------------------------------------------------------------
// Cluster list UI
// ---------------------------------------------------------------------------

function renderClusterList() {
  const list = $("cluster-list");
  list.innerHTML = "";
  for (const c of clusters) {
    const card = document.createElement("div");
    card.className = "cluster-card" + (c.label === selectedCluster ? " selected" : "");
    const sw = document.createElement("div");
    sw.className = "cluster-swatch";
    sw.style.background = CLUSTER_COLORS[c.label % CLUSTER_COLORS.length];
    const info = document.createElement("div");
    info.className = "cluster-info";
    info.innerHTML = `<b>cluster ${c.label}</b> · ${c.count} samples · ` +
      `${c.full_goal_reach.reached ? "reaches goal" : "misses goal"}<br>` +
      `oracle ${c.oracle_score.toFixed(3)} · full-path viol ${c.full_metrics.mean_violation.toFixed(3)}`;
    card.appendChild(sw);
    card.appendChild(info);
    card.onclick = () => pickCluster(c.label);
    list.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// Legend / timeline
// ---------------------------------------------------------------------------

function refreshLegend() {
  const legend = $("legend");
  legend.innerHTML = "";
  const mk = (label, color, checked, onchange) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checked;
    cb.onchange = () => { onchange(cb.checked); renderAll(); refreshTimeline(); };
    const sw = document.createElement("span");
    sw.className = "legend-swatch";
    sw.style.background = color;
    lab.appendChild(cb); lab.appendChild(sw);
    lab.appendChild(document.createTextNode(label));
    legend.appendChild(lab);
  };
  if (startPreview) mk("start pose", START_COLOR, showStart, (v) => { showStart = v; });
  if (trajs.base) mk("MDM start pose", MDM_START_COLOR, showMdmStart, (v) => { showMdmStart = v; });
  for (const [key, t] of Object.entries(trajs)) {
    mk(t.label, t.color, t.visible, (v) => { trajs[key].visible = v; });
  }
  if (clusters.length) {
    mk("cluster traces", "#888", showClusters, (v) => { showClusters = v; });
  }
}

function refreshTimeline() {
  const slider = $("frame-slider");
  slider.max = Math.max(0, maxFrames() - 1);
  if (+slider.value > +slider.max) slider.value = slider.max;
  frame = +slider.value;
  $("frame-label").textContent = `frame ${frame} / ${slider.max}`;
}

function tick(ts) {
  if (playing) {
    if (ts - lastTick > 50) {
      lastTick = ts;
      const slider = $("frame-slider");
      frame = (+slider.value + 1) % (+slider.max + 1 || 1);
      slider.value = frame;
      $("frame-label").textContent = `frame ${frame} / ${slider.max}`;
      renderAll();
    }
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// Skeleton rendering
// ---------------------------------------------------------------------------

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return null;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}

function goalWorld() {
  const g = getGoal();
  return [0, 1, 2].map((i) => INIT.spine3_pos[i] + g[i]);
}

function renderSkeleton() {
  const canvas = $("skeleton-canvas");
  const fit = fitCanvas(canvas);
  if (!fit) return;
  const { ctx, w, h } = fit;
  ctx.clearRect(0, 0, w, h);
  const vw = w / VIEWS.length;

  // Gather points for shared world bounds so all views use the same scale.
  const pts = INIT.body_pos.slice();
  pts.push(goalWorld());
  for (const [, t] of visibleTrajs()) {
    for (const f of t.data.arm_positions) pts.push(f[WRIST]);
  }
  if (showClusters) {
    for (const c of clusters) for (const f of c.full.arm_positions) pts.push(f[WRIST]);
  }
  if (startPreview) for (const p of startPreview) pts.push(p);

  VIEWS.forEach((view, vi) => {
    const x0 = vi * vw;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, 0, vw, h);
    ctx.clip();

    let hmin = Infinity, hmax = -Infinity, vmin = Infinity, vmax = -Infinity;
    for (const p of pts) {
      hmin = Math.min(hmin, p[view.hi]); hmax = Math.max(hmax, p[view.hi]);
      vmin = Math.min(vmin, p[view.vi]); vmax = Math.max(vmax, p[view.vi]);
    }
    const pad = 0.12;
    hmin -= pad; hmax += pad; vmin -= pad; vmax += pad;
    const scale = Math.min(vw / (hmax - hmin), h / (vmax - vmin));
    const cx = x0 + vw / 2, cy = h / 2;
    const hc = (hmin + hmax) / 2, vc = (vmin + vmax) / 2;
    const X = (p) => cx + (p[view.hi] - hc) * scale;
    const Y = (p) => cy - (p[view.vi] - vc) * scale;

    // title
    ctx.fillStyle = "#667";
    ctx.font = "11px sans-serif";
    ctx.fillText(view.name, x0 + 8, 14);

    // background body (left-arm bones excluded server-side)
    ctx.strokeStyle = "#c3c9d1";
    ctx.lineWidth = 2;
    for (const [p, c] of INIT.bone_pairs) {
      ctx.beginPath();
      ctx.moveTo(X(INIT.body_pos[p]), Y(INIT.body_pos[p]));
      ctx.lineTo(X(INIT.body_pos[c]), Y(INIT.body_pos[c]));
      ctx.stroke();
    }

    // goal star
    const gp = goalWorld();
    ctx.strokeStyle = "#c92a9b";
    ctx.lineWidth = 2;
    const gx = X(gp), gy = Y(gp), r = 7;
    for (let k = 0; k < 4; k++) {
      const a = (Math.PI / 4) * k;
      ctx.beginPath();
      ctx.moveTo(gx - r * Math.cos(a), gy - r * Math.sin(a));
      ctx.lineTo(gx + r * Math.cos(a), gy + r * Math.sin(a));
      ctx.stroke();
    }

    // cluster full-path wrist traces
    if (showClusters) {
      for (const c of clusters) {
        ctx.strokeStyle = CLUSTER_COLORS[c.label % CLUSTER_COLORS.length];
        ctx.lineWidth = c.label === selectedCluster ? 2 : 1;
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        c.full.arm_positions.forEach((f, i) => {
          const px = X(f[WRIST]), py = Y(f[WRIST]);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // main trajectories: wrist trace + arm chain at current frame
    for (const [key, t] of visibleTrajs()) {
      const T = t.data.n_frames;
      // trace, red where the hidden bounds are violated
      ctx.lineWidth = 1.5;
      for (let i = 1; i < T; i++) {
        ctx.strokeStyle = t.data.violations[i] > 0 ? "#ff2222" : t.color;
        ctx.globalAlpha = t.data.violations[i] > 0 ? 0.9 : 0.45;
        ctx.beginPath();
        ctx.moveTo(X(t.data.arm_positions[i - 1][WRIST]), Y(t.data.arm_positions[i - 1][WRIST]));
        ctx.lineTo(X(t.data.arm_positions[i][WRIST]), Y(t.data.arm_positions[i][WRIST]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      // arm chain at the scrubbed frame
      const f = t.data.arm_positions[Math.min(frame, T - 1)];
      ctx.strokeStyle = t.color;
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      f.forEach((p, i) => { if (i === 0) ctx.moveTo(X(p), Y(p)); else ctx.lineTo(X(p), Y(p)); });
      ctx.stroke();
      ctx.fillStyle = t.color;
      for (const p of f) { ctx.beginPath(); ctx.arc(X(p), Y(p), 3, 0, 7); ctx.fill(); }
      // trigger marker on the base trajectory
      if (key === "base" && baseTrigger !== null) {
        const tp = t.data.arm_positions[baseTrigger][WRIST];
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(X(tp), Y(tp), 6, 0, 7); ctx.stroke();
      }
    }

    // start-pose preview arm
    if (startPreview && showStart) {
      ctx.strokeStyle = START_COLOR;
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      startPreview.forEach((p, i) => { if (i === 0) ctx.moveTo(X(p), Y(p)); else ctx.lineTo(X(p), Y(p)); });
      ctx.stroke();
    }

    // MDM start pose: the arm MDM generation is conditioned on — the
    // feedback-trigger frame, or the start pose when the base never violates
    if (showMdmStart && trajs.base && trajs.base.data) {
      const src = baseTrigger !== null
        ? trajs.base.data.arm_positions[baseTrigger]
        : trajs.base.data.arm_positions[0];
      ctx.strokeStyle = MDM_START_COLOR;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([7, 4]);
      ctx.beginPath();
      src.forEach((p, i) => { if (i === 0) ctx.moveTo(X(p), Y(p)); else ctx.lineTo(X(p), Y(p)); });
      ctx.stroke();
      ctx.setLineDash([]);
      const wp = src[WRIST];
      ctx.fillStyle = MDM_START_COLOR;
      ctx.beginPath(); ctx.arc(X(wp), Y(wp), 4, 0, 7); ctx.fill();
    }

    ctx.restore();
    if (vi > 0) {
      ctx.strokeStyle = "#e0e4ea";
      ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, h); ctx.stroke();
    }
  });
}

// ---------------------------------------------------------------------------
// Feature graphs
// ---------------------------------------------------------------------------

function buildGraphs() {
  const holder = $("graphs");
  holder.innerHTML = "";
  // Pose-dependent (coupled) bounds get a feature-vs-feature phase plot,
  // rebuilt per persona; empty for personas without a coupled bound.
  const coupled = document.createElement("div");
  coupled.id = "coupled-graphs";
  holder.appendChild(coupled);
  const controls = document.createElement("div");
  controls.className = "panel graph-controls";
  const title = document.createElement("div");
  title.className = "graph-title";
  title.textContent = "Feature limits";
  controls.appendChild(title);
  const addToggle = (label, color, checked, onchange) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checked;
    cb.onchange = () => { onchange(cb.checked); renderAll(); };
    const sw = document.createElement("span");
    sw.className = "legend-swatch graph-limit-swatch";
    sw.style.background = color;
    lab.appendChild(cb); lab.appendChild(sw);
    lab.appendChild(document.createTextNode(label));
    controls.appendChild(lab);
  };
  addToggle("oracle limits", "rgba(180,40,40,0.8)", showOracleBounds,
    (v) => { showOracleBounds = v; });
  addToggle("generated-cost limits", GENERATED_LIMIT_COLOR, showGeneratedBounds,
    (v) => { showGeneratedBounds = v; });
  holder.appendChild(controls);
  for (const name of INIT.feature_names) {
    const panel = document.createElement("div");
    panel.className = "panel";
    const title = document.createElement("div");
    title.className = "graph-title";
    title.textContent = name;
    const canvas = document.createElement("canvas");
    canvas.className = "feature-graph";
    canvas.id = `graph-${name}`;
    panel.appendChild(title);
    panel.appendChild(canvas);
    holder.appendChild(panel);
  }
}

function personaConstantBounds(feature) {
  const p = getPersona();
  if (!p) return [];
  return p.bounds.filter((b) => b.kind === "hidden" && b.feature === feature);
}

function renderGraphs() {
  const T = maxFrames();
  for (const name of INIT.feature_names) {
    const canvas = $(`graph-${name}`);
    if (!canvas) continue;
    const fit = fitCanvas(canvas);
    if (!fit) continue;
    const { ctx, w, h } = fit;
    ctx.clearRect(0, 0, w, h);
    const ml = 34, mr = 6, mt = 6, mb = 16;
    const pw = w - ml - mr, ph = h - mt - mb;

    // y-range from visible series + bounds
    let ymin = Infinity, ymax = -Infinity;
    const seriesList = [];
    for (const [, t] of visibleTrajs()) {
      const s = t.data.features[name];
      seriesList.push({ s, color: t.color, bounds: t.data.bounds });
      for (const v of s) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
    }
    if (showClusters) {
      for (const c of clusters) {
        const s = c.full.features[name];
        seriesList.push({ s, color: CLUSTER_COLORS[c.label % CLUSTER_COLORS.length], alpha: 0.5 });
        for (const v of s) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
      }
    }
    const oracleBounds = showOracleBounds ? personaConstantBounds(name) : [];
    const costBounds = showGeneratedBounds
      ? generatedBounds.filter((b) => b.feature === name)
      : [];
    for (const b of [...oracleBounds, ...costBounds]) {
      if (b.low !== null && b.low !== undefined) { ymin = Math.min(ymin, b.low); ymax = Math.max(ymax, b.low); }
      if (b.high !== null && b.high !== undefined) { ymin = Math.min(ymin, b.high); ymax = Math.max(ymax, b.high); }
    }
    if (!seriesList.length) { ymin = Math.min(ymin, -0.2); ymax = Math.max(ymax, Math.PI); }
    if (!isFinite(ymin)) { ymin = 0; ymax = Math.PI; }
    const span = Math.max(0.3, ymax - ymin);
    ymin -= span * 0.15; ymax += span * 0.15;

    const X = (i) => ml + (i / Math.max(1, T - 1)) * pw;
    const Y = (v) => mt + (1 - (v - ymin) / (ymax - ymin)) * ph;

    // axes + ticks
    ctx.strokeStyle = "#d0d5dc";
    ctx.strokeRect(ml, mt, pw, ph);
    ctx.fillStyle = "#889";
    ctx.font = "9px sans-serif";
    for (let k = 0; k <= 3; k++) {
      const v = ymin + ((ymax - ymin) * k) / 3;
      ctx.fillText(v.toFixed(2), 4, Y(v) + 3);
      ctx.strokeStyle = "#eef0f4";
      ctx.beginPath(); ctx.moveTo(ml, Y(v)); ctx.lineTo(ml + pw, Y(v)); ctx.stroke();
    }

    const drawBounds = (bounds, fill, stroke) => {
      ctx.fillStyle = fill;
      for (const b of bounds) {
        if (b.bound_type === "upper_bound") {
          ctx.fillRect(ml, mt, pw, Y(b.high) - mt);
        } else if (b.bound_type === "lower_bound") {
          ctx.fillRect(ml, Y(b.low), pw, mt + ph - Y(b.low));
        } else if (b.bound_type === "avoid_band") {
          ctx.fillRect(ml, Y(b.high), pw, Y(b.low) - Y(b.high));
        } else if (b.bound_type === "band") {
          ctx.fillRect(ml, Y(b.low), pw, mt + ph - Y(b.low));
          ctx.fillRect(ml, mt, pw, Y(b.high) - mt);
        }
      }
      ctx.strokeStyle = stroke;
      ctx.setLineDash([5, 3]);
      for (const b of bounds) {
        for (const v of [b.low, b.high]) {
          if (v === null || v === undefined) continue;
          ctx.beginPath(); ctx.moveTo(ml, Y(v)); ctx.lineTo(ml + pw, Y(v)); ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    };
    drawBounds(oracleBounds, "rgba(214,69,65,0.13)", "rgba(180,40,40,0.8)");
    drawBounds(costBounds, "rgba(53,103,214,0.10)", GENERATED_LIMIT_COLOR);

    // coupled per-frame thresholds (dashed, per trajectory)
    for (const { color, bounds } of showOracleBounds ? seriesList : []) {
      if (!bounds) continue;
      for (const b of bounds) {
        if (b.feature !== name) continue;
        const thr = Array.isArray(b.low) ? b.low : Array.isArray(b.high) ? b.high : null;
        if (!thr) continue;
        ctx.strokeStyle = color;
        ctx.setLineDash([3, 3]);
        ctx.globalAlpha = 0.8;
        ctx.beginPath();
        thr.forEach((v, i) => {
          const y = Math.min(mt + ph, Math.max(mt, Y(v)));
          if (i === 0) ctx.moveTo(X(i), y); else ctx.lineTo(X(i), y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
    }

    // series
    for (const { s, color, alpha } of seriesList) {
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha || 1;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      s.forEach((v, i) => { if (i === 0) ctx.moveTo(X(i), Y(v)); else ctx.lineTo(X(i), Y(v)); });
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // time cursor
    ctx.strokeStyle = "#333";
    ctx.beginPath(); ctx.moveTo(X(frame), mt); ctx.lineTo(X(frame), mt + ph); ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Coupled-feature phase graphs (pose-dependent limits)
// ---------------------------------------------------------------------------
//
// For a CoupledBound the limit on `feature` moves with `cond_feature`
// (threshold = intercept + slope·cond). A time series can't show that the
// limit itself is sliding, so we plot cond_feature (x) against feature (y):
// the bound becomes a straight line, the violating side is shaded, and each
// trajectory traces a path through the plane with a dot at the scrubbed frame.

function currentCoupledBounds() {
  const p = getPersona();
  if (!p) return [];
  return p.bounds.filter((b) => b.kind === "coupled");
}

function buildCoupledGraphs() {
  const holder = $("coupled-graphs");
  if (!holder) return;
  holder.innerHTML = "";
  currentCoupledBounds().forEach((b, i) => {
    const panel = document.createElement("div");
    panel.className = "panel";
    const title = document.createElement("div");
    title.className = "graph-title";
    title.textContent = `${b.feature} vs ${b.cond_feature} (pose-dependent limit)`;
    const canvas = document.createElement("canvas");
    canvas.className = "coupled-graph";
    canvas.id = `coupled-graph-${i}`;
    panel.appendChild(title);
    panel.appendChild(canvas);
    holder.appendChild(panel);
  });
}

function coupledSeriesList(b) {
  // (x=cond, y=feature) paths for the same trajectories the feature graphs show.
  const list = [];
  for (const [, t] of visibleTrajs()) {
    list.push({
      xs: t.data.features[b.cond_feature],
      ys: t.data.features[b.feature],
      color: t.color,
      alpha: 1,
    });
  }
  if (showClusters) {
    for (const c of clusters) {
      list.push({
        xs: c.full.features[b.cond_feature],
        ys: c.full.features[b.feature],
        color: CLUSTER_COLORS[c.label % CLUSTER_COLORS.length],
        alpha: 0.5,
      });
    }
  }
  return list;
}

function renderCoupledGraphs() {
  currentCoupledBounds().forEach((b, gi) => {
    const canvas = $(`coupled-graph-${gi}`);
    if (!canvas) return;
    const fit = fitCanvas(canvas);
    if (!fit) return;
    const { ctx, w, h } = fit;
    ctx.clearRect(0, 0, w, h);
    const ml = 48, mr = 10, mt = 8, mb = 30;
    const pw = w - ml - mr, ph = h - mt - mb;
    const thr = (x) => b.intercept + b.slope * x;

    const series = coupledSeriesList(b);
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const { xs, ys } of series) {
      for (const v of xs) { xmin = Math.min(xmin, v); xmax = Math.max(xmax, v); }
      for (const v of ys) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
    }
    if (!isFinite(xmin)) { xmin = -0.2; xmax = 2.6; ymin = -0.2; ymax = 2.6; }
    // keep the limit line in view across the plotted x-range
    for (const x of [xmin, xmax]) { const y = thr(x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); }
    // keep generated-cost limits on either feature in view
    const genBounds = showGeneratedBounds
      ? generatedBounds.filter((g) => g.feature === b.feature || g.feature === b.cond_feature)
      : [];
    for (const g of genBounds) {
      for (const v of [g.low, g.high]) {
        if (v === null || v === undefined) continue;
        if (g.feature === b.feature) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
        else { xmin = Math.min(xmin, v); xmax = Math.max(xmax, v); }
      }
    }
    const xs = Math.max(0.3, xmax - xmin), ys = Math.max(0.3, ymax - ymin);
    xmin -= xs * 0.1; xmax += xs * 0.1; ymin -= ys * 0.1; ymax += ys * 0.1;

    const X = (v) => ml + ((v - xmin) / (xmax - xmin)) * pw;
    const Y = (v) => mt + (1 - (v - ymin) / (ymax - ymin)) * ph;
    const violates = (x, y) => b.bound_type === "upper_bound" ? y > thr(x) : y < thr(x);

    ctx.save();
    ctx.beginPath(); ctx.rect(ml, mt, pw, ph); ctx.clip();

    // shaded violating half-plane (past the pose-dependent limit)
    const lx0 = X(xmin), ly0 = Y(thr(xmin)), lx1 = X(xmax), ly1 = Y(thr(xmax));
    const far = b.bound_type === "upper_bound" ? mt - ph : mt + 2 * ph;
    ctx.fillStyle = "rgba(214,69,65,0.13)";
    ctx.beginPath();
    ctx.moveTo(lx0, ly0); ctx.lineTo(lx1, ly1);
    ctx.lineTo(lx1, far); ctx.lineTo(lx0, far); ctx.closePath(); ctx.fill();

    // limit line
    ctx.strokeStyle = "rgba(180,40,40,0.85)";
    ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]);
    ctx.beginPath(); ctx.moveTo(lx0, ly0); ctx.lineTo(lx1, ly1); ctx.stroke();
    ctx.setLineDash([]);

    // generated-cost limits: a constant single-feature bound is a horizontal
    // line (on the y feature) or a vertical line (on the x/cond feature) here —
    // an axis-aligned approximation of the diagonal pose-dependent limit.
    if (genBounds.length) {
      ctx.fillStyle = "rgba(53,103,214,0.10)";
      const shadeH = (v, above) => { const y = Y(v); above ? ctx.fillRect(ml, mt, pw, y - mt) : ctx.fillRect(ml, y, pw, mt + ph - y); };
      const shadeV = (v, right) => { const x = X(v); right ? ctx.fillRect(x, mt, ml + pw - x, ph) : ctx.fillRect(ml, mt, x - ml, ph); };
      ctx.strokeStyle = GENERATED_LIMIT_COLOR; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]);
      for (const g of genBounds) {
        const onY = g.feature === b.feature;
        if (g.bound_type === "upper_bound") onY ? shadeH(g.high, true) : shadeV(g.high, true);
        else if (g.bound_type === "lower_bound") onY ? shadeH(g.low, false) : shadeV(g.low, false);
        else if (g.bound_type === "band") {
          if (onY) { shadeH(g.low, false); shadeH(g.high, true); }
          else { shadeV(g.low, false); shadeV(g.high, true); }
        }
        for (const v of [g.low, g.high]) {
          if (v === null || v === undefined) continue;
          ctx.beginPath();
          if (onY) { ctx.moveTo(ml, Y(v)); ctx.lineTo(ml + pw, Y(v)); }
          else { ctx.moveTo(X(v), mt); ctx.lineTo(X(v), mt + ph); }
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }

    // trajectory paths, red on the violating side; dot at the scrubbed frame
    for (const { xs: sx, ys: sy, color, alpha } of series) {
      const T = sx.length;
      ctx.lineWidth = 1.6;
      for (let i = 1; i < T; i++) {
        const bad = violates(sx[i], sy[i]);
        ctx.strokeStyle = bad ? "#ff2222" : color;
        ctx.globalAlpha = bad ? 0.9 : alpha;
        ctx.beginPath();
        ctx.moveTo(X(sx[i - 1]), Y(sy[i - 1]));
        ctx.lineTo(X(sx[i]), Y(sy[i]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      const fi = Math.min(frame, T - 1);
      ctx.fillStyle = violates(sx[fi], sy[fi]) ? "#ff2222" : color;
      ctx.strokeStyle = "#111"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(X(sx[fi]), Y(sy[fi]), 4, 0, 7); ctx.fill(); ctx.stroke();
    }
    ctx.restore();

    // frame + ticks + axis labels
    ctx.strokeStyle = "#d0d5dc"; ctx.lineWidth = 1;
    ctx.strokeRect(ml, mt, pw, ph);
    ctx.fillStyle = "#889"; ctx.font = "9px sans-serif";
    ctx.textAlign = "right";
    for (let k = 0; k <= 3; k++) {
      const yv = ymin + ((ymax - ymin) * k) / 3;
      ctx.fillText(yv.toFixed(2), ml - 4, Y(yv) + 3);
    }
    ctx.textAlign = "center";
    for (let k = 0; k <= 3; k++) {
      const xv = xmin + ((xmax - xmin) * k) / 3;
      ctx.fillText(xv.toFixed(2), X(xv), mt + ph + 11);
    }
    ctx.fillStyle = "#667";
    ctx.fillText(`${b.cond_feature} →`, ml + pw / 2, h - 3);
    ctx.save();
    ctx.translate(9, mt + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(`${b.feature} →`, 0, 0);
    ctx.restore();
    ctx.textAlign = "left";
  });
}

// ---------------------------------------------------------------------------
// Violation strips
// ---------------------------------------------------------------------------

function renderStrips() {
  const canvas = $("violation-strips");
  const fit = fitCanvas(canvas);
  if (!fit) return;
  const { ctx, w, h } = fit;
  ctx.clearRect(0, 0, w, h);
  const entries = visibleTrajs();
  if (!entries.length) return;
  const T = maxFrames();
  const rh = Math.min(8, h / entries.length);
  entries.forEach(([key, t], row) => {
    const y = row * rh;
    for (let i = 0; i < t.data.n_frames; i++) {
      ctx.fillStyle = t.data.violations[i] > 0 ? "#e02020" : t.color + "44";
      ctx.fillRect((i / T) * w, y, Math.max(1, w / T), rh - 1);
    }
    if (key === "base" && baseTrigger !== null) {
      ctx.fillStyle = "#000";
      ctx.fillRect((baseTrigger / T) * w - 1, y, 2, rh - 1);
    }
    if (key === "full" && selectedClusterSegments) {
      const mdmEnd = selectedClusterSegments.history + selectedClusterSegments.correction;
      const x = (mdmEnd / T) * w;
      ctx.fillStyle = "#000";
      ctx.fillRect(x - 1, y, 2, rh - 1);
      ctx.fillStyle = "#000";
      ctx.font = "9px sans-serif";
      ctx.fillText("MDM end", x + 3, y + rh - 2);
    }
  });
  // cursor
  ctx.strokeStyle = "#333";
  ctx.beginPath();
  ctx.moveTo((frame / T) * w, 0);
  ctx.lineTo((frame / T) * w, h);
  ctx.stroke();
}

function renderAll() {
  renderSkeleton();
  renderGraphs();
  renderCoupledGraphs();
  renderStrips();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function main() {
  INIT = await api("/api/init", undefined, "loading");
  personas = INIT.personas;
  currentPersona = INIT.current_persona;

  buildArmEditor();
  buildGraphs();
  refreshPersonaSelect();
  setArmEditorValues(INIT.start_arm_aa);
  onPersonaChange(currentPersona);

  $("n-samples").value = INIT.uq.diffusion_samples;
  $("n-clusters").value = INIT.uq.n_clusters;
  $("scale").value = INIT.uq.scale;
  $("scale-num").value = INIT.uq.scale;
  $("cost-backend").value = INIT.cost_backend;

  $("persona-select").onchange = (e) => onPersonaChange(e.target.value);
  $("persona-edit").onclick = () => openPersonaModal(getPersona());
  $("persona-new").onclick = () => openPersonaModal(null);
  $("persona-delete").onclick = async () => {
    const data = await apiDelete(`/api/personas/${currentPersona}`, "deleting persona");
    personas = data.personas;
    currentPersona = personas[0].name;
    refreshPersonaSelect();
    onPersonaChange(currentPersona);
  };
  $("modal-cancel").onclick = () => $("modal-backdrop").classList.remove("open");
  $("modal-save").onclick = savePersona;
  $("add-bound").onclick = () => $("bound-rows").appendChild(boundRow({}));

  $("reset-pose").onclick = () => setArmEditorValues(INIT.start_arm_aa);
  $("run-base").onclick = runBase;
  $("generate").onclick = generate;
  $("recluster").onclick = recluster;
  $("generate-cost").onclick = generateCost;

  $("scale").oninput = () => { $("scale-num").value = $("scale").value; };
  $("scale").onchange = onScaleChange;
  $("scale-num").onchange = () => { $("scale").value = $("scale-num").value; onScaleChange(); };
  async function onScaleChange() {
    if (!clusters.length) return;
    const prev = selectedCluster;
    await recluster();
    if (prev !== null) await pickCluster(prev);
  }

  $("frame-slider").oninput = () => {
    frame = +$("frame-slider").value;
    $("frame-label").textContent = `frame ${frame} / ${$("frame-slider").max}`;
    renderAll();
  };
  $("play").onclick = () => {
    playing = !playing;
    $("play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
  };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    const slider = $("frame-slider");
    if (e.key === "ArrowRight") { slider.value = Math.min(+slider.max, frame + 1); slider.oninput(); }
    if (e.key === "ArrowLeft") { slider.value = Math.max(0, frame - 1); slider.oninput(); }
    if (e.key === " ") { e.preventDefault(); $("play").onclick(); }
  });
  [$("goal-x"), $("goal-y"), $("goal-z")].forEach((el) => { el.onchange = renderAll; });

  // server console: poll the stdout tee so long-running stages show progress
  const consoleEl = $("console");
  let logIndex = 0;
  setInterval(async () => {
    try {
      const resp = await fetch(`/api/logs?since=${logIndex}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.lines.length) {
        const atBottom =
          consoleEl.scrollTop + consoleEl.clientHeight >= consoleEl.scrollHeight - 24;
        consoleEl.textContent += data.lines.join("\n") + "\n";
        if (consoleEl.textContent.length > 200000) {
          consoleEl.textContent = consoleEl.textContent.slice(-200000);
        }
        if (atBottom) consoleEl.scrollTop = consoleEl.scrollHeight;
      }
      logIndex = data.next;
    } catch (err) { /* server busy or down; retry next poll */ }
  }, 1200);

  window.addEventListener("resize", renderAll);
  refreshLegend();
  refreshTimeline();
  renderAll();
  requestAnimationFrame(tick);
}

main();
