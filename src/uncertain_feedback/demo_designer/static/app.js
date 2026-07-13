import * as THREE from "three";

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
let clusterDepth = 0;
let clusterPath = [];
let canGoBack = false;
let showClusters = true;
let showStart = true;
let showMdmStart = false;
let showOracleBounds = true;
let showGeneratedBounds = true;
let generatedBounds = [];
let costField = null;       // {features:{name:[..]}, penalty:[..]} — compiled cost sampled over a pose cloud
let startPreview = null;    // {arm_positions, mesh_vertices} for the edited pose
const meshData = new Map(); // mesh id -> {frames, vertices, data: Float32Array}
const meshLoads = new Map();
let baseTrigger = null;
let multiTurnActive = false;

function resetClusterNavigation() {
  clusterDepth = 0;
  clusterPath = [];
  canGoBack = false;
}

// Multi-round state: committed rounds + the unified replacement cost. The
// unified visuals survive base rollouts (only Reset rounds clears them).
let rounds = [];
let unified = null;         // {description, code} or null
let unifiedCostField = null;
let showUnifiedBounds = true;

const MDM_START_COLOR = "#7b2fbe";
const GENERATED_LIMIT_COLOR = "#3567d6";
const UNIFIED_COLOR = "#00838f";

let frame = 0;
let playing = false;
let lastTick = 0;

const TRAJ_STYLES = {
  base: { color: "#e05252", label: "base rollout" },
  oracle: { color: "#9c5f17", label: "oracle-cost rollout" },
  full: { color: "#0e7a63", label: "full corrected path" },
  generated: { color: "#3567d6", label: "generated-cost corrected path" },
  generated_start: { color: "#6fa8ff", label: "generated-cost from start" },
  unified: { color: UNIFIED_COLOR, label: "unified-cost rollout" },
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

// --- client-side hidden-bound math (mirrors simulated_users/base.py) --------
//
// Violations and coupled per-frame thresholds are computed here from the
// packaged `features` + the LIVE persona bounds, so dragging a bound on a
// graph updates every red segment/strip instantly. Only the joint-box part
// (`limit_violations`) is server-baked — the raw joint angles are not sent.

function hiddenViolation(b, v) {
  if (b.bound_type === "upper_bound") {
    return b.high === null ? 0 : Math.max(0, v - b.high);
  }
  if (b.bound_type === "lower_bound") {
    return b.low === null ? 0 : Math.max(0, b.low - v);
  }
  if (b.low === null || b.high === null) return 0;
  return Math.max(0, Math.min(v - b.low, b.high - v));
}

function computeViolations(data) {
  const out = (data.limit_violations || new Array(data.n_frames).fill(0)).slice();
  const p = getPersona();
  if (!p) return out;
  for (const b of p.bounds) {
    const f = data.features[b.feature];
    if (!f) continue;
    if (b.kind === "coupled") {
      const cond = data.features[b.cond_feature];
      if (!cond) continue;
      for (let i = 0; i < out.length; i++) {
        const thr = b.intercept + b.slope * cond[i];
        out[i] += b.bound_type === "upper_bound"
          ? Math.max(0, f[i] - thr) : Math.max(0, thr - f[i]);
      }
    } else {
      for (let i = 0; i < out.length; i++) out[i] += hiddenViolation(b, f[i]);
    }
  }
  return out;
}

function coupledThresholdSeries(b, data) {
  const cond = data.features[b.cond_feature];
  if (!cond) return null;
  return cond.map((c) => b.intercept + b.slope * c);
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
const JOINT_KEYS = ["left_shoulder", "left_elbow", "left_wrist"];
const AXES = ["x", "y", "z"];
let armAA = null; // (3,3)

function armLimit(jointIndex) {
  const p = getPersona();
  return p && p.joint_limits.find((limit) => limit.joint === JOINT_KEYS[jointIndex]);
}

function clampArmValue(value, jointIndex, axisIndex) {
  const limit = armLimit(jointIndex);
  return limit ? Math.min(limit.high[axisIndex], Math.max(limit.low[axisIndex], value)) : value;
}

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
      const limit = armLimit(j);
      lab.textContent = limit ? `${ax} [${limit.low[a]}, ${limit.high[a]}]` : ax;
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = limit ? limit.low[a] : -3.14;
      slider.max = limit ? limit.high[a] : 3.14;
      slider.step = 0.01;
      const num = document.createElement("input");
      num.type = "number";
      num.min = slider.min; num.max = slider.max;
      num.step = 0.01;
      slider.id = `arm-s-${j}-${a}`;
      num.id = `arm-n-${j}-${a}`;
      slider.oninput = () => { num.value = slider.value; armAA[j][a] = +slider.value; schedulePreview(); };
      num.onchange = () => {
        const value = clampArmValue(+num.value, j, a);
        num.value = value;
        slider.value = value;
        armAA[j][a] = value;
        schedulePreview();
      };
      grid.appendChild(lab);
      grid.appendChild(slider);
      grid.appendChild(num);
    });
  });
}

function setArmEditorValues(aa) {
  armAA = aa.map((row, j) => row.map((value, a) => clampArmValue(value, j, a)));
  for (let j = 0; j < 3; j++) for (let a = 0; a < 3; a++) {
    $(`arm-s-${j}-${a}`).value = armAA[j][a].toFixed(3);
    $(`arm-n-${j}-${a}`).value = armAA[j][a].toFixed(3);
  }
  schedulePreview();
}

let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    const data = await api("/api/preview_pose", { arm_aa: armAA });
    startPreview = data;
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
  const currentArmAA = armAA;
  buildArmEditor();
  setArmEditorValues(currentArmAA.map((row, j) =>
    row.map((value, a) => clampArmValue(value, j, a))));
  buildCoupledGraphs();
  renderBoundControls();
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
  if (data.retriggered) {
    baseTrigger = data.trigger_step;
    applyTriggerHint(data.trigger_step);
  }
  onPersonaChange(name);
  $("modal-backdrop").classList.remove("open");
  if (editingBuiltin) setStatus("built-in persona edited in memory only (until restart)");
}

// ---------------------------------------------------------------------------
// Pipeline actions
// ---------------------------------------------------------------------------

function setScenarioLocked(locked) {
  for (const id of ["persona-select", "persona-edit", "persona-new", "persona-delete",
    "goal-x", "goal-y", "goal-z", "reset-pose"]) {
    $(id).disabled = locked;
  }
  for (const input of document.querySelectorAll("#arm-editor input")) input.disabled = locked;
}

async function runBase() {
  await startManualTrajectory();
}

function renderTrajectorySession(data) {
  const out = $("trajectory-session");
  out.className = `trajectory-session ${data.status}`;
  if (data.status === "paused") {
    const violation = data.trigger.violation === null
      ? "n/a"
      : data.trigger.violation.toFixed(3);
    out.textContent = `Paused at frame ${data.trigger.step}/${data.step_limit} · ` +
      `${data.trigger.reason} · violation ${violation} · ` +
      `feedback turn ${data.rounds.length + 1}`;
  } else {
    out.textContent = `Trajectory complete at frame ${data.step}/${data.step_limit}` +
      (data.reached_goal ? " · goal reached" : "") +
      (data.error ? ` · error: ${data.error}` : "");
  }
}

async function startManualTrajectory() {
  const data = await api("/api/manual_trajectory/start", {
    arm_aa: armAA,
    goal: getGoal(),
    persona: currentPersona,
  }, "advancing trajectory to the next feedback trigger");
  multiTurnActive = true;
  rounds = data.rounds;
  unified = data.unified;
  unifiedCostField = null;
  clearTraj("base", "oracle", "correction", "full", "generated", "generated_start");
  clearTraj("unified");
  setTraj("base", data.trajectory);
  clearTraj("oracle");
  baseTrigger = data.trigger ? data.trigger.step : null;
  showStart = false;
  clusters = [];
  selectedCluster = null;
  selectedClusterSegments = null;
  resetClusterNavigation();
  generatedBounds = [];
  costField = null;
  renderClusterList();
  $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) +
    `\nexecuted ${data.step}/${data.step_limit} steps` +
    (data.error ? `\nerror: ${data.error}` : "");
  renderTrajectorySession(data);
  setScenarioLocked(data.status === "paused");
  $("run-base").disabled = data.status === "paused";
  $("exit-trajectory").disabled = false;
  $("ignore-violation").disabled = data.status !== "paused" ||
    data.trigger.reason !== "discomfort";
  $("generate").disabled = data.status !== "paused";
  $("recluster").disabled = true;
  $("generate-cost").disabled = true;
  $("commit-round").disabled = true;
  $("apply-round").disabled = true;
  renderRounds();
  renderUnifiedOutput();
  refreshLegend(); refreshTimeline(); renderAll();
}

async function exitManualTrajectory() {
  const data = await api("/api/manual_trajectory/exit", {}, "exiting trajectory");
  multiTurnActive = false;
  baseTrigger = null;
  rounds = [];
  unified = null;
  unifiedCostField = null;
  clusters = [];
  selectedCluster = null;
  selectedClusterSegments = null;
  resetClusterNavigation();
  generatedBounds = [];
  costField = null;
  clearTraj(...Object.keys(trajs));
  showStart = true;
  showMdmStart = false;
  setScenarioLocked(false);
  $("persona-delete").disabled = getPersona().builtin;
  $("run-base").disabled = false;
  $("exit-trajectory").disabled = true;
  for (const id of ["ignore-violation", "generate", "recluster",
    "generate-cost", "commit-round", "apply-round"]) {
    $(id).disabled = true;
  }
  $("trajectory-session").className = "trajectory-session";
  $("trajectory-session").textContent = "";
  $("base-metrics").textContent = "";
  $("correction-metrics").textContent = "";
  $("cost-output").innerHTML = "";
  renderClusterList();
  renderRounds();
  renderUnifiedOutput();
  setStatus(`trajectory exited · artifacts retained at ${data.artifact_dir}`);
  refreshLegend(); refreshTimeline(); renderAll();
}

async function ignoreComfortViolation() {
  const data = await api("/api/manual_trajectory/ignore_violation", {},
    "ignoring the current comfort violation and advancing the trajectory");
  setTraj("base", data.trajectory);
  baseTrigger = data.trigger ? data.trigger.step : null;
  clearTraj("correction", "full", "generated", "generated_start");
  clusters = [];
  selectedCluster = null;
  selectedClusterSegments = null;
  resetClusterNavigation();
  generatedBounds = [];
  costField = null;
  renderClusterList();
  $("cost-output").innerHTML = "";
  $("correction-metrics").textContent = "";
  $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) +
    `\nexecuted ${data.step}/${data.step_limit} steps` +
    (data.error ? `\nerror: ${data.error}` : "");
  renderTrajectorySession(data);
  setScenarioLocked(data.status === "paused");
  $("run-base").disabled = data.status === "paused";
  $("ignore-violation").disabled = data.status !== "paused" ||
    data.trigger.reason !== "discomfort";
  $("generate").disabled = data.status !== "paused";
  $("recluster").disabled = true;
  $("generate-cost").disabled = true;
  $("commit-round").disabled = true;
  $("apply-round").disabled = true;
  refreshLegend(); refreshTimeline(); renderAll();
}

function applyClusterPayload(data) {
  clusters = data.clusters;
  selectedCluster = data.selected_label ?? null;
  clusterDepth = data.depth;
  clusterPath = data.path;
  canGoBack = data.can_go_back;
  $("scale").value = data.scale;
  $("scale-num").value = data.scale;
  clearTraj("correction", "full", "generated", "generated_start");
  generatedBounds = [];
  costField = null;
  const selected = clusters.find((c) => c.label === selectedCluster);
  selectedClusterSegments = selected ? selected.full_segments : null;
  if (selected) {
    setTraj("full", selected.full);
    $("correction-metrics").textContent =
      "full path: " + fmtMetrics(selected.full_metrics, selected.full_goal_reach);
  } else {
    $("correction-metrics").textContent = "";
  }
  $("generate-cost").disabled = !selected;
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

async function refineCluster() {
  if (selectedCluster === null) return;
  const data = await api("/api/refine_cluster", {
    label: selectedCluster,
    n_clusters: +$("n-clusters").value,
    scale: getScale(),
  }, "refining selected cluster + assembling child paths");
  applyClusterPayload(data);
}

async function backCluster() {
  const data = await api("/api/back_cluster", {},
    "restoring parent cluster level");
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

function renderCostGeneration(data) {
  generatedBounds = data.generated_bounds || [];
  costField = data.cost_field || null;
  setTraj("generated", data.trajectory);
  setTraj("generated_start", data.start_trajectory);
  const out = $("cost-output");
  out.innerHTML = "";
  const desc = document.createElement("div");
  desc.innerHTML = `<b>${data.description || "(no description)"}</b><br>` +
    "corrected path: " + fmtMetrics(data.metrics, data.goal_reach).replace("\n", "<br>") +
    "<br>from start: " + fmtMetrics(data.start_metrics, data.start_goal_reach).replace("\n", "<br>") +
    `<br>artifacts: ${data.artifact_dir}`;
  const pre = document.createElement("pre");
  pre.textContent = data.code;
  out.appendChild(desc);
  out.appendChild(pre);
}

async function generateCost() {
  const backend = $("cost-backend").value;
  const data = await api("/api/generate_cost", { backend },
    `generating ${backend} cost + evaluation rollout — this can take several minutes`);
  renderCostGeneration(data);
  $("commit-round").disabled = multiTurnActive;
  $("apply-round").disabled = !multiTurnActive;
}

// ---------------------------------------------------------------------------
// Multi-round feedback
// ---------------------------------------------------------------------------

function renderRounds() {
  const list = $("round-list");
  list.innerHTML = "";
  for (const r of rounds) {
    const div = document.createElement("div");
    div.className = "round-card";
    div.innerHTML = `<b>round ${r.index}</b> · goal [${r.goal.map((v) => v.toFixed(2)).join(", ")}]` +
      ` · trigger @ ${r.trigger_step}<br>“${r.feedback_text}”<br>${r.description || ""}`;
    list.appendChild(div);
  }
  $("combine-rounds").disabled = rounds.length < 2;
  $("reset-rounds").disabled = rounds.length === 0 && !unified;
}

function renderUnifiedOutput(extraHtml) {
  const out = $("unified-output");
  out.innerHTML = "";
  if (!unified) return;
  const desc = document.createElement("div");
  desc.innerHTML = `<b>unified: ${unified.description || "(no description)"}</b>` + (extraHtml || "");
  const pre = document.createElement("pre");
  pre.textContent = unified.code;
  out.appendChild(desc);
  out.appendChild(pre);
}

async function commitRound() {
  const data = await api("/api/commit_round", {}, "committing round");
  rounds = data.rounds;
  unified = data.unified;
  // Round 1's cost IS the unified cost, so its field carries over directly.
  if (rounds.length === 1) unifiedCostField = costField;
  if (!unified) {
    unifiedCostField = null;
    clearTraj("unified");
  }
  $("commit-round").disabled = true;
  $("apply-round").disabled = true;
  renderRounds();
  renderUnifiedOutput();
  renderAll();
}

async function applyRoundAndContinue() {
  const data = await api("/api/apply_round", {},
    "applying feedback and advancing the same trajectory to its next trigger");
  rounds = data.rounds;
  unified = data.unified;
  if (!unified) {
    unifiedCostField = null;
    clearTraj("unified");
  }
  setTraj("base", data.trajectory);
  baseTrigger = data.trigger ? data.trigger.step : null;
  clusters = [];
  selectedCluster = null;
  selectedClusterSegments = null;
  resetClusterNavigation();
  generatedBounds = [];
  costField = null;
  clearTraj("correction", "full", "generated", "generated_start");
  renderClusterList();
  $("cost-output").innerHTML = "";
  $("correction-metrics").textContent = "";
  $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) +
    `\nexecuted ${data.step}/${data.step_limit} steps` +
    (data.error ? `\nerror: ${data.error}` : "");
  renderTrajectorySession(data);
  setScenarioLocked(data.status === "paused");
  $("run-base").disabled = data.status === "paused";
  $("ignore-violation").disabled = data.status !== "paused" ||
    data.trigger.reason !== "discomfort";
  $("generate").disabled = data.status !== "paused";
  $("recluster").disabled = true;
  $("generate-cost").disabled = true;
  $("commit-round").disabled = true;
  $("apply-round").disabled = true;
  renderRounds();
  renderUnifiedOutput();
  refreshLegend(); refreshTimeline(); renderAll();
}

async function combineRounds() {
  const data = await api("/api/combine_rounds", {},
    "combining rounds with codex — this can take a long time; watch the console");
  rounds = data.rounds;
  unified = { description: data.description, code: data.code };
  unifiedCostField = data.cost_field || null;
  setTraj("unified", data.trajectory);
  let extra = "<br>" + fmtMetrics(data.metrics, data.goal_reach).replace("\n", "<br>") +
    `<br>artifacts: ${data.artifact_dir}`;
  if (data.scores) {
    extra += `<br>scores: mean ${data.scores.mean.toFixed(3)} · per round ` +
      Object.entries(data.scores.per_round).map(([k, v]) => `#${k}=${v.toFixed(3)}`).join(" ");
  }
  renderRounds();
  renderUnifiedOutput(extra);
  renderAll();
}

async function resetRounds() {
  await api("/api/reset_rounds", {}, "resetting rounds");
  rounds = [];
  unified = null;
  unifiedCostField = null;
  clearTraj("unified");
  $("commit-round").disabled = true;
  $("apply-round").disabled = true;
  renderRounds();
  renderUnifiedOutput();
  refreshLegend(); refreshTimeline(); renderAll();
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
  const selected = clusters.find((c) => c.label === selectedCluster);
  const nClusters = +$("n-clusters").value;
  $("cluster-refine").disabled = !selected || nClusters < 2 ||
    selected.count < nClusters;
  $("cluster-back").disabled = !canGoBack;
  const path = clusterPath.length
    ? "root > " + clusterPath.map((label) => `cluster ${label}`).join(" > ")
    : "root";
  const sampleCount = clusters.reduce((total, c) => total + c.count, 0);
  $("cluster-navigation").textContent = clusters.length
    ? `${path} · level ${clusterDepth} · ${sampleCount} samples`
    : "";
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
    mk("selected cluster body + trace", "#888", showClusters, (v) => { showClusters = v; });
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
      const maxFrame = +slider.max;
      if (frame >= maxFrame) {
        playing = false;
        $("play").innerHTML = "&#9654;";
      } else {
        frame = Math.min(maxFrame, +slider.value + 1);
        slider.value = frame;
        $("frame-label").textContent = `frame ${frame} / ${slider.max}`;
        renderAll();
      }
    }
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// SMPL mesh rendering
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

let smplRenderer = null;
let smplScene = null;

function disposeScene(scene) {
  if (!scene) return;
  scene.traverse((object) => {
    if (object.geometry) object.geometry.dispose();
    if (object.material) {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) material.dispose();
    }
  });
}

function requestMesh(data) {
  const id = data && data.mesh_id;
  if (!id || meshData.has(id) || meshLoads.has(id)) return;
  const load = fetch(`/api/mesh/${encodeURIComponent(id)}`).then(async (response) => {
    if (!response.ok) throw new Error((await response.json()).error);
    meshData.set(id, {
      frames: +response.headers.get("X-Mesh-Frames"),
      vertices: +response.headers.get("X-Mesh-Vertices"),
      data: new Float32Array(await response.arrayBuffer()),
    });
    meshLoads.delete(id);
    renderAll();
  }).catch((error) => {
    meshLoads.delete(id);
    setStatus(`mesh load failed: ${error}`, "error");
  });
  meshLoads.set(id, load);
}

function lineObject(points, color, opacity = 0.7) {
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map((p) => new THREE.Vector3(...p))
  );
  return new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity })
  );
}

function meshObject(vertices, color, opacity) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(INIT.smpl_faces.flat());
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({
    color, transparent: opacity < 1, opacity, side: THREE.FrontSide, depthWrite: true,
  }));
}

function trajectoryMesh(data, color, opacity = 0.32, atFrame = frame) {
  requestMesh(data);
  const cached = meshData.get(data.mesh_id);
  if (!cached) return null;
  const fi = Math.min(atFrame, cached.frames - 1);
  const start = fi * cached.vertices * 3;
  return meshObject(cached.data.subarray(start, start + cached.vertices * 3), color, opacity);
}

function renderSkeleton() {
  const canvas = $("skeleton-canvas");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height || !INIT) return;
  if (!smplRenderer) smplRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  smplRenderer.setPixelRatio(window.devicePixelRatio || 1);
  smplRenderer.setSize(rect.width, rect.height, false);
  smplRenderer.setScissorTest(true);
  smplRenderer.setClearColor(0xffffff, 1);

  disposeScene(smplScene);
  const scene = new THREE.Scene();
  smplScene = scene;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x667788, 2.2));
  const light = new THREE.DirectionalLight(0xffffff, 1.5);
  light.position.set(2, 3, 4); scene.add(light);

  const selected = showClusters
    ? clusters.find((c) => c.label === selectedCluster)
    : null;
  const pts = INIT.body_pos.slice();
  pts.push(goalWorld());
  for (const [, t] of visibleTrajs()) for (const f of t.data.arm_positions) pts.push(f[WRIST]);
  if (selected) for (const f of selected.full.arm_positions) pts.push(f[WRIST]);
  if (startPreview) for (const p of startPreview.arm_positions) pts.push(p);
  const framingVertices = startPreview
    ? startPreview.mesh_vertices
    : INIT.smpl_reference_vertices;
  const min = [0, 1, 2].map((axis) => Math.min(
    ...pts.map((p) => p[axis]), ...framingVertices.map((p) => p[axis]),
  ) - 0.14);
  const max = [0, 1, 2].map((axis) => Math.max(
    ...pts.map((p) => p[axis]), ...framingVertices.map((p) => p[axis]),
  ) + 0.14);
  const center = min.map((v, i) => (v + max[i]) / 2);

  if (selected) {
    const color = CLUSTER_COLORS[selected.label % CLUSTER_COLORS.length];
    scene.add(lineObject(selected.full.arm_positions.map((f) => f[WRIST]), color, 0.4));
    const mesh = trajectoryMesh(selected.full, color, 0.26);
    if (mesh) scene.add(mesh);
  }
  for (const [key, t] of visibleTrajs()) {
    const mesh = trajectoryMesh(t.data, t.color);
    if (mesh) scene.add(mesh);
    const viol = computeViolations(t.data);
    for (let i = 1; i < t.data.n_frames; i++) {
      scene.add(lineObject(
        [t.data.arm_positions[i - 1][WRIST], t.data.arm_positions[i][WRIST]],
        viol[i] > 0 ? "#ff2222" : t.color,
        viol[i] > 0 ? 0.95 : 0.45,
      ));
    }
    if (key === "base" && baseTrigger !== null) {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0x111111, wireframe: true }),
      );
      marker.position.set(...t.data.arm_positions[baseTrigger][WRIST]); scene.add(marker);
    }
  }
  if (startPreview && showStart) {
    scene.add(meshObject(new Float32Array(startPreview.mesh_vertices.flat()), START_COLOR, 0.72));
  }
  if (showMdmStart && trajs.base && trajs.base.data) {
    const fi = baseTrigger === null ? 0 : baseTrigger;
    const mesh = trajectoryMesh(trajs.base.data, MDM_START_COLOR, 0.16, fi);
    if (mesh) scene.add(mesh);
  }
  const goal = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.025),
    new THREE.MeshBasicMaterial({ color: 0xc92a9b }),
  );
  goal.position.set(...goalWorld()); scene.add(goal);

  const vw = rect.width / 3;
  VIEWS.forEach((view, vi) => {
    const horizontal = max[view.hi] - min[view.hi];
    const vertical = max[view.vi] - min[view.vi];
    const halfH = Math.max(vertical / 2, horizontal * rect.height / (2 * vw));
    const halfW = halfH * vw / rect.height;
    const camera = new THREE.OrthographicCamera(-halfW, halfW, halfH, -halfH, 0.01, 20);
    if (vi === 0) { camera.position.set(center[0], center[1], center[2] + 5); camera.up.set(0, 1, 0); }
    else if (vi === 1) { camera.position.set(center[0] + 5, center[1], center[2]); camera.up.set(0, 1, 0); }
    else { camera.position.set(center[0], center[1] + 5, center[2]); camera.up.set(0, 0, -1); }
    camera.lookAt(...center);
    const x = Math.floor(vi * vw), width = Math.ceil(vw), height = Math.floor(rect.height);
    smplRenderer.setViewport(x, 0, width, height);
    smplRenderer.setScissor(x, 0, width, height);
    smplRenderer.render(scene, camera);
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
  addToggle("unified-cost limits", UNIFIED_COLOR, showUnifiedBounds,
    (v) => { showUnifiedBounds = v; });
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
    canvas.addEventListener("mousedown", (e) => startFeatureDrag(name, canvas, e));
    canvas.addEventListener("mousemove", (e) => hoverFeature(name, canvas, e));
    const bctl = document.createElement("div");
    bctl.className = "gbound-controls";
    bctl.id = `gbounds-${name}`;
    panel.appendChild(title);
    panel.appendChild(canvas);
    panel.appendChild(bctl);
    holder.appendChild(panel);
  }
}

function personaConstantBounds(feature) {
  const p = getPersona();
  if (!p) return [];
  return p.bounds.filter((b) => b.kind === "hidden" && b.feature === feature);
}

// ---------------------------------------------------------------------------
// On-graph persona bound editing
// ---------------------------------------------------------------------------
//
// The graphs read bounds live from the local persona object, so editing is:
// mutate the bound, re-render (violations recompute client-side), and save
// the persona back to the server debounced. The server re-detects the
// feedback trigger on the stored base rollout after each save.

const graphTX = {};    // feature name -> plot transform from the last render
const coupledTX = [];  // coupled plot index -> transform
let boundDrag = null;

let personaSaveTimer = null;
function schedulePersonaSave() {
  clearTimeout(personaSaveTimer);
  personaSaveTimer = setTimeout(async () => {
    const p = getPersona();
    const data = await api("/api/personas", p, "saving persona bounds");
    if (data.retriggered) {
      baseTrigger = data.trigger_step;
      applyTriggerHint(data.trigger_step);
      renderAll();
    }
    if (p.builtin) setStatus("built-in persona edited in memory only (until restart)");
  }, 600);
}

function applyTriggerHint(step) {
  $("mdm-start-hint").textContent = step === null
    ? "MDM will generate from the start pose (base never violates)."
    : `MDM will generate from the feedback-trigger pose at frame ${step} (purple ghost body).`;
}

function retypeHiddenBound(b, newType, feature) {
  const tx = graphTX[feature];
  const def = tx ? tx.ymin + 0.6 * (tx.ymax - tx.ymin) : 1.0;
  b.bound_type = newType;
  if (newType === "upper_bound") {
    b.high = b.high ?? b.low ?? +def.toFixed(2);
    b.low = null;
  } else if (newType === "lower_bound") {
    b.low = b.low ?? b.high ?? +def.toFixed(2);
    b.high = null;
  } else {
    if (b.high === null || b.high === undefined) b.high = (b.low ?? def) + 0.4;
    if (b.low === null || b.low === undefined) b.low = b.high - 0.4;
  }
}

function renderBoundControls() {
  const p = getPersona();
  for (const name of INIT.feature_names) {
    const holder = $(`gbounds-${name}`);
    if (!holder) continue;
    holder.innerHTML = "";
    if (!p) continue;
    for (const b of personaConstantBounds(name)) {
      const idx = p.bounds.indexOf(b);
      const row = document.createElement("div");
      row.className = "gbound-row";
      const type = document.createElement("select");
      for (const t of ["upper_bound", "lower_bound", "avoid_band"]) {
        const opt = document.createElement("option");
        opt.value = t; opt.textContent = t;
        type.appendChild(opt);
      }
      type.value = b.bound_type;
      type.onchange = () => {
        retypeHiddenBound(b, type.value, name);
        renderBoundControls(); renderAll(); schedulePersonaSave();
      };
      row.appendChild(type);
      for (const key of ["low", "high"]) {
        if (b[key] === null || b[key] === undefined) continue;
        const n = document.createElement("input");
        n.type = "number"; n.step = 0.05;
        n.value = (+b[key]).toFixed(2);
        n.id = `gb-${name}-${idx}-${key}`;
        n.onchange = () => { b[key] = +n.value; renderAll(); schedulePersonaSave(); };
        row.appendChild(n);
      }
      const del = document.createElement("button");
      del.textContent = "✕";
      del.onclick = () => {
        p.bounds.splice(p.bounds.indexOf(b), 1);
        renderBoundControls(); renderAll(); schedulePersonaSave();
      };
      row.appendChild(del);
      holder.appendChild(row);
    }
    const add = document.createElement("button");
    add.textContent = "+ bound";
    add.onclick = () => {
      const tx = graphTX[name];
      const v = tx ? tx.ymin + 0.75 * (tx.ymax - tx.ymin) : 1.0;
      p.bounds.push({
        kind: "hidden", feature: name, bound_type: "upper_bound",
        low: null, high: +v.toFixed(2),
      });
      renderBoundControls(); renderAll(); schedulePersonaSave();
    };
    holder.appendChild(add);
  }
}

function featureHit(name, canvas, e) {
  const tx = graphTX[name];
  if (!tx || !showOracleBounds) return null;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (mx < tx.ml || mx > tx.ml + tx.pw) return null;
  const Y = (v) => tx.mt + (1 - (v - tx.ymin) / (tx.ymax - tx.ymin)) * tx.ph;
  for (const b of personaConstantBounds(name)) {
    for (const key of ["low", "high"]) {
      if (b[key] === null || b[key] === undefined) continue;
      if (Math.abs(my - Y(b[key])) <= 6) return { b, key };
    }
  }
  return null;
}

function hoverFeature(name, canvas, e) {
  if (boundDrag) return;
  canvas.style.cursor = featureHit(name, canvas, e) ? "ns-resize" : "";
}

function startFeatureDrag(name, canvas, e) {
  const hit = featureHit(name, canvas, e);
  if (!hit) return;
  const tx = graphTX[name];
  boundDrag = {
    kind: "feature", feature: name, canvas, b: hit.b, key: hit.key,
    idx: getPersona().bounds.indexOf(hit.b),
    frozen: { ymin: tx.ymin, ymax: tx.ymax },
  };
  e.preventDefault();
}

function dragFeature(e) {
  const { canvas, feature, b, key, idx, frozen } = boundDrag;
  const tx = graphTX[feature];
  const rect = canvas.getBoundingClientRect();
  const frac = 1 - (e.clientY - rect.top - tx.mt) / tx.ph;
  let v = frozen.ymin + frac * (frozen.ymax - frozen.ymin);
  if (b.bound_type === "avoid_band") {
    if (key === "low") v = Math.min(v, b.high - 0.01);
    else v = Math.max(v, b.low + 0.01);
  }
  b[key] = +v.toFixed(3);
  const input = $(`gb-${feature}-${idx}-${key}`);
  if (input) input.value = b[key].toFixed(2);
  renderAll();
}

function coupledHit(gi, canvas, e) {
  const tx = coupledTX[gi];
  if (!tx) return null;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const b = currentCoupledBounds()[gi];
  if (!b) return null;
  const X = (v) => tx.ml + ((v - tx.xmin) / (tx.xmax - tx.xmin)) * tx.pw;
  const Y = (v) => tx.mt + (1 - (v - tx.ymin) / (tx.ymax - tx.ymin)) * tx.ph;
  for (const [h, hx] of [[0, tx.hx0], [1, tx.hx1]]) {
    const hy = b.intercept + b.slope * hx;
    if (Math.hypot(mx - X(hx), my - Y(hy)) <= 9) return { b, handle: h };
  }
  return null;
}

function hoverCoupled(gi, canvas, e) {
  if (boundDrag) return;
  canvas.style.cursor = coupledHit(gi, canvas, e) ? "grab" : "";
}

function startCoupledDrag(gi, canvas, e) {
  const hit = coupledHit(gi, canvas, e);
  if (!hit) return;
  const tx = coupledTX[gi];
  const hx = hit.handle === 0 ? tx.hx0 : tx.hx1;
  const ox = hit.handle === 0 ? tx.hx1 : tx.hx0;
  boundDrag = {
    kind: "coupled", index: gi, canvas, b: hit.b,
    hx, otherX: ox, otherY: hit.b.intercept + hit.b.slope * ox,
    frozen: { xmin: tx.xmin, xmax: tx.xmax, ymin: tx.ymin, ymax: tx.ymax },
  };
  e.preventDefault();
}

function dragCoupled(e) {
  const { canvas, index, b, hx, otherX, otherY, frozen } = boundDrag;
  const tx = coupledTX[index];
  const rect = canvas.getBoundingClientRect();
  const frac = 1 - (e.clientY - rect.top - tx.mt) / tx.ph;
  const y = frozen.ymin + frac * (frozen.ymax - frozen.ymin);
  const slope = (y - otherY) / (hx - otherX);
  b.slope = +slope.toFixed(3);
  b.intercept = +(y - slope * hx).toFixed(3);
  for (const key of ["intercept", "slope"]) {
    const input = $(`cb-${index}-${key}`);
    if (input) input.value = b[key].toFixed(2);
  }
  renderAll();
}

window.addEventListener("mousemove", (e) => {
  if (!boundDrag) return;
  if (boundDrag.kind === "feature") dragFeature(e); else dragCoupled(e);
});
window.addEventListener("mouseup", () => {
  if (!boundDrag) return;
  boundDrag = null;
  renderBoundControls();
  renderAll();
  schedulePersonaSave();
});

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
      seriesList.push({ s, color: t.color, data: t.data });
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
      ? generatedBounds.filter((b) => b.feature === name && !b.coupled)
      : [];
    for (const b of [...oracleBounds, ...costBounds]) {
      if (b.low !== null && b.low !== undefined) { ymin = Math.min(ymin, b.low); ymax = Math.max(ymax, b.low); }
      if (b.high !== null && b.high !== undefined) { ymin = Math.min(ymin, b.high); ymax = Math.max(ymax, b.high); }
    }
    if (!seriesList.length) { ymin = Math.min(ymin, -0.2); ymax = Math.max(ymax, Math.PI); }
    if (!isFinite(ymin)) { ymin = 0; ymax = Math.PI; }
    const span = Math.max(0.3, ymax - ymin);
    ymin -= span * 0.15; ymax += span * 0.15;
    // Freeze the scale while a bound is being dragged on this graph, so the
    // value under the cursor doesn't shift as the range refits around it.
    if (boundDrag && boundDrag.kind === "feature" && boundDrag.feature === name) {
      ({ ymin, ymax } = boundDrag.frozen);
    }
    graphTX[name] = { ml, mt, pw, ph, ymin, ymax };

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

    // drag handles on the persona bound edges
    ctx.fillStyle = "rgba(180,40,40,0.9)";
    for (const b of oracleBounds) {
      for (const key of ["low", "high"]) {
        if (b[key] === null || b[key] === undefined) continue;
        ctx.fillRect(ml + pw - 26, Y(b[key]) - 3, 22, 6);
      }
    }

    // coupled per-frame thresholds (dashed, per trajectory)
    for (const { color, data } of showOracleBounds ? seriesList : []) {
      if (!data) continue;
      for (const b of currentCoupledBounds()) {
        if (b.feature !== name) continue;
        const thr = coupledThresholdSeries(b, data);
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

function featureBoxRange(name) {
  const p = getPersona();
  const range = p && p.feature_box_ranges && p.feature_box_ranges[name];
  if (!range || range.length !== 2 || !isFinite(range[0]) || !isFinite(range[1])) return null;
  return range;
}

function buildCoupledGraphs() {
  const holder = $("coupled-graphs");
  if (!holder) return;
  holder.innerHTML = "";
  const p = getPersona();
  currentCoupledBounds().forEach((b, i) => {
    const panel = document.createElement("div");
    panel.className = "panel";
    const title = document.createElement("div");
    title.className = "graph-title";
    title.textContent = `${b.feature} vs ${b.cond_feature} (pose-dependent limit)`;
    const del = document.createElement("button");
    del.textContent = "✕";
    del.className = "coupled-del";
    del.onclick = () => {
      p.bounds.splice(p.bounds.indexOf(b), 1);
      buildCoupledGraphs(); renderAll(); schedulePersonaSave();
    };
    title.appendChild(del);
    const canvas = document.createElement("canvas");
    canvas.className = "coupled-graph";
    canvas.id = `coupled-graph-${i}`;
    canvas.addEventListener("mousedown", (e) => startCoupledDrag(i, canvas, e));
    canvas.addEventListener("mousemove", (e) => hoverCoupled(i, canvas, e));
    const row = document.createElement("div");
    row.className = "gbound-row";
    const type = document.createElement("select");
    for (const t of ["upper_bound", "lower_bound"]) {
      const opt = document.createElement("option");
      opt.value = t; opt.textContent = t;
      type.appendChild(opt);
    }
    type.value = b.bound_type;
    type.onchange = () => { b.bound_type = type.value; renderAll(); schedulePersonaSave(); };
    row.appendChild(type);
    for (const key of ["intercept", "slope"]) {
      const lab = document.createElement("span");
      lab.textContent = key;
      row.appendChild(lab);
      const n = document.createElement("input");
      n.type = "number"; n.step = 0.05;
      n.value = (+b[key]).toFixed(2);
      n.id = `cb-${i}-${key}`;
      n.onchange = () => { b[key] = +n.value; renderAll(); schedulePersonaSave(); };
      row.appendChild(n);
    }
    panel.appendChild(title);
    panel.appendChild(canvas);
    panel.appendChild(row);
    holder.appendChild(panel);
  });
  if (!p) return;
  const addRow = document.createElement("div");
  addRow.className = "panel gbound-row coupled-add";
  const mkFeat = () => {
    const s = document.createElement("select");
    for (const f of INIT.feature_names) {
      const opt = document.createElement("option");
      opt.value = f; opt.textContent = f;
      s.appendChild(opt);
    }
    return s;
  };
  const fSel = mkFeat(), cSel = mkFeat();
  cSel.value = INIT.feature_names[1] || INIT.feature_names[0];
  const vs = document.createElement("span");
  vs.textContent = "vs";
  const add = document.createElement("button");
  add.textContent = "+ coupled bound";
  add.onclick = () => {
    let mid = 1.0;
    const t = visibleTrajs()[0];
    if (t) {
      const f = t[1].data.features[fSel.value];
      mid = f.reduce((a, v) => a + v, 0) / f.length;
    }
    p.bounds.push({
      kind: "coupled", feature: fSel.value, bound_type: "upper_bound",
      cond_feature: cSel.value, intercept: +mid.toFixed(2), slope: 0,
    });
    buildCoupledGraphs(); renderAll(); schedulePersonaSave();
  };
  addRow.appendChild(fSel);
  addRow.appendChild(vs);
  addRow.appendChild(cSel);
  addRow.appendChild(add);
  holder.appendChild(addRow);
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
    const xBox = featureBoxRange(b.cond_feature);
    const yBox = featureBoxRange(b.feature);
    if (xBox) { xmin = Math.min(xmin, xBox[0]); xmax = Math.max(xmax, xBox[1]); }
    if (yBox) { ymin = Math.min(ymin, yBox[0]); ymax = Math.max(ymax, yBox[1]); }
    if (!isFinite(xmin)) { xmin = -0.2; xmax = 2.6; ymin = -0.2; ymax = 2.6; }
    const xs = Math.max(0.3, xmax - xmin), ys = Math.max(0.3, ymax - ymin);
    xmin -= xs * 0.1; xmax += xs * 0.1; ymin -= ys * 0.1; ymax += ys * 0.1;
    // Freeze the scale while a handle is being dragged on this plot.
    if (boundDrag && boundDrag.kind === "coupled" && boundDrag.index === gi) {
      ({ xmin, xmax, ymin, ymax } = boundDrag.frozen);
    }
    coupledTX[gi] = {
      ml, mt, pw, ph, xmin, xmax, ymin, ymax,
      hx0: xmin + 0.15 * (xmax - xmin),
      hx1: xmax - 0.15 * (xmax - xmin),
    };

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

    // drag handles for editing the limit line
    ctx.fillStyle = "rgba(180,40,40,0.95)";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    for (const hx of [coupledTX[gi].hx0, coupledTX[gi].hx1]) {
      ctx.beginPath(); ctx.arc(X(hx), Y(thr(hx)), 6, 0, 7); ctx.fill(); ctx.stroke();
    }

    // generated/unified-cost penalty fields: the ACTUAL compiled cost evaluated
    // over a cloud of plausible poses, read straight from the cost — no
    // declared-bound parsing — so it works for any cost shape/backend and its
    // penalized region can be compared directly against the red oracle limit.
    const drawField = (field, rgb) => {
      if (!field || !field.penalty.length) return;
      const cond = field.features[b.cond_feature];
      const feat = field.features[b.feature];
      const pen = field.penalty;
      let pmax = 0;
      for (const p of pen) if (p > pmax) pmax = p;
      if (!cond || !feat || pmax <= 0) return;
      const eps = pmax * 0.02;
      for (let i = 0; i < pen.length; i++) {
        if (pen[i] <= eps) continue;
        ctx.fillStyle = `rgba(${rgb},${0.12 + 0.5 * Math.min(1, pen[i] / pmax)})`;
        ctx.beginPath(); ctx.arc(X(cond[i]), Y(feat[i]), 3, 0, 7); ctx.fill();
      }
    };
    if (showGeneratedBounds) drawField(costField, "53,103,214");
    if (showUnifiedBounds) drawField(unifiedCostField, "0,131,143");

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
    const viol = computeViolations(t.data);
    for (let i = 0; i < t.data.n_frames; i++) {
      ctx.fillStyle = viol[i] > 0 ? "#e02020" : t.color + "44";
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

  if (INIT.manual_trajectory) {
    const data = INIT.manual_trajectory;
    multiTurnActive = true;
    setTraj("base", data.trajectory);
    baseTrigger = data.trigger ? data.trigger.step : null;
    renderTrajectorySession(data);
    setScenarioLocked(data.status === "paused");
    $("run-base").disabled = data.status === "paused";
    $("exit-trajectory").disabled = false;
    $("generate").disabled = data.status !== "paused";
    $("ignore-violation").disabled = data.status !== "paused" ||
      data.trigger.reason !== "discomfort";
    $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) +
      `\nexecuted ${data.step}/${data.step_limit} steps`;
  }
  if (INIT.pending_cost) {
    renderCostGeneration(INIT.pending_cost);
    $("commit-round").disabled = multiTurnActive;
    $("apply-round").disabled = !multiTurnActive;
  }

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
  $("exit-trajectory").onclick = exitManualTrajectory;
  $("ignore-violation").onclick = ignoreComfortViolation;
  $("generate").onclick = generate;
  $("recluster").onclick = recluster;
  $("cluster-refine").onclick = refineCluster;
  $("cluster-back").onclick = backCluster;
  $("generate-cost").onclick = generateCost;
  $("n-clusters").onchange = renderClusterList;

  rounds = INIT.rounds || [];
  unified = INIT.unified || null;
  $("commit-round").onclick = commitRound;
  $("apply-round").onclick = applyRoundAndContinue;
  $("combine-rounds").onclick = combineRounds;
  $("reset-rounds").onclick = resetRounds;
  renderRounds();
  renderUnifiedOutput();

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
    const slider = $("frame-slider");
    if (!playing && frame >= +slider.max) {
      slider.value = 0;
      slider.oninput();
    }
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
