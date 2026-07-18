import * as THREE from "three";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let INIT = null;
let personas = [];
let currentPersona = null;
let session = null;
let trajectoryConfigs = { initial_poses: [], goals: [] };

// key -> {data: packaged trajectory, color, label, visible}
const trajs = {};
let clusters = [];          // [{label, count, oracle_score, trajectory}]
let selectedCluster = null;
let undesirableClusters = new Set();
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
let costField = null;       // {features:{name:[..]}, penalty:[..], active_features:[..]} — compiled cost sampled over a pose cloud
let startPreview = null;    // {arm_positions, mesh_vertices} for the edited pose
const meshData = new Map(); // mesh id -> {frames, vertices, data: Float32Array}
const meshLoads = new Map();
let liveBaseMeshIds = [];
let baseTrigger = null;
let multiTurnActive = false;

// Both modes share the guided workflow. Demo mode hides setup and hidden-cost
// machinery; dev mode reveals it inside the same stages. Replay steps through a
// recorded session's beats and is orthogonal to the mode: demo+replay is the
// presentation, dev+replay reads the same beats with the graphs and code visible.
let demoMode = localStorage.getItem("demo_runner_mode") !== "dev";
let costLogActive = false;
let combineLogActive = false;
let logCharIndex = 0;
let logPoll = null;
let workflowStage = 1;
let displayedStage = 1;
let replayActive = false;
let replayName = null;
let replayBeats = [];
let replayIndex = 0;
const replayCache = new Map(); // beat index -> served beat (keeps mesh ids stable)
// Set while re-applying intermediate beats during a rebuild: the state math runs
// but the (expensive) draw is skipped until the destination beat.
let suppressRender = false;

// The persona is read-only unless it is both editable in this mode and live.
function boundsEditable() {
  return !demoMode && !replayActive;
}

function resetClusterNavigation() {
  clusterDepth = 0;
  clusterPath = [];
  canGoBack = false;
}

// Multi-round state: committed rounds + the unified replacement cost. The
// unified visuals survive base rollouts (only Reset rounds clears them).
let rounds = [];
let unified = null;         // {description, code} or null
let pendingCost = null;
let unifiedCostField = null;
let showUnifiedBounds = true;

const MDM_START_COLOR = "#7b2fbe";
const GENERATED_LIMIT_COLOR = "#3567d6";
const UNIFIED_COLOR = "#00838f";

let frame = 0;
let playing = false;
let lastTick = 0;

const TRAJ_STYLES = {
  clean_base: { color: "#6b7280", label: "baseline (box limits, no feedback)" },
  base: { color: "#e05252", label: "rollout" },
  oracle: { color: "#9c5f17", label: "oracle-cost rollout" },
  full: { color: "#0e7a63", label: "full corrected path" },
  generated: { color: "#3567d6", label: "generated-cost corrected path" },
  generated_start: { color: "#6fa8ff", label: "generated-cost from start" },
  unified: { color: UNIFIED_COLOR, label: "unified-cost rollout" },
};
const CLUSTER_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#9a6324", "#800000", "#469990"];
const START_COLOR = "#e8a33d";
const LIVE_FRAME_INTERVAL_MS = 150;
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

async function pollLogs() {
  if (logPoll) return logPoll;
  logPoll = (async () => {
    try {
      const resp = await fetch(`/api/logs?char_since=${logCharIndex}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.text) {
        const consoleEl = $("console");
        const consoleAtBottom =
          consoleEl.scrollTop + consoleEl.clientHeight >= consoleEl.scrollHeight - 24;
        consoleEl.textContent += data.text;
        if (consoleEl.textContent.length > 200000) {
          consoleEl.textContent = consoleEl.textContent.slice(-200000);
        }
        if (consoleAtBottom) consoleEl.scrollTop = consoleEl.scrollHeight;

        if (costLogActive) {
          const thoughtLog = $("cost-thought-log");
          thoughtLog.textContent += data.text;
          if (thoughtLog.textContent.length > 200000) {
            thoughtLog.textContent = thoughtLog.textContent.slice(-200000);
          }
          thoughtLog.scrollTop = thoughtLog.scrollHeight;
        }
        if (combineLogActive) {
          const combineLog = $("combine-log");
          if (combineLog) {
            combineLog.textContent += data.text;
            if (combineLog.textContent.length > 200000) {
              combineLog.textContent = combineLog.textContent.slice(-200000);
            }
            combineLog.scrollTop = combineLog.scrollHeight;
          }
        }
      }
      logCharIndex = data.next_char;
    } catch (err) { /* server busy or down; retry next poll */ }
  })();
  try {
    await logPoll;
  } finally {
    logPoll = null;
  }
}

function clearCostThoughtLog() {
  costLogActive = false;
  $("cost-thought-log-wrap").hidden = true;
  $("cost-thought-log").textContent = "";
}

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

// Bounds parsed from the pending cost plus every committed round, so committed
// limits stay on the graphs after the pipeline moves past cost generation.
function activeGeneratedBounds() {
  const out = [...generatedBounds];
  for (const r of rounds) out.push(...(r.generated_bounds || []));
  return out;
}

function generatedCoupledBoundsFor(feature, condFeature) {
  return activeGeneratedBounds().filter((candidate) => candidate.coupled &&
    candidate.feature === feature && candidate.cond_feature === condFeature);
}

function generatedCoupledThreshold(b, x) {
  const points = b.points;
  if (x <= points[0][0]) return points[0][1];
  if (x >= points[points.length - 1][0]) return points[points.length - 1][1];
  for (let i = 1; i < points.length; i++) {
    if (x <= points[i][0]) {
      const [x0, y0] = points[i - 1];
      const [x1, y1] = points[i];
      return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
    }
  }
  return points[points.length - 1][1];
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
  trajs[key] = { data, visible: key === "base", ...TRAJ_STYLES[key] };
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
// Named initial-pose and goal configs
// ---------------------------------------------------------------------------

function refreshTrajectoryConfigSelects() {
  const populate = (id, configs, placeholder) => {
    const select = $(id);
    const selected = select.value;
    select.innerHTML = `<option value="">${placeholder}</option>`;
    for (const config of configs) {
      const option = document.createElement("option");
      option.value = config.name;
      option.textContent = config.name;
      select.appendChild(option);
    }
    select.value = configs.some((config) => config.name === selected) ? selected : "";
  };
  populate("initial-pose-config", trajectoryConfigs.initial_poses, "Configured default / current");
  populate("goal-config", trajectoryConfigs.goals, "Persona default / current");
}

function selectInitialPoseConfig(name) {
  if (!name) {
    $("initial-pose-config-name").value = "";
    setArmEditorValues(INIT.start_arm_aa);
    return;
  }
  const config = trajectoryConfigs.initial_poses.find((item) => item.name === name);
  $("initial-pose-config-name").value = config.name;
  setArmEditorValues(config.arm_aa);
}

function selectGoalConfig(name) {
  if (!name) {
    $("goal-config-name").value = "";
    const goals = INIT.persona_goals[currentPersona];
    const goal = goals && goals.cartesian.length ? goals.cartesian[0] : INIT.default_goal;
    [$("goal-x").value, $("goal-y").value, $("goal-z").value] = goal.map((v) => v.toFixed(2));
  } else {
    const config = trajectoryConfigs.goals.find((item) => item.name === name);
    $("goal-config-name").value = config.name;
    [$("goal-x").value, $("goal-y").value, $("goal-z").value] =
      config.goal.map((v) => Number(v).toFixed(3));
  }
  renderAll();
}

async function saveTrajectoryConfig(kind) {
  const isPose = kind === "initial_poses";
  const nameInput = $(isPose ? "initial-pose-config-name" : "goal-config-name");
  const name = nameInput.value.trim();
  if (!name) {
    setStatus("config name required", "error");
    return;
  }
  const payload = isPose ? { name, arm_aa: armAA } : { name, goal: getGoal() };
  const data = await api(
    `/api/trajectory-configs/${kind}`,
    payload,
    `saving ${isPose ? "initial pose" : "goal"}`,
  );
  trajectoryConfigs = data.trajectory_configs;
  refreshTrajectoryConfigSelects();
  $(isPose ? "initial-pose-config" : "goal-config").value = name;
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
  const defaultPrompt = INIT.default_prompts[name];
  if (defaultPrompt) $("prompt").value = defaultPrompt;
  else if (p.feedback_text) $("prompt").value = p.feedback_text;
  const goals = INIT.persona_goals[name];
  const goal = goals && goals.cartesian.length ? goals.cartesian[0] : INIT.default_goal;
  [$("goal-x").value, $("goal-y").value, $("goal-z").value] = goal.map((v) => v.toFixed(2));
  $("goal-config").value = "";
  $("goal-config-name").value = "";
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
  currentPersona = session ? session.persona : name;
  refreshPersonaSelect();
  if (data.retriggered) {
    baseTrigger = data.trigger_step;
    applyTriggerHint(data.trigger_step);
  }
  onPersonaChange(currentPersona);
  $("modal-backdrop").classList.remove("open");
  renderSession();
  if (editingBuiltin) setStatus("built-in persona edited in memory only (until restart)");
}

// ---------------------------------------------------------------------------
// Pipeline actions
// ---------------------------------------------------------------------------

function setScenarioLocked(locked) {
  for (const id of ["goal-x", "goal-y", "goal-z", "reset-pose",
    "initial-pose-config", "initial-pose-config-name", "save-initial-pose-config",
    "goal-config", "goal-config-name", "save-goal-config"]) {
    $(id).disabled = locked;
  }
  for (const input of document.querySelectorAll("#arm-editor input")) input.disabled = locked;
  $("persona-select").disabled = Boolean(session);
}

function setDecisionControls(data) {
  const paused = data.status === "paused";
  $("enter-correction").disabled = !paused;
  const ignoreDisabled = !paused || data.trigger?.reason !== "discomfort";
  $("ignore-violation").disabled = ignoreDisabled;
}

function enterCorrection() {
  $("enter-correction").disabled = true;
  $("ignore-violation").disabled = true;
  setWorkflowStage(2);
}

function renderCorpus() {
  const list = $("corpus-list");
  list.innerHTML = "";
  if (!session) {
    list.innerHTML = '<div class="hint">Start or resume a session to view its corpus.</div>';
    return;
  }
  if (!session.corpus.length) {
    list.innerHTML = '<div class="hint">No executed trajectory segments yet.</div>';
    return;
  }
  for (const entry of session.corpus) {
    const div = document.createElement("div");
    div.className = "corpus-card";
    const goal = entry.goal.map((v) => Number(v).toFixed(2)).join(", ");
    const trigger = entry.trigger_step === null
      ? "no trigger"
      : `trigger @ ${entry.trigger_step} (${Number(entry.trigger_violation).toFixed(3)})`;
    const details = document.createElement("div");
    details.textContent = `#${entry.index} · ${entry.kind} · round ${entry.round + 1} · ` +
      `goal [${goal}] · comfortable 0–${entry.comfortable_until} of ${entry.n_frames} frames · ${trigger}`;
    const remove = document.createElement("button");
    remove.className = "danger";
    remove.textContent = "Delete entry";
    remove.onclick = () => deleteCorpusEntry(entry.index);
    div.appendChild(details);
    div.appendChild(remove);
    list.appendChild(div);
  }
}

function renderSession() {
  const summary = $("session-summary");
  if (session) {
    summary.innerHTML = `<span><b>${escapeHtml(session.persona)}</b> · ${escapeHtml(session.started)}</span>` +
      `<span>${session.trajectory_count} trajectories · ${session.corpus.length} corpus entries · ` +
      `${rounds.length} rounds · unified: ${unified ? "yes" : "no"}</span>`;
  } else {
    summary.textContent = "No active session";
  }
  $("persona-select").disabled = Boolean(session);
  const paused = Boolean(session && session.trajectory && session.trajectory.status === "paused");
  $("run-base").disabled = !session || paused;
  $("run-base").title = session ? "" : "Start or resume a session first";
  renderCorpus();
}

function closeSessionDropdown() {
  $("session-dropdown").classList.remove("open");
  $("session-summary").setAttribute("aria-expanded", "false");
}

function renderSessionPicker(sessions) {
  const picker = $("session-picker");
  picker.innerHTML = "";
  if (!sessions.length) {
    picker.innerHTML = '<div class="hint">No saved sessions found.</div>';
    return;
  }
  for (const item of sessions) {
    const card = document.createElement("div");
    card.className = "session-picker-card";
    const details = document.createElement("div");
    details.textContent = `${item.persona} · ${item.started} · ${item.trajectory_count} trajectories · ` +
      `${item.corpus_count} corpus entries · ${item.round_count} rounds`;
    const actions = document.createElement("div");
    actions.className = "session-picker-actions";
    const resume = document.createElement("button");
    resume.className = "primary";
    resume.textContent = "Resume";
    resume.onclick = () => resumeSession(item.dir);
    const replay = document.createElement("button");
    replay.textContent = "Replay";
    replay.onclick = () => loadReplay(item.name);
    const remove = document.createElement("button");
    remove.className = "danger";
    remove.textContent = "Delete";
    remove.onclick = () => deleteSession(item);
    actions.appendChild(resume);
    actions.appendChild(replay);
    actions.appendChild(remove);
    card.appendChild(details);
    card.appendChild(actions);
    picker.appendChild(card);
  }
}

async function openSessionDropdown() {
  const dropdown = $("session-dropdown");
  const opening = !dropdown.classList.contains("open");
  if (!opening) {
    closeSessionDropdown();
    return;
  }
  const personaPicker = $("session-new-persona");
  personaPicker.innerHTML = $("persona-select").innerHTML;
  personaPicker.value = currentPersona;
  dropdown.classList.add("open");
  $("session-summary").setAttribute("aria-expanded", "true");
  $("session-picker").innerHTML = '<div class="hint">Loading saved sessions…</div>';
  renderSessionPicker(await api("/api/sessions", undefined, "loading sessions"));
}

function syncSessionContext(data) {
  if (!session) return;
  if (Object.hasOwn(data, "trajectory_count")) {
    session.trajectory_count = data.trajectory_count;
  }
  if (Object.hasOwn(data, "corpus")) session.corpus = data.corpus;
  if (Object.hasOwn(data, "rounds")) {
    rounds = data.rounds;
    session.rounds = data.rounds;
  }
  if (Object.hasOwn(data, "unified")) {
    unified = data.unified;
    session.unified = data.unified;
  }
  renderSession();
}

function clearTrajectoryUi() {
  multiTurnActive = false;
  baseTrigger = null;
  clusters = [];
  selectedCluster = null;
  undesirableClusters = new Set();
  selectedClusterSegments = null;
  resetClusterNavigation();
  generatedBounds = [];
  costField = null;
  pendingCost = null;
  unifiedCostField = null;
  clearTraj(...Object.keys(trajs));
  liveBaseMeshIds = [];
  $("scenario-details").open = true;
  showStart = true;
  showMdmStart = false;
  setScenarioLocked(false);
  $("exit-trajectory").disabled = true;
  for (const id of ["ignore-violation", "enter-correction", "generate", "recluster",
    "correction-next", "generate-cost", "cost-next", "commit-round", "apply-round"]) {
    $(id).disabled = true;
  }
  $("trajectory-session").className = "trajectory-session";
  $("trajectory-session").textContent = "";
  $("base-metrics").textContent = "";
  $("oracle-metrics").textContent = "";
  $("correction-metrics").textContent = "";
  $("cost-output").innerHTML = "";
  clearCostThoughtLog();
  renderActiveCosts();
  renderClusterList();
  setWorkflowStage(1);
}

function hydrateTrajectory(data) {
  multiTurnActive = true;
  $("scenario-details").open = false;
  setTraj("base", data.trajectory);
  if (data.oracle) {
    setTraj("oracle", data.oracle.trajectory);
    renderOracleMetrics(data.oracle);
  }
  if (data.clean_base) setTraj("clean_base", data.clean_base);
  baseTrigger = data.trigger ? data.trigger.step : null;
  renderTrajectorySession(data);
  setScenarioLocked(data.status !== "complete");
  $("run-base").disabled = data.status !== "complete";
  $("exit-trajectory").disabled = data.status === "running";
  $("generate").disabled = data.status !== "paused";
  setDecisionControls(data);
  $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) +
    `\nexecuted ${data.step}/${data.step_limit} steps`;
  if (data.pending_cost) {
    renderCostGeneration(data.pending_cost);
    $("commit-round").disabled = multiTurnActive;
    $("apply-round").disabled = !multiTurnActive;
  }
  setWorkflowStage(data.pending_cost ? 3 : 1);
}

function activateSession(data) {
  session = data;
  rounds = data.rounds || [];
  unified = data.unified || null;
  currentPersona = data.persona;
  refreshPersonaSelect();
  onPersonaChange(currentPersona);
  clearTrajectoryUi();
  if (data.trajectory) hydrateTrajectory(data.trajectory);
  renderRounds();
  renderUnifiedOutput();
  renderSession();
  refreshLegend(); refreshTimeline(); renderAll();
  if (data.trajectory?.status === "running" && !replayActive) {
    void rollLiveTrajectory(data.trajectory);
  }
}

async function newSession(persona) {
  const data = await api(
    "/api/session/start",
    { persona: persona || currentPersona },
    "starting session",
  );
  closeSessionDropdown();
  activateSession(data);
}

async function resumeSession(dir) {
  const data = await api("/api/session/resume", { dir }, "resuming session");
  closeSessionDropdown();
  activateSession(data);
}

async function deleteSession(item) {
  if (!window.confirm(`Delete session ${item.persona} from ${item.started}?`)) return;
  const data = await apiDelete(
    `/api/sessions/${encodeURIComponent(item.name)}`,
    "deleting session",
  );
  if (data.active_deleted) {
    session = null;
    rounds = [];
    unified = null;
    clearTrajectoryUi();
    renderRounds();
    renderUnifiedOutput();
    renderSession();
  }
  renderSessionPicker(data.sessions);
}

async function deleteCorpusEntry(index) {
  const data = await apiDelete(`/api/corpus/${index}`, "deleting corpus entry");
  syncSessionContext(data);
}

function renderTrajectorySession(data) {
  const out = $("trajectory-session");
  out.className = `trajectory-session ${data.status}`;
  if (data.status === "running") {
    out.textContent = `Rolling out MPC · frame ${data.step}/${data.step_limit}`;
  } else if (data.status === "paused") {
    const violation = data.trigger.violation === null
      ? "n/a"
      : data.trigger.violation.toFixed(3);
    out.textContent = `Paused at frame ${data.trigger.step}/${data.step_limit} · ` +
      `${data.trigger.reason} · violation ${violation} · ` +
      `feedback turn ${data.rounds.length + 1}`;
  } else {
    out.textContent = (data.reached_goal
      ? `Trajectory complete at frame ${data.step}/${data.step_limit} · goal reached`
      : `Goal not reached at frame ${data.step}/${data.step_limit}`) +
      (data.error ? ` · error: ${data.error}` : "");
  }
}

function renderLiveTrajectoryFrame(data) {
  session.trajectory = data;
  syncSessionContext(data);
  const meshId = data.trajectory.mesh_id;
  if (liveBaseMeshIds.at(-1) !== meshId) {
    liveBaseMeshIds.push(meshId);
    if (liveBaseMeshIds.length > 64) liveBaseMeshIds.shift();
  }
  setTraj("base", data.trajectory);
  baseTrigger = data.trigger ? data.trigger.step : null;
  $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) +
    `\nexecuted ${data.step}/${data.step_limit} steps` +
    (data.error ? `\nerror: ${data.error}` : "");
  renderTrajectorySession(data);
  setScenarioLocked(data.status !== "complete");
  $("run-base").disabled = data.status !== "complete";
  $("exit-trajectory").disabled = data.status === "running";
  setDecisionControls(data);
  $("generate").disabled = data.status !== "paused";
  $("frame-slider").value = Math.max(0, data.trajectory.n_frames - 1);
  refreshLegend();
  refreshTimeline();
  renderAll();
}

async function rollLiveTrajectory(data) {
  renderLiveTrajectoryFrame(data);
  while (data.status === "running") {
    const frameStarted = performance.now();
    setStatus(`rolling out MPC · frame ${data.step}/${data.step_limit}`, "busy");
    await new Promise((resolve) => requestAnimationFrame(resolve));
    data = await api("/api/live_trajectory/step", {});
    renderLiveTrajectoryFrame(data);
    const remaining = LIVE_FRAME_INTERVAL_MS - (performance.now() - frameStarted);
    if (remaining > 0) {
      await new Promise((resolve) => setTimeout(resolve, remaining));
    }
  }
  setStatus(data.status === "paused" ? "paused for input" : "ready");
  return data;
}

async function startManualTrajectory() {
  const data = await api("/api/live_trajectory/start", {
    arm_aa: armAA,
    goal: getGoal(),
  }, "preparing live trajectory");
  $("scenario-details").open = false;
  session.trajectory = data;
  syncSessionContext(data);
  multiTurnActive = true;
  unifiedCostField = null;
  clearTraj("base", "oracle", "correction", "full", "generated", "generated_start");
  clearTraj("unified");
  liveBaseMeshIds = [];
  setTraj("base", data.trajectory);
  setTraj("oracle", data.oracle.trajectory);
  if (data.clean_base) setTraj("clean_base", data.clean_base);
  baseTrigger = data.trigger ? data.trigger.step : null;
  showStart = false;
  clusters = [];
  selectedCluster = null;
  undesirableClusters = new Set();
  selectedClusterSegments = null;
  resetClusterNavigation();
  generatedBounds = [];
  costField = null;
  pendingCost = null;
  renderClusterList();
  $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) +
    `\nexecuted ${data.step}/${data.step_limit} steps` +
    (data.error ? `\nerror: ${data.error}` : "");
  renderOracleMetrics(data.oracle);
  renderTrajectorySession(data);
  setScenarioLocked(data.status === "paused");
  $("run-base").disabled = data.status === "paused";
  $("exit-trajectory").disabled = false;
  setDecisionControls(data);
  $("generate").disabled = data.status !== "paused";
  $("recluster").disabled = true;
  $("generate-cost").disabled = true;
  $("commit-round").disabled = true;
  $("apply-round").disabled = true;
  renderRounds();
  renderUnifiedOutput();
  setWorkflowStage(1);
  refreshLegend(); refreshTimeline(); renderAll();
  await rollLiveTrajectory(data);
}

async function exitManualTrajectory() {
  const data = await api("/api/manual_trajectory/exit", {}, "exiting trajectory");
  session.trajectory = null;
  multiTurnActive = false;
  baseTrigger = null;
  unifiedCostField = null;
  clusters = [];
  selectedCluster = null;
  undesirableClusters = new Set();
  selectedClusterSegments = null;
  resetClusterNavigation();
  generatedBounds = [];
  costField = null;
  pendingCost = null;
  clearTraj(...Object.keys(trajs));
  $("scenario-details").open = true;
  showStart = true;
  showMdmStart = false;
  setScenarioLocked(false);
  $("exit-trajectory").disabled = true;
  for (const id of ["ignore-violation", "enter-correction", "generate", "recluster",
    "correction-next", "generate-cost", "cost-next", "commit-round", "apply-round"]) {
    $(id).disabled = true;
  }
  $("trajectory-session").className = "trajectory-session";
  $("trajectory-session").textContent = "";
  $("base-metrics").textContent = "";
  $("oracle-metrics").textContent = "";
  $("correction-metrics").textContent = "";
  $("cost-output").innerHTML = "";
  clearCostThoughtLog();
  renderClusterList();
  renderRounds();
  renderUnifiedOutput();
  renderSession();
  setWorkflowStage(1);
  setStatus(`trajectory exited · artifacts retained at ${data.artifact_dir}`);
  refreshLegend(); refreshTimeline(); renderAll();
}

function renderOracleMetrics(oracle) {
  const source = oracle.source === "trigger" ? "current MDM trigger" : "initial pose";
  $("oracle-metrics").textContent =
    `oracle from ${source}: ${fmtMetrics(oracle.metrics, oracle.goal_reach)}`;
}

async function ignoreComfortViolation() {
  const data = await api("/api/manual_trajectory/ignore_violation", {},
    "ignoring the current comfort violation and advancing the trajectory");
  session.trajectory = data;
  syncSessionContext(data);
  setTraj("base", data.trajectory);
  baseTrigger = data.trigger ? data.trigger.step : null;
  clearTraj("correction", "full", "generated", "generated_start");
  clusters = [];
  selectedCluster = null;
  undesirableClusters = new Set();
  selectedClusterSegments = null;
  resetClusterNavigation();
  generatedBounds = [];
  costField = null;
  pendingCost = null;
  renderClusterList();
  renderActiveCosts();
  $("cost-output").innerHTML = "";
  clearCostThoughtLog();
  $("correction-metrics").textContent = "";
  $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) +
    `\nexecuted ${data.step}/${data.step_limit} steps` +
    (data.error ? `\nerror: ${data.error}` : "");
  renderTrajectorySession(data);
  setScenarioLocked(data.status === "paused");
  $("run-base").disabled = data.status === "paused";
  setDecisionControls(data);
  $("generate").disabled = data.status !== "paused";
  $("recluster").disabled = true;
  $("generate-cost").disabled = true;
  $("commit-round").disabled = true;
  $("apply-round").disabled = true;
  setWorkflowStage(1);
  refreshLegend(); refreshTimeline(); renderAll();
}

function applyClusterPayload(data) {
  clusters = data.clusters;
  selectedCluster = data.selected_label ?? null;
  undesirableClusters = new Set(data.undesirable_labels ?? []);
  clusterDepth = data.depth;
  clusterPath = data.path;
  canGoBack = data.can_go_back;
  // A payload that landed while the slider kept moving carries the scale it was
  // requested at, which is behind the cursor: writing it back would fight the drag.
  if (pendingScale === null) {
    $("scale").value = data.scale;
    $("scale-num").value = data.scale;
  }
  clearTraj("correction", "full", "generated", "generated_start");
  generatedBounds = [];
  costField = null;
  pendingCost = null;
  renderActiveCosts();
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
  setWorkflowStage(2);
  renderClusterList();
  refreshLegend(); refreshTimeline(); renderAll();
}

async function generate() {
  const data = await api("/api/generate", {
    prompt: $("prompt").value,
    n_samples: +$("n-samples").value,
    n_clusters: +$("n-clusters").value,
    scale: getScale(),
    clusterer: $("clusterer-select").value,
  }, "generating MDM samples + assembling cluster paths (this can take minutes)");
  $("recluster").disabled = false;
  applyClusterPayload(data);
}

// Magnitude slider: re-assemble the current level at the new scale as the
// handle moves. Only one request is in flight at a time and intermediate
// positions are dropped rather than queued, so the redraw trails the cursor by
// at most one round trip.
let rescaleInFlight = false;
let pendingScale = null;

async function liveRescale() {
  if (!clusters.length) return;
  pendingScale = getScale();
  if (rescaleInFlight) return;
  rescaleInFlight = true;
  try {
    while (pendingScale !== null) {
      const scale = pendingScale;
      pendingScale = null;
      applyClusterPayload(await api("/api/rescale", { scale }));
    }
  } finally {
    rescaleInFlight = false;
  }
  renderAll();
}

async function recluster() {
  const data = await api("/api/recluster", {
    n_clusters: +$("n-clusters").value, scale: getScale(),
    clusterer: $("clusterer-select").value,
  }, "re-clustering + assembling cluster paths");
  applyClusterPayload(data);
}

async function refineCluster() {
  if (selectedCluster === null) return;
  const data = await api("/api/refine_cluster", {
    label: selectedCluster,
    n_clusters: +$("n-clusters").value,
    scale: getScale(),
    clusterer: $("clusterer-select").value,
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
  applyPick(label);
}

// Split from pickCluster so replay can re-apply a recorded pick without a call.
function applyPick(label) {
  selectedCluster = label;
  const c = clusters.find((x) => x.label === label);
  selectedClusterSegments = c.full_segments;
  setTraj("full", c.full);
  $("correction-metrics").textContent =
    "full path: " + fmtMetrics(c.full_metrics, c.full_goal_reach);
  $("generate-cost").disabled = false;
  setWorkflowStage(2);
  renderClusterList();
}

function continueFromCorrection() {
  if (selectedCluster === null || undesirableClusters.has(selectedCluster)) return;
  setWorkflowStage(3);
}

function continueFromCost() {
  setWorkflowStage(4);
}

async function markCluster(label) {
  const undesirable = !undesirableClusters.has(label);
  const data = await api("/api/mark_cluster", { label, undesirable },
    undesirable ? "marking cluster wrong" : "unmarking cluster");
  undesirableClusters = new Set(data.undesirable_labels);
  renderClusterList();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function artifactHref(artifactDir, filename) {
  let relative = String(artifactDir || "").replaceAll("\\", "/");
  const marker = "demo_designer_artifacts/";
  const markerIndex = relative.lastIndexOf("/" + marker);
  if (markerIndex >= 0) relative = relative.slice(markerIndex + marker.length + 1);
  else if (relative.startsWith(marker)) relative = relative.slice(marker.length);
  else return null;
  const encoded = relative.split("/").filter(Boolean).map(encodeURIComponent).join("/");
  return encoded ? `/api/artifact/${encoded}/${encodeURIComponent(filename)}` : null;
}

function rationaleHtml(r, artifactDir) {
  if (!r) return "";
  const interpret = r.interpret || {};
  const evidence = interpret.evidence || {};
  const ground = r.ground || {};
  const final = r.final || {};
  const ranking = r.ranking;
  const lines = [];

  if (interpret.preference) {
    lines.push(`<div><b>interpretation:</b> ${escapeHtml(interpret.preference)}</div>`);
  }
  if (interpret.distinguishing_dimension || interpret.direction) {
    lines.push(`<div><b>dimension:</b> ${escapeHtml(interpret.distinguishing_dimension)}` +
      `${interpret.direction ? ` · ${escapeHtml(interpret.direction)}` : ""}</div>`);
  }
  for (const key of ["preference", "distinguishing_dimension", "direction"]) {
    if (evidence[key]) lines.push(`<div class="rationale-evidence"><b>${escapeHtml(key)} evidence:</b> ${escapeHtml(evidence[key])}</div>`);
  }
  if (Object.hasOwn(interpret, "goal_conflict")) {
    lines.push(`<div><b>goal conflict:</b> ${interpret.goal_conflict ? "yes" : "no"}</div>`);
  }
  for (const term of Array.isArray(ground.terms) ? ground.terms : []) {
    lines.push(`<div class="rationale-term"><b>${escapeHtml(term.feature)}</b> ` +
      `${escapeHtml(term.bound_type)} ${escapeHtml(JSON.stringify(term.values || {}))}` +
      `${term.source ? ` — ${escapeHtml(term.source)}` : ""}</div>`);
  }
  if (ground.goal_safety_check) {
    lines.push(`<div><b>goal safety check:</b> ${escapeHtml(ground.goal_safety_check)}</div>`);
  }
  if (final.explanation) {
    lines.push(`<div><b>explanation:</b> ${escapeHtml(final.explanation)}</div>`);
  }
  if (final.recipient_explanation) {
    lines.push(`<div><b>recipient explanation:</b> ${escapeHtml(final.recipient_explanation)}</div>`);
  }
  if (ranking) {
    const rows = Object.entries(ranking.costs || {}).sort(([a], [b]) =>
      a === "chosen_correction" ? -1 : b === "chosen_correction" ? 1 : a.localeCompare(b));
    lines.push(`<div><b>ranking:</b> accuracy ${Number(ranking.rank_accuracy).toFixed(2)}` +
      ` · margin ${Number(ranking.normalized_margin).toFixed(2)}` +
      ` · improves original ${ranking.improves_original_plan === true ? "yes" : ranking.improves_original_plan === false ? "no" : "n/a"}` +
      ` · inert ${ranking.inert ? "yes" : "no"}</div>`);
    lines.push(`<table class="rationale-ranking"><tbody>${rows.map(([name, cost]) => {
      const content = `${escapeHtml(name)}</td><td>${escapeHtml(Number(cost).toPrecision(5))}`;
      return `<tr${name === "chosen_correction" ? ' class="chosen"' : ""}><td>${content}</td></tr>`;
    }).join("")}</tbody></table>`);
  }
  const rationaleUrl = artifactHref(artifactDir, "rationale.json");
  const stageLogUrl = artifactHref(artifactDir, "stage_log.md");
  if (rationaleUrl && stageLogUrl) {
    lines.push(`<div class="rationale-links"><a href="${rationaleUrl}" target="_blank" rel="noopener">rationale.json</a>` +
      ` · <a href="${stageLogUrl}" target="_blank" rel="noopener">stage_log.md</a></div>`);
  }
  return `<details class="cost-rationale"><summary>why this cost</summary>${lines.join("")}</details>`;
}

function renderCostGeneration(data) {
  pendingCost = data;
  generatedBounds = data.generated_bounds || [];
  costField = data.cost_field || null;
  setTraj("generated", data.trajectory);
  setTraj("generated_start", data.start_trajectory);
  const out = $("cost-output");
  out.innerHTML = "";
  const desc = document.createElement("div");
  const finalRationale = data.rationale?.final || {};
  const naturalLanguageDescription = finalRationale.recipient_explanation ||
    finalRationale.explanation || data.description || "(no description)";
  desc.innerHTML = `<b>${escapeHtml(naturalLanguageDescription)}</b>` +
    rationaleHtml(data.rationale, data.artifact_dir);
  const pre = document.createElement("pre");
  pre.dataset.mode = "dev";
  pre.textContent = data.code;
  out.appendChild(desc);
  out.appendChild(pre);
  renderActiveCosts();
  renderUnifiedOutput();
  $("cost-next").disabled = false;
  setWorkflowStage(3);
}

async function generateCost() {
  const backend = $("cost-backend").value;
  const thoughtLog = $("cost-thought-log");
  $("cost-next").disabled = true;
  $("cost-thought-log-wrap").hidden = false;
  thoughtLog.textContent = "";
  await pollLogs();
  costLogActive = true;
  try {
    const data = await api("/api/generate_cost", { backend },
      `generating ${backend} cost + evaluation rollout — this can take several minutes`);
    renderCostGeneration(data);
    $("commit-round").disabled = multiTurnActive;
    $("apply-round").disabled = !multiTurnActive;
  } finally {
    await pollLogs();
    await pollLogs();
    costLogActive = false;
  }
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
    const details = document.createElement("div");
    details.innerHTML = `<b>round ${r.index + 1}</b> · goal [${r.goal.map((v) => v.toFixed(2)).join(", ")}]` +
      ` · trigger @ ${r.trigger_step} (${r.trigger_reason || "feedback"})` +
      `<br>“${escapeHtml(r.feedback_text)}”<br>${escapeHtml(r.description || "")}` +
      rationaleHtml(r.rationale, r.artifact_dir);
    const remove = document.createElement("button");
    remove.className = "danger round-remove";
    remove.dataset.mode = "dev";
    remove.textContent = "Remove feedback";
    remove.onclick = () => removeRound(r.index);
    div.appendChild(details);
    div.appendChild(remove);
    list.appendChild(div);
  }
  $("combine-rounds").disabled = rounds.length < 2;
  $("reset-rounds").disabled = rounds.length === 0 && !unified;
  renderActiveCosts();
}

function activeCostSummary(cost) {
  const finalRationale = cost.rationale?.final || {};
  return finalRationale.recipient_explanation || finalRationale.explanation ||
    cost.description || "No summary available";
}

function renderActiveCosts() {
  const list = $("active-cost-list");
  if (!list) return;
  list.innerHTML = "";
  const costs = [];
  if (pendingCost) {
    costs.push({ label: "Pending generated cost", ...pendingCost });
  }
  if (unified) {
    costs.push({
      label: rounds.length > 1 ? "Active unified cost" : "Active cost",
      ...unified,
    });
  } else {
    for (const round of rounds) {
      costs.push({ label: `Active cost · round ${round.index + 1}`, ...round });
    }
  }
  if (!costs.length) {
    list.innerHTML = '<div class="hint">No active cost functions.</div>';
    return;
  }
  for (const cost of costs) {
    const details = document.createElement("details");
    details.className = "active-cost";
    const summary = document.createElement("summary");
    const label = document.createElement("span");
    label.className = "active-cost-label";
    label.textContent = cost.label;
    const description = document.createElement("span");
    description.className = "active-cost-summary";
    description.textContent = activeCostSummary(cost);
    summary.appendChild(label);
    summary.appendChild(description);
    const pre = document.createElement("pre");
    pre.textContent = cost.code || "(cost code unavailable)";
    details.appendChild(summary);
    details.appendChild(pre);
    list.appendChild(details);
  }
}

function renderUnifiedOutput(extraHtml) {
  const out = $("unified-output");
  out.innerHTML = "";
  const cost = pendingCost || unified;
  if (!cost) return;
  const desc = document.createElement("div");
  const label = pendingCost ? "new cost to apply" : "unified";
  desc.innerHTML = `<b>${label}: ${cost.description || "(no description)"}</b>` +
    (extraHtml || "");
  const pre = document.createElement("pre");
  pre.dataset.mode = "dev";
  pre.textContent = cost.code;
  out.appendChild(desc);
  out.appendChild(pre);
}

async function commitRound() {
  const data = await api("/api/commit_round", {}, "committing round");
  syncSessionContext(data);
  if (session.trajectory) session.trajectory.pending_cost = null;
  pendingCost = null;
  // The committed round's record now carries these bounds.
  generatedBounds = [];
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
  setWorkflowStage(4);
  renderAll();
}

async function applyRoundAndContinue() {
  const data = await api("/api/live_trajectory/apply_round", {},
    "applying feedback before resuming the trajectory");
  session.trajectory = data;
  syncSessionContext(data);
  if (!unified) {
    unifiedCostField = null;
    clearTraj("unified");
  }
  setTraj("base", data.trajectory);
  baseTrigger = data.trigger ? data.trigger.step : null;
  clusters = [];
  selectedCluster = null;
  undesirableClusters = new Set();
  selectedClusterSegments = null;
  resetClusterNavigation();
  generatedBounds = [];
  costField = null;
  pendingCost = null;
  clearTraj("correction", "full", "generated", "generated_start");
  renderClusterList();
  $("cost-output").innerHTML = "";
  clearCostThoughtLog();
  $("correction-metrics").textContent = "";
  $("base-metrics").textContent = fmtMetrics(data.metrics, data.goal_reach) +
    `\nexecuted ${data.step}/${data.step_limit} steps` +
    (data.error ? `\nerror: ${data.error}` : "");
  renderTrajectorySession(data);
  setScenarioLocked(data.status === "paused");
  $("run-base").disabled = data.status === "paused";
  setDecisionControls(data);
  $("generate").disabled = data.status !== "paused";
  $("recluster").disabled = true;
  $("generate-cost").disabled = true;
  $("cost-next").disabled = true;
  $("commit-round").disabled = true;
  $("apply-round").disabled = true;
  renderRounds();
  renderUnifiedOutput();
  setWorkflowStage(1);
  refreshLegend(); refreshTimeline(); renderAll();
  await rollLiveTrajectory(data);
}

async function removeRound(index) {
  const data = await apiDelete(`/api/rounds/${index}`, "removing feedback");
  syncSessionContext(data);
  unifiedCostField = null;
  clearTraj("unified");
  renderRounds();
  renderUnifiedOutput();
  refreshLegend(); refreshTimeline(); renderAll();
}

// The combine request returns immediately; poll for the background result so a
// multi-minute codex run never rides on one long-lived (droppable) connection.
async function waitForCombine() {
  while (true) {
    await new Promise((r) => setTimeout(r, 1500));
    await pollLogs();
    let s;
    try {
      const resp = await fetch("/api/combine_rounds/status");
      if (!resp.ok) continue;
      s = await resp.json();
    } catch (err) { continue; }
    if (s.status === "done") {
      setStatus("ready");
      return s.result;
    }
    if (s.status === "error") {
      setStatus(s.error, "error");
      const err = new Error(s.error);
      err.reported = true;
      throw err;
    }
  }
}

async function combineRounds() {
  const combineLogWrap = $("combine-log-wrap");
  const combineLog = $("combine-log");
  if (combineLogWrap) {
    combineLog.textContent = "";
    combineLogWrap.hidden = false;
  }
  await pollLogs();
  combineLogActive = true;
  let data;
  try {
    setStatus("combining rounds with codex — this can take a long time; watch the console …", "busy");
    await api("/api/combine_rounds", {});
    data = await waitForCombine();
  } finally {
    await pollLogs();
    await pollLogs();
    combineLogActive = false;
  }
  rounds = data.rounds;
  unified = { description: data.description, code: data.code };
  session.rounds = rounds;
  session.unified = unified;
  unifiedCostField = data.cost_field || null;
  // Combined between trajectories there is nothing to roll out on: the unified
  // cost is stored and the next trajectory starts with it installed.
  let extra = "";
  if (data.trajectory) {
    setTraj("unified", data.trajectory);
    extra = "<br>" + fmtMetrics(data.metrics, data.goal_reach).replace("\n", "<br>");
  } else {
    clearTraj("unified");
  }
  extra += `<br>artifacts: ${data.artifact_dir}`;
  if (data.scores) {
    extra += `<br>scores: mean ${data.scores.mean.toFixed(3)} · per round ` +
      Object.entries(data.scores.per_round).map(([k, v]) => `#${k}=${v.toFixed(3)}`).join(" ");
  }
  renderRounds();
  renderUnifiedOutput(extra);
  renderSession();
  renderAll();
}

async function resetRounds() {
  await api("/api/reset_rounds", {}, "resetting rounds");
  rounds = [];
  unified = null;
  pendingCost = null;
  session.rounds = [];
  session.unified = null;
  unifiedCostField = null;
  clearTraj("unified");
  $("commit-round").disabled = true;
  $("apply-round").disabled = true;
  renderRounds();
  renderUnifiedOutput();
  renderSession();
  refreshLegend(); refreshTimeline(); renderAll();
}

// ---------------------------------------------------------------------------
// Cluster list UI
// ---------------------------------------------------------------------------

function renderClusterList() {
  const list = $("cluster-list");
  list.innerHTML = "";
  $("correction-next").disabled =
    selectedCluster === null || undesirableClusters.has(selectedCluster);
  for (const c of clusters) {
    const card = document.createElement("div");
    card.className = "cluster-card" +
      (c.label === selectedCluster ? " selected" : "") +
      (undesirableClusters.has(c.label) ? " undesirable" : "");
    const sw = document.createElement("div");
    sw.className = "cluster-swatch";
    sw.style.background = CLUSTER_COLORS[c.label % CLUSTER_COLORS.length];
    const info = document.createElement("div");
    info.className = "cluster-info";
    info.innerHTML = `<b>cluster ${c.label}</b> · ${c.count} samples` +
      `${c.full_goal_reach.reached ? " · reaches goal" : ""}` +
      `<div data-mode="dev">oracle ${c.oracle_score.toFixed(3)} · ` +
      `full-path viol ${c.full_metrics.mean_violation.toFixed(3)}</div>`;
    card.appendChild(sw);
    card.appendChild(info);
    const mark = document.createElement("button");
    mark.className = "cluster-mark";
    mark.textContent = undesirableClusters.has(c.label) ? "Restore" : "Uncomf";
    mark.onclick = (event) => {
      event.stopPropagation();
      markCluster(c.label);
    };
    card.appendChild(mark);
    card.onclick = () => pickCluster(c.label);
    list.appendChild(card);
  }
  const selected = clusters.find((c) => c.label === selectedCluster);
  const nClusters = +$("n-clusters").value;
  $("cluster-refine").disabled = !selected || nClusters < 2;
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

// Returns the meshData key for this request. `atFrame` fetches that frame
// alone, which is what the magnitude slider wants: a full trajectory mesh is an
// SMPL pass over every frame plus a multi-MB download, far too much to mint per
// slider position when only the frame on screen is drawn.
function requestMesh(data, atFrame = null) {
  const id = data && data.mesh_id;
  if (!id) return null;
  const key = atFrame === null ? id : `${id}@${atFrame}`;
  if (atFrame === null && rescaleInFlight) return key;
  if (meshData.has(key) || meshLoads.has(key)) return key;
  const url = atFrame === null
    ? `/api/mesh/${encodeURIComponent(id)}`
    : `/api/mesh/${encodeURIComponent(id)}?frame=${atFrame}`;
  const load = fetch(url).then(async (response) => {
    if (!response.ok) throw new Error((await response.json()).error);
    const entry = {
      frames: +response.headers.get("X-Mesh-Frames"),
      vertices: +response.headers.get("X-Mesh-Vertices"),
      data: new Float32Array(await response.arrayBuffer()),
    };
    meshLoads.delete(key);
    if (atFrame !== null) {
      // Slider fetches overlap and can land out of order: drop anything the
      // slider has already moved past, then keep only the position it is on.
      if (key !== previewKey) return;
      for (const other of meshData.keys()) {
        if (other.includes("@")) meshData.delete(other);
      }
    }
    meshData.set(key, entry);
    renderAll();
  }).catch((error) => {
    meshLoads.delete(key);
    setStatus(`mesh load failed: ${error}`, "error");
  });
  meshLoads.set(key, load);
  return key;
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

function meshFromKey(key, color, opacity, atFrame) {
  const cached = meshData.get(key);
  if (!cached) return null;
  const fi = Math.min(atFrame, cached.frames - 1);
  const start = fi * cached.vertices * 3;
  return meshObject(cached.data.subarray(start, start + cached.vertices * 3), color, opacity);
}

function trajectoryMesh(data, color, opacity = 0.32, atFrame = frame) {
  return meshFromKey(requestMesh(data), color, opacity, atFrame);
}

// The selected cluster is the body the magnitude slider reshapes, so it takes
// the single-frame path while the slider moves and holds the last body it drew
// until the new one lands — a rescale mints a fresh mesh id, so drawing only
// what has arrived would blank the figure on every slider position.
let lastClusterMesh = null;
let previewKey = null;

function clusterMesh(data, color, opacity) {
  const atFrame = rescaleInFlight ? 0 : frame;
  let key;
  if (rescaleInFlight) {
    previewKey = `${data.mesh_id}@${frame}`;
    key = requestMesh(data, frame);
  } else {
    key = requestMesh(data);
  }
  const mesh = meshFromKey(key, color, opacity, atFrame);
  if (mesh) {
    lastClusterMesh = { key, atFrame };
    return mesh;
  }
  if (lastClusterMesh) {
    return meshFromKey(lastClusterMesh.key, color, opacity, lastClusterMesh.atFrame);
  }
  return null;
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
  let bodyMeshShown = false;

  if (selected) {
    const color = CLUSTER_COLORS[selected.label % CLUSTER_COLORS.length];
    scene.add(lineObject(selected.full.arm_positions.map((f) => f[WRIST]), color, 0.4));
    const mesh = clusterMesh(selected.full, color, 0.26);
    if (mesh) {
      scene.add(mesh);
      bodyMeshShown = true;
    }
  }
  for (const [key, t] of visibleTrajs()) {
    let mesh = trajectoryMesh(t.data, t.color);
    if (!mesh && key === "base") {
      const fallbackId = liveBaseMeshIds.findLast((id) => meshData.has(id));
      if (fallbackId !== undefined) {
        mesh = trajectoryMesh({ mesh_id: fallbackId }, t.color, 0.32, Infinity);
      }
    }
    if (mesh) {
      scene.add(mesh);
      bodyMeshShown = true;
    }
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
  // The "nothing else to draw" fallback is for stages with no body of their own;
  // a slider position whose mesh has not landed yet is not one of those, and
  // swapping the correction body for the start pose there just reads as a glitch.
  if (startPreview && (showStart || (!bodyMeshShown && !rescaleInFlight))) {
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
  const activeCosts = document.createElement("div");
  activeCosts.id = "active-costs-panel";
  activeCosts.className = "panel";
  activeCosts.innerHTML = '<h2>Cost functions</h2><div id="active-cost-list"></div>' +
    '<div class="row">' +
    '<button id="combine-rounds" class="primary" disabled>Combine rounds (codex)</button></div>' +
    '<div id="combine-log-wrap" hidden><pre id="combine-log" aria-live="polite"></pre></div>';
  holder.appendChild(activeCosts);
  renderActiveCosts();
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
  // In dev+replay the bound controls are live, but the persona on screen is a
  // recorded snapshot: saving it would write a past demo into the real library.
  if (replayActive) return;
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
  if (boundDrag || !boundsEditable()) return;
  canvas.style.cursor = featureHit(name, canvas, e) ? "ns-resize" : "";
}

function startFeatureDrag(name, canvas, e) {
  if (!boundsEditable()) return;
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
  const b = coupledGraphSpecs()[gi]?.bound;
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
  if (boundDrag || !boundsEditable()) return;
  canvas.style.cursor = coupledHit(gi, canvas, e) ? "grab" : "";
}

function startCoupledDrag(gi, canvas, e) {
  if (!boundsEditable()) return;
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
      ? activeGeneratedBounds().filter((b) => b.feature === name && !b.coupled)
      : [];
    const oneFeatureFields = [
      showGeneratedBounds ? costField : null,
      showUnifiedBounds ? unifiedCostField : null,
    ].filter((field) => field?.active_features?.length === 1 &&
      field.active_features[0] === name && field.features?.[name]);
    for (const b of [...oracleBounds, ...costBounds]) {
      if (b.low !== null && b.low !== undefined) { ymin = Math.min(ymin, b.low); ymax = Math.max(ymax, b.low); }
      if (b.high !== null && b.high !== undefined) { ymin = Math.min(ymin, b.high); ymax = Math.max(ymax, b.high); }
    }
    for (const field of oneFeatureFields) {
      const pmax = Math.max(0, ...field.penalty);
      const eps = pmax * 0.02;
      for (let i = 0; i < field.penalty.length; i++) {
        if (field.penalty[i] <= eps || !Number.isFinite(field.features[name][i])) continue;
        ymin = Math.min(ymin, field.features[name][i]);
        ymax = Math.max(ymax, field.features[name][i]);
      }
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
    const drawOneFeatureField = (field, rgb) => {
      if (!field || field.active_features?.length !== 1 ||
          field.active_features[0] !== name || !field.penalty.length) return;
      const feature = field.features[name];
      const penalty = field.penalty;
      if (!feature) return;
      let pmax = 0;
      for (const p of penalty) if (Number.isFinite(p) && p > pmax) pmax = p;
      if (pmax <= 0) return;
      const bins = new Array(Math.max(1, Math.floor(ph))).fill(0);
      for (let i = 0; i < penalty.length; i++) {
        if (!Number.isFinite(feature[i]) || !Number.isFinite(penalty[i])) continue;
        const bin = Math.floor(((ymax - feature[i]) / (ymax - ymin)) * bins.length);
        if (bin >= 0 && bin < bins.length) bins[bin] = Math.max(bins[bin], penalty[i]);
      }
      const eps = pmax * 0.02;
      for (let i = 0; i < bins.length; i++) {
        if (bins[i] <= eps) continue;
        ctx.fillStyle = `rgba(${rgb},${0.05 + 0.2 * Math.min(1, bins[i] / pmax)})`;
        ctx.fillRect(ml, mt + i, pw, 1);
      }
    };
    if (showGeneratedBounds) drawOneFeatureField(costField, "53,103,214");
    if (showUnifiedBounds) drawOneFeatureField(unifiedCostField, "0,131,143");
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

// One phase plot per (feature, cond_feature) pair: every persona coupled bound
// plus any generated coupled bound whose pair has no persona plot to land on.
// `bound` is the editable persona bound, or null for generated-only plots.
function coupledGraphSpecs() {
  const specs = currentCoupledBounds().map((b) => ({
    feature: b.feature, cond_feature: b.cond_feature, bound: b,
  }));
  for (const g of activeGeneratedBounds()) {
    if (!g.coupled) continue;
    if (!specs.some((s) => s.feature === g.feature && s.cond_feature === g.cond_feature)) {
      specs.push({ feature: g.feature, cond_feature: g.cond_feature, bound: null });
    }
  }
  return specs;
}

function coupledGraphsKeyOf(specs) {
  return specs.map((s) => `${s.feature}|${s.cond_feature}|${s.bound ? "p" : "g"}`).join(";");
}
let coupledGraphsKey = null;

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
  const specs = coupledGraphSpecs();
  coupledGraphsKey = coupledGraphsKeyOf(specs);
  specs.forEach((spec, i) => {
    const b = spec.bound;
    const panel = document.createElement("div");
    panel.className = "panel";
    const title = document.createElement("div");
    title.className = "graph-title";
    title.textContent = `${spec.feature} vs ${spec.cond_feature} (pose-dependent limit)`;
    const canvas = document.createElement("canvas");
    canvas.className = "coupled-graph";
    canvas.id = `coupled-graph-${i}`;
    canvas.addEventListener("mousedown", (e) => startCoupledDrag(i, canvas, e));
    canvas.addEventListener("mousemove", (e) => hoverCoupled(i, canvas, e));
    panel.appendChild(title);
    panel.appendChild(canvas);
    if (b) {
      const del = document.createElement("button");
      del.textContent = "✕";
      del.className = "coupled-del";
      del.onclick = () => {
        p.bounds.splice(p.bounds.indexOf(b), 1);
        buildCoupledGraphs(); renderAll(); schedulePersonaSave();
      };
      title.appendChild(del);
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
      panel.appendChild(row);
    }
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
  const specs = coupledGraphSpecs();
  if (coupledGraphsKeyOf(specs) !== coupledGraphsKey) buildCoupledGraphs();
  specs.forEach((spec, gi) => {
    const b = spec.bound;
    const canvas = $(`coupled-graph-${gi}`);
    if (!canvas) return;
    const fit = fitCanvas(canvas);
    if (!fit) return;
    const { ctx, w, h } = fit;
    ctx.clearRect(0, 0, w, h);
    const ml = 48, mr = 10, mt = 8, mb = 30;
    const pw = w - ml - mr, ph = h - mt - mb;
    const thr = (x) => b.intercept + b.slope * x;
    const parsedCostBounds = showGeneratedBounds
      ? generatedCoupledBoundsFor(spec.feature, spec.cond_feature)
      : [];

    const series = coupledSeriesList(spec);
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const { xs, ys } of series) {
      for (const v of xs) { xmin = Math.min(xmin, v); xmax = Math.max(xmax, v); }
      for (const v of ys) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
    }
    const xBox = featureBoxRange(spec.cond_feature);
    const yBox = featureBoxRange(spec.feature);
    if (xBox) { xmin = Math.min(xmin, xBox[0]); xmax = Math.max(xmax, xBox[1]); }
    if (yBox) { ymin = Math.min(ymin, yBox[0]); ymax = Math.max(ymax, yBox[1]); }
    for (const costBound of parsedCostBounds) {
      for (const [x, y] of costBound.points) {
        xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
        ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
      }
    }
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
    const violates = (x, y) =>
      b !== null && (b.bound_type === "upper_bound" ? y > thr(x) : y < thr(x));

    ctx.save();
    ctx.beginPath(); ctx.rect(ml, mt, pw, ph); ctx.clip();

    if (b) {
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
    }

    for (const costBound of parsedCostBounds) {
      const path = [
        [xmin, generatedCoupledThreshold(costBound, xmin)],
        ...costBound.points.filter(([x]) => x > xmin && x < xmax),
        [xmax, generatedCoupledThreshold(costBound, xmax)],
      ];
      const costFar = costBound.bound_type === "upper_bound" ? mt - ph : mt + 2 * ph;
      ctx.fillStyle = "rgba(53,103,214,0.12)";
      ctx.beginPath();
      path.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(X(x), Y(y)); else ctx.lineTo(X(x), Y(y));
      });
      ctx.lineTo(X(xmax), costFar); ctx.lineTo(X(xmin), costFar); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = GENERATED_LIMIT_COLOR;
      ctx.lineWidth = 2;
      ctx.beginPath();
      path.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(X(x), Y(y)); else ctx.lineTo(X(x), Y(y));
      });
      ctx.stroke();
    }

    // drag handles for editing the limit line
    if (b) {
      ctx.fillStyle = "rgba(180,40,40,0.95)";
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      for (const hx of [coupledTX[gi].hx0, coupledTX[gi].hx1]) {
        ctx.beginPath(); ctx.arc(X(hx), Y(thr(hx)), 6, 0, 7); ctx.fill(); ctx.stroke();
      }
    }

    // Sampled fields show the compiled cost alongside any exact parsed boundary,
    // including shapes that cannot be represented by grounded bound terms.
    const drawField = (field, rgb) => {
      if (!field || !field.penalty.length) return;
      const cond = field.features[spec.cond_feature];
      const feat = field.features[spec.feature];
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
    // A parsed generated bound is drawn as an exact line above; the sampled dot
    // cloud only fills in when no such line exists for this pair.
    if (!parsedCostBounds.length) {
      if (showGeneratedBounds) drawField(costField, "53,103,214");
      if (showUnifiedBounds) drawField(unifiedCostField, "0,131,143");
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
    ctx.fillText(`${spec.cond_feature} →`, ml + pw / 2, h - 3);
    ctx.save();
    ctx.translate(9, mt + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(`${spec.feature} →`, 0, 0);
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
  const selected = showClusters
    ? clusters.find((c) => c.label === selectedCluster)
    : null;
  if (selected) {
    entries.push(["selected_cluster", {
      data: selected.full,
      color: CLUSTER_COLORS[selected.label % CLUSTER_COLORS.length],
    }]);
  }
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
    if ((key === "full" || key === "selected_cluster") && selectedClusterSegments) {
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
  if (suppressRender) return;
  renderSkeleton();
  renderGraphs();
  renderCoupledGraphs();
  renderStrips();
}

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------

function renderWorkflowStage() {
  for (const button of document.querySelectorAll("#workflow-stage-nav button")) {
    const stage = Number(button.dataset.stage);
    button.disabled = stage > workflowStage;
    button.classList.toggle("completed", stage < workflowStage);
    button.classList.toggle("viewed", stage === displayedStage);
    if (stage === workflowStage) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  }
  for (const panel of document.querySelectorAll("[data-workflow-stage]")) {
    panel.classList.toggle(
      "workflow-stage-visible",
      Number(panel.dataset.workflowStage) === displayedStage,
    );
  }
}

function setWorkflowStage(stage) {
  workflowStage = stage;
  displayedStage = stage;
  renderWorkflowStage();
}

function reviewStage(stage) {
  if (stage > workflowStage) return;
  displayedStage = stage;
  renderWorkflowStage();
}

function applyMode() {
  document.body.classList.toggle("demo", demoMode);
  $("mode-toggle").textContent = demoMode ? "Dev mode" : "Demo mode";
  renderWorkflowStage();
  // Hiding panels changes the column width, so the 3-view scissor rects are
  // stale until the canvas is re-fit.
  refreshLegend();
  renderAll();
}

function toggleMode() {
  demoMode = !demoMode;
  localStorage.setItem("demo_runner_mode", demoMode ? "demo" : "dev");
  applyMode();
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

// Replays a recorded session by re-feeding each stored payload through the live
// render path. No MDM/LLM/MPC calls, and the live session is left untouched.
async function loadReplay(name) {
  const index = await api(`/api/replay/${name}`, undefined, "loading replay");
  replayName = name;
  replayBeats = index.beats;
  replayActive = true;
  replayIndex = 0;
  replayCache.clear();
  $("replay-bar").classList.add("active");
  closeSessionDropdown();
  await showBeat(0);
}

// Beats are fetched once and kept: re-applying a beat must reuse its mesh ids,
// or a rebuild would re-mint them and churn the server's mesh cache.
async function beatData(i) {
  if (!replayCache.has(i)) {
    replayCache.set(i, await api(`/api/replay/${replayName}/${i}`, undefined,
      `replay beat ${i + 1}/${replayBeats.length}`));
  }
  return replayCache.get(i);
}

async function applyBeat(i) {
  const beat = await beatData(i);

  // Install the recorded persona: every violation and graph bound is derived
  // from it client-side, so the demo must not shift under a later edit.
  const at = personas.findIndex((p) => p.name === beat.persona.name);
  if (at >= 0) personas[at] = beat.persona; else personas.push(beat.persona);
  currentPersona = beat.persona.name;
  buildCoupledGraphs();
  renderBoundControls();

  const data = beat.data;
  if (beat.kind === "trajectory") {
    clearTraj("correction", "full", "generated", "generated_start");
    hydrateTrajectory(data);
  } else if (beat.kind === "oracle") {
    setTraj("oracle", data.trajectory);
    renderOracleMetrics(data);
  } else if (beat.kind === "clusters") {
    $("prompt").value = data.prompt || "";
    applyClusterPayload(data);
  } else if (beat.kind === "pick") {
    applyPick(data.label);
  } else if (beat.kind === "cost") {
    renderCostGeneration(data);
  } else if (beat.kind === "round") {
    rounds = data.rounds;
    unified = data.unified;
    renderRounds();
    renderUnifiedOutput();
    setWorkflowStage(4);
  }
  return beat;
}

// A beat holds only its own payload — a clusters beat carries no base rollout —
// so state accumulates forward and a backward step has to rebuild from beat 0
// rather than try to undo. Only the destination beat renders.
async function showBeat(i) {
  const target = Math.max(0, Math.min(replayBeats.length - 1, i));
  const rebuilding = target <= replayIndex;
  if (rebuilding) resetReplayState();
  let beat;
  try {
    for (let k = rebuilding ? 0 : replayIndex + 1; k <= target; k++) {
      suppressRender = k !== target;
      beat = await applyBeat(k);
    }
  } finally {
    suppressRender = false;
  }
  replayIndex = target;

  disableLiveControls();
  $("replay-label").textContent =
    `${replayIndex + 1} / ${replayBeats.length} · ${beat.kind} · ${beat.time}`;
  $("replay-prev").disabled = replayIndex === 0;
  $("replay-next").disabled = replayIndex === replayBeats.length - 1;
  refreshLegend(); refreshTimeline(); renderAll();
}

// Fork the recorded session into a fresh live one and hand control to the live
// UI. The branch inherits the full accumulated context (corpus + committed
// rounds + unified cost); the user continues by running new manual trajectories.
// The original recording is untouched, so its replay still reads cleanly.
async function forkReplay() {
  const data = await api(`/api/replay/${replayName}/fork`, {},
    "forking session for live editing");
  replayActive = false;
  replayName = null;
  replayBeats = [];
  replayIndex = 0;
  replayCache.clear();
  $("replay-bar").classList.remove("active");
  activateSession(data);
}

// Everything a replayed beat can set, back to the state before beat 0.
function resetReplayState() {
  clearTrajectoryUi();
  rounds = [];
  unified = null;
  renderRounds();
  renderUnifiedOutput();
  $("prompt").value = "";
}

// Recorded payloads re-enable buttons (hydrateTrajectory arms Generate when the
// beat was paused), but there is no live session behind them.
function disableLiveControls() {
  for (const id of ["run-base", "exit-trajectory", "ignore-violation",
    "enter-correction", "generate", "recluster", "cluster-refine", "cluster-back",
    "correction-next", "generate-cost", "cost-next", "commit-round", "apply-round",
    "combine-rounds", "reset-rounds"]) {
    $(id).disabled = true;
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function main() {
  INIT = await api("/api/init", undefined, "loading");
  personas = INIT.personas;
  trajectoryConfigs = INIT.trajectory_configs;
  currentPersona = INIT.session ? INIT.session.persona : INIT.default_persona;

  buildArmEditor();
  buildGraphs();
  refreshPersonaSelect();
  refreshTrajectoryConfigSelects();
  setArmEditorValues(INIT.start_arm_aa);
  onPersonaChange(currentPersona);

  $("n-samples").value = INIT.uq.diffusion_samples;
  $("n-clusters").value = INIT.uq.n_clusters;
  $("clusterer-select").value = INIT.uq.clusterer;
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
  $("session-summary").onclick = (event) => {
    event.stopPropagation();
    openSessionDropdown();
  };
  $("session-dropdown").onclick = (event) => event.stopPropagation();
  $("session-new-start").onclick = () => newSession($("session-new-persona").value);
  document.addEventListener("click", closeSessionDropdown);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSessionDropdown();
  });

  $("initial-pose-config").onchange = (event) => selectInitialPoseConfig(event.target.value);
  $("goal-config").onchange = (event) => selectGoalConfig(event.target.value);
  $("save-initial-pose-config").onclick = () => saveTrajectoryConfig("initial_poses");
  $("save-goal-config").onclick = () => saveTrajectoryConfig("goals");
  $("reset-pose").onclick = () => {
    $("initial-pose-config").value = "";
    $("initial-pose-config-name").value = "";
    setArmEditorValues(INIT.start_arm_aa);
  };
  $("run-base").onclick = startManualTrajectory;
  $("exit-trajectory").onclick = exitManualTrajectory;
  $("ignore-violation").onclick = ignoreComfortViolation;
  $("enter-correction").onclick = enterCorrection;
  $("generate").onclick = generate;
  $("recluster").onclick = recluster;
  $("cluster-refine").onclick = refineCluster;
  $("cluster-back").onclick = backCluster;
  $("correction-next").onclick = continueFromCorrection;
  $("generate-cost").onclick = generateCost;
  $("cost-next").onclick = continueFromCost;
  $("n-clusters").onchange = renderClusterList;

  $("commit-round").onclick = commitRound;
  $("apply-round").onclick = applyRoundAndContinue;
  $("combine-rounds").onclick = combineRounds;
  $("reset-rounds").onclick = resetRounds;

  $("mode-toggle").onclick = toggleMode;
  for (const button of document.querySelectorAll("#workflow-stage-nav button")) {
    button.onclick = () => reviewStage(Number(button.dataset.stage));
  }
  $("replay-prev").onclick = () => showBeat(replayIndex - 1);
  $("replay-next").onclick = () => showBeat(replayIndex + 1);
  $("replay-fork").onclick = forkReplay;
  // Reload rather than unwind: replay overwrites the in-memory persona entry
  // with its snapshot, and a fresh /api/init is the honest way back.
  $("replay-exit").onclick = () => location.reload();

  if (INIT.session) {
    activateSession(INIT.session);
  } else {
    session = null;
    rounds = [];
    unified = null;
    clearTrajectoryUi();
    renderRounds();
    renderUnifiedOutput();
    renderSession();
  }

  $("scale").oninput = () => { $("scale-num").value = $("scale").value; liveRescale(); };
  $("scale-num").onchange = () => { $("scale").value = $("scale-num").value; liveRescale(); };

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
  setInterval(pollLogs, 1200);

  window.addEventListener("resize", renderAll);
  refreshLegend();
  refreshTimeline();
  applyMode();
  renderAll();
  requestAnimationFrame(tick);
}

main();
