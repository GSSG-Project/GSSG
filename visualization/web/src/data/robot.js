// Robot store: single source of truth for the robot connection, goal, and plan.
// The top bar (connect + E-STOP), Nav panel, query rows, and inspector all read
// from here, so there is exactly one connection state and one GO pipeline.
//
// Motion contract (the whole point of this design):
//   - selecting an object only PREVIEWS a route (plan, no send)
//   - the robot moves ONLY when goRobot(false) runs — the explicit GO buttons
//   - goRobot(true) = dry-run: full robot pipeline, commands recorded not driven
//   - the connection is never persisted across reloads; the address is.

import {
  navPlan, fetchRobotStatus, setRobotConfig, robotCancel, robotEstop,
} from './api.js?v=3';

const LS_KEY = 'atlas.robot';

function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch (_) { return {}; }
}

const prefs = loadPrefs();

export const robot = {
  address: prefs.address ?? '',   // "host" or "host:port" (default port 8100)
  thresh: prefs.thresh ?? 0.2,    // PRM link threshold (m)
  connected: false,
  connecting: false,
  reachable: false,
  status: null,      // raw robot nav service /api/status payload (via proxy)
  goal: null,        // {x, y, label} in the data-frame ground plane
  plan: null,        // last /api/nav/plan response + {mode: 'preview'|'dry'|'live'}
  planning: false,
  message: null,     // one-line activity note for the panels
  lastError: null,
};

const listeners = new Set();
let pollTimer = null;

function emit() { for (const cb of [...listeners]) cb(robot); }

export function onRobot(cb) {
  listeners.add(cb);
  cb(robot);
  return () => listeners.delete(cb);
}

function persist() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({ address: robot.address, thresh: robot.thresh }));
  } catch (_) { /* private mode */ }
}

export function setAddress(v) {
  robot.address = v.trim();
  persist();
  emit();
}

export function setThresh(v) {
  robot.thresh = v;
  persist();
}

function parseAddress() {
  const a = robot.address;
  if (!a) throw new Error('enter the robot address first');
  const m = a.match(/^([^:\s]+)(?::(\d+))?$/);
  if (!m) throw new Error(`bad address: "${a}" (use host or host:port)`);
  return { host: m[1], port: m[2] ? Number(m[2]) : 8100 };
}

async function pollOnce() {
  try {
    const s = await fetchRobotStatus();
    robot.reachable = !!s.reachable;
    robot.status = s.status ?? null;
    if (!s.reachable) robot.lastError = s.error;
  } catch (e) {
    robot.reachable = false;
    robot.status = null;
    robot.lastError = e.message;
  }
  emit();
}

export async function connectRobot() {
  const { host, port } = parseAddress();   // throws → caller shows the message
  robot.connecting = true;
  robot.lastError = null;
  emit();
  try {
    await setRobotConfig({ host, port, enabled: true });
    robot.connected = true;
    clearInterval(pollTimer);
    pollTimer = setInterval(pollOnce, 2000);
    await pollOnce();
    if (robot.goal) previewPlan();
  } catch (e) {
    robot.lastError = `connect failed: ${e.message}`;
  } finally {
    robot.connecting = false;
    emit();
  }
}

export async function disconnectRobot() {
  clearInterval(pollTimer);
  pollTimer = null;
  robot.connected = false;
  robot.reachable = false;
  robot.status = null;
  robot.plan = null;
  robot.message = null;
  emit();
  try { await setRobotConfig({ enabled: false }); } catch (_) { /* server may be down */ }
}

// New selection → new goal. Clears the previous plan and previews when connected.
export function setRobotGoal(goal) {
  robot.goal = goal;
  robot.plan = null;
  robot.message = null;
  emit();
  if (goal && robot.connected) previewPlan();
}

export async function previewPlan() {
  if (!robot.connected || !robot.goal || robot.planning) return;
  robot.planning = true;
  robot.message = `planning to ${robot.goal.label}…`;
  emit();
  try {
    const res = await navPlan({
      x: robot.goal.x, y: robot.goal.y, thresh: robot.thresh, send: false,
    });
    robot.plan = { ...res, mode: 'preview' };
    robot.message = null;
    robot.lastError = null;
  } catch (e) {
    robot.plan = null;
    robot.lastError = `plan: ${e.message}`;
  } finally {
    robot.planning = false;
    emit();
  }
}

// THE only way the robot moves: dryRun=false submits a live trajectory.
export async function goRobot(dryRun = false) {
  if (!robot.connected) {
    robot.lastError = 'robot not connected — use the top bar';
    emit();
    return;
  }
  if (!robot.goal) {
    robot.lastError = 'no goal — select an object first';
    emit();
    return;
  }
  robot.planning = true;
  robot.message = dryRun ? 'starting dry-run…' : 'GO — submitting live trajectory…';
  emit();
  try {
    const res = await navPlan({
      x: robot.goal.x, y: robot.goal.y, thresh: robot.thresh,
      send: true, dry_run: dryRun,
    });
    robot.plan = { ...res, mode: dryRun ? 'dry' : 'live' };
    robot.message = dryRun
      ? `dry-run started → ${robot.goal.label}`
      : `driving → ${robot.goal.label}`;
    robot.lastError = null;
  } catch (e) {
    robot.lastError = e.message;
    robot.message = null;
  } finally {
    robot.planning = false;
    emit();
  }
}

export async function cancelRobot() {
  try {
    await robotCancel();
    robot.message = 'run cancelled';
  } catch (e) {
    robot.lastError = `cancel: ${e.message}`;
  }
  emit();
}

export async function estopRobot() {
  try {
    await robotEstop();
    robot.message = 'EMERGENCY STOP sent';
  } catch (e) {
    robot.lastError = `estop: ${e.message}`;
  }
  emit();
}

export function clearPlan() {
  robot.plan = null;
  robot.message = null;
  emit();
}
