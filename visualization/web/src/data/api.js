// Backend API client.
//
// All endpoints are served by visualization/server.py on the same
// origin. Paths are absolute (`/api/...`) so the frontend can live anywhere
// the FastAPI app mounts /.

const API = '/api';

export async function fetchScenes() {
  const res = await fetch(`${API}/scenes`);
  if (!res.ok) throw new Error(`scenes fetch failed: ${res.status}`);
  return res.json();
}

export async function selectScene(name) {
  const res = await fetch(`${API}/scenes/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scene: name }),
  });
  if (!res.ok) throw new Error(`select scene failed: ${res.status}`);
  return res.json();
}

export async function fetchSceneMeta() {
  const res = await fetch(`${API}/scene`);
  if (!res.ok) throw new Error(`scene meta failed: ${res.status}`);
  return res.json();
}

export async function fetchSceneGraphJson() {
  const res = await fetch(`${API}/scene/scene_graph`);
  if (!res.ok) throw new Error(`scene graph fetch failed: ${res.status}`);
  return res.json();
}

export const PLY_URL = `${API}/scene/ply`;

// Per-cell viewer manifest. Returns the parsed cells.json, or null when the
// active scene has no per-cell export (the caller falls back to the single PLY).
export async function fetchCellsManifest() {
  const res = await fetch(`${API}/scene/cells`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`cells manifest fetch failed: ${res.status}`);
  return res.json();
}

// URL of one cell's PLY (FileResponse from the backend).
export function cellPlyUrl(cellId) {
  return `${API}/scene/cell/${cellId}`;
}

// ── nav / robot bridge ───────────────────────────────────────────────────────

async function jsonOrThrow(res) {
  if (!res.ok) {
    let detail = `${res.status}`;
    try { detail = (await res.json()).detail ?? detail; } catch (_) { /* keep status */ }
    throw new Error(detail);
  }
  return res.json();
}

// Recorded trajectory nodes (data frame). null when the scene has none.
export async function fetchTrajectory() {
  const res = await fetch(`${API}/nav/trajectory`);
  if (res.status === 404) return null;
  return jsonOrThrow(res);
}

// Proximity (shortcut) edges of the PRM at the given threshold (meters).
export async function fetchPrm(thresh) {
  return jsonOrThrow(await fetch(`${API}/nav/prm?thresh=${encodeURIComponent(thresh)}`));
}

// Plan to a data-frame ground-plane goal; optionally submit to the robot.
export async function navPlan(body) {
  return jsonOrThrow(await fetch(`${API}/nav/plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }));
}

export async function fetchRobotConfig() {
  return jsonOrThrow(await fetch(`${API}/robot/config`));
}

export async function setRobotConfig(cfg) {
  return jsonOrThrow(await fetch(`${API}/robot/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  }));
}

export async function fetchRobotStatus() {
  return jsonOrThrow(await fetch(`${API}/robot/status`));
}

export async function robotCancel() {
  return jsonOrThrow(await fetch(`${API}/robot/cancel`, { method: 'POST' }));
}

export async function robotEstop() {
  return jsonOrThrow(await fetch(`${API}/robot/estop`, { method: 'POST' }));
}

export async function clipQuery(text, k = 10) {
  const res = await fetch(`${API}/clip_query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, k }),
  });
  if (!res.ok) throw new Error(`clip query failed: ${res.status}`);
  return res.json();
}

export async function llmQuery(text) {
  const res = await fetch(`${API}/llm_query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`llm query failed: ${res.status}`);
  return res.json();
}

// Open an SSE stream for the agent. `onEvent({type, ...})` is called for
// every event; resolves when the stream closes (done or error), or when the
// caller invokes `.abort()` on the returned controller.
export function llmQueryStream(text, onEvent) {
  const url = `${API}/llm_query_stream?q=${encodeURIComponent(text)}`;
  const es = new EventSource(url);

  const handle = (e) => {
    if (e.data == null || e.data === '') return; // EventSource keepalive / empty
    try {
      onEvent(JSON.parse(e.data));
    } catch (err) {
      onEvent({ type: 'error', message: `bad event payload: ${err}` });
    }
  };
  for (const t of ['meta', 'assistant_text', 'tool_call', 'tool_result',
                   'terminal', 'done', 'error']) {
    es.addEventListener(t, handle);
  }
  es.onerror = () => {
    // EventSource reconnects automatically; we close it on `done`/`error`
    // events from the server. A transport error here means we couldn't reach
    // the server at all.
    onEvent({ type: 'error', message: 'sse connection lost' });
    es.close();
  };

  return {
    abort() { es.close(); },
    source: es,
  };
}