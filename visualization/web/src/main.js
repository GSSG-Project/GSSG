import * as pc from 'playcanvas';
import { fetchPly, parsePly, plyBoundingBox } from './data/ply.js?v=3';
import { fetchSceneGraph } from './data/objects.js?v=3';
import { aabbReplicaToPC, REPLICA_TO_PC_EULER_X, setUpAxis, pcEulerX, pointReplicaToPC, pointPCToReplica } from './data/frame.js?v=3';
import { fetchScenes, selectScene, fetchSceneMeta, fetchCellsManifest, fetchTrajectory, fetchPrm, PLY_URL } from './data/api.js?v=3';
import { dedupByIoU } from './data/iou.js?v=3';

import { createViewer, frameBounds, setView } from './viewer/viewer.js?v=3';
import { SplatLayer } from './viewer/splat.js?v=3';
import { CellManager } from './viewer/cells.js?v=3';
import { BoxLayer } from './viewer/boxes.js?v=3';
import { TrajectoryLayer } from './viewer/trajectory.js?v=3';
import { Picker } from './viewer/picking.js?v=3';
import { PickProxies } from './viewer/pick-proxies.js?v=3';
import { LabelLayer } from './viewer/labels.js?v=3';
import { applyTheme, getInitialTheme } from './viewer/theme.js?v=3';

import { buildShell } from './ui/shell.js?v=3';
import { createLayers } from './ui/layers.js?v=3';
import { createSceneGraph } from './ui/scene-graph.js?v=3';
import { createRoomFilter } from './ui/class-filter.js?v=3';
import { createInspector } from './ui/inspector.js?v=4';
import { createSettings } from './ui/settings.js?v=3';
import { createFloorStrip } from './ui/floor-strip.js?v=3';
import { createViewportTools } from './ui/viewport-tools.js?v=3';
import { createHud, createStatusBar } from './ui/hud.js?v=3';
import { createQuery } from './ui/query.js?v=6';
import { createNav } from './ui/nav.js?v=4';
import { createRobotBar } from './ui/robot-bar.js?v=4';
import { robot, onRobot, setRobotGoal } from './data/robot.js?v=4';

const els = buildShell(document.getElementById('app'));

applyTheme(getInitialTheme(), null);
const viewer = createViewer(els.canvas);
applyTheme(getInitialTheme(), viewer.camera);

const hud = createHud(els);
const statusbar = createStatusBar(els.statusbar);

// AABBs with this IoU or higher (OR full containment) are considered
// duplicates of the same object — visual cleanup only; backend still
// sees them all.
const IOU_DEDUP_THRESHOLD = 0.1;

// AABBs larger than this volume (m³) are walls / floors / room shells —
// hide their boxes outright and skip them as NMS anchors so they don't
// absorb every smaller object via the containment check.
const MAX_BOX_VOLUME = 6.0;

// Default camera view applied after every scene load. (PC Y-up world space —
// same frame the on-screen HUD reports.) Use the "Frame scene" button to
// snap back to a splat-fit view instead.
const DEFAULT_VIEW = {
  position: [2.5, 1.5, 1.0],
  target:   [23.8, 0.7, -3.9],
};

const state = {
  selectedId: null,
  hoverId: null,
  density: 1.0,
  sizeScale: 1.0,
  opacityScale: 1.0,
  rotX: REPLICA_TO_PC_EULER_X,
  depthMode: 'soft',
  ghost: true,
  boxMode: 'nearby',
  upAxis: 2,
  layers: { splats: true, boxes: true, labels: true, grid: true },
  roomFilter: null,
  theme: getInitialTheme(),
  objects: [],         // visible (deduped) objects shown in viewer / panels
  objectMap: new Map(),// id → visible object
  canonicalById: new Map(),  // any id (incl. absorbed) → canonical-visible id
  sceneName: null,
  sceneTitles: {},
};

let splatLayer = null;   // single-PLY fallback path
let cellManager = null;  // per-cell LOD path (preferred when cells.json exists)
let boxLayer = null;
let picker = null;
let labelLayer = null;
let sceneGraph = null;
let inspector = null;
let queryPanel = null;
let navPanel = null;
let trajectoryLayer = null;  // PRM render layer (scenes with trajectory.csv)
let trajData = null;         // raw /api/nav/trajectory payload (data frame)
let panelsBuilt = false;

// The active splat source — the CellManager (per-cell LOD) when present, else the
// single-PLY SplatLayer. Both expose the same parameter passthroughs + currentCount,
// so settings/layers/HUD code routes through this without caring which path is live.
function activeSplats() {
  return cellManager || splatLayer;
}

function splatCount() {
  return activeSplats()?.currentCount ?? 0;
}

function updateSplatCount() {
  const n = splatCount();
  els.splatCountText.textContent = `${formatK(n)} splats`;
  statusbar.setCount(n);
  hud.setSplats(n);
}

// Splat size/opacity are baked into the PLY (no shader uniform exists), so a
// change means a debounced rebuild. Single-PLY path only; the per-cell path
// keeps its own appearance handling.
let _rebakeTimer = null;
function rebakeAppearance() {
  if (!splatLayer) return;
  clearTimeout(_rebakeTimer);
  _rebakeTimer = setTimeout(async () => {
    els.loaderStatus.textContent = 'applying…';
    els.loader.classList.remove('hidden');
    els.loaderFill.style.width = '30%';
    await splatLayer.refresh();
    els.loaderFill.style.width = '100%';
    els.loaderStatus.textContent = 'ready';
    updateSplatCount();
    setTimeout(() => els.loader.classList.add('hidden'), 200);
  }, 220);
}

// ─── boot ────────────────────────────────────────────────────────────────────

boot().catch((err) => {
  console.error(err);
  els.loaderStatus.textContent = `error: ${err.message}`;
  els.loaderFill.style.background = 'var(--danger)';
});

async function boot() {
  els.loaderStatus.textContent = 'discovering scenes…';
  els.loaderFill.style.width = '5%';
  const { scenes, titles, active } = await fetchScenes();
  state.sceneTitles = titles || {};

  populateSceneSelect(scenes, active);
  await loadActiveScene(active);
}

function populateSceneSelect(scenes, active) {
  if (!scenes.length) {
    els.sceneSelect.innerHTML = `<option>no scenes</option>`;
    els.sceneSelect.disabled = true;
    return;
  }
  els.sceneSelect.innerHTML = scenes.map((s) =>
    `<option value="${s}" ${s === active ? 'selected' : ''}>${state.sceneTitles[s] || s}</option>`).join('');
  els.sceneSelect.addEventListener('change', async () => {
    const name = els.sceneSelect.value;
    try {
      els.loader.classList.remove('hidden');
      els.loaderStatus.textContent = `switching to ${name}…`;
      els.loaderFill.style.width = '5%';
      await selectScene(name);
      await reloadScene(name);
    } catch (e) {
      els.loaderStatus.textContent = `error: ${e.message}`;
      els.loaderFill.style.background = 'var(--danger)';
    }
  });
}

async function loadActiveScene(name) {
  if (!name) {
    els.loaderStatus.textContent = 'no scene loaded';
    els.loaderFill.style.background = 'var(--danger)';
    return;
  }
  await loadSceneData(name);
}

async function reloadScene(name) {
  // Tear down current viewer state.
  if (splatLayer?.entity) splatLayer.entity.destroy();
  splatLayer = null;
  cellManager?.dispose?.();
  cellManager = null;
  boxLayer?.dispose?.();
  boxLayer = null;
  labelLayer?.dispose?.();
  labelLayer = null;
  picker?.dispose?.();
  picker = null;
  trajectoryLayer?.dispose?.();
  trajectoryLayer = null;
  trajData = null;
  state.objects = [];
  state.objectMap.clear();
  state.selectedId = null;

  // Tear down panels that depended on the prior scene's objects.
  panelsBuilt = false;
  await loadSceneData(name);
}

async function loadSceneData(name) {
  state.sceneName = name;
  els.crumbScene.textContent = state.sceneTitles?.[name] || name;

  // The scene's vertical axis (from the run manifest) decides the data→PC
  // transform. Must run before any geometry is transformed (scene graph, splats).
  try {
    const meta0 = await fetchSceneMeta();
    state.upAxis = meta0.up_axis === 1 ? 1 : 2;
  } catch (_) {
    state.upAxis = 2;
  }
  setUpAxis(state.upAxis);
  state.rotX = pcEulerX();

  els.loaderStatus.textContent = 'fetching scene graph…';
  els.loaderFill.style.width = '20%';
  const sg = await fetchSceneGraph();

  // NMS-style dedup + huge-box filter + containment hide. See data/iou.js
  // for details. `canonicalById` maps any id (visible, absorbed, or huge)
  // to its visible canonical, so external selections (CLIP / LLM query)
  // still land on a drawn box.
  const { visible, canonicalById, huge } = dedupByIoU(sg.objects, {
    iouThreshold: IOU_DEDUP_THRESHOLD,
    maxVolume: MAX_BOX_VOLUME,
  });
  console.info(
    `[atlas] objects: ${sg.objects.length} input → ${visible.length} visible ` +
    `(huge>${MAX_BOX_VOLUME}m³: ${huge.length} hidden, IoU≥${IOU_DEDUP_THRESHOLD} + containment dedup)`
  );
  sg.objects = visible;
  state.objects = visible;
  state.canonicalById = canonicalById;
  state.objectMap.clear();
  // Visible objects: drawn + selectable. Huge ones: kept in objectMap only
  // so inspector / status bar still work if something picks them by id.
  for (const o of visible) state.objectMap.set(o.id, o);
  for (const o of huge) state.objectMap.set(o.id, o);

  if (!panelsBuilt) {
    buildPanels(sg);
    panelsBuilt = true;
  } else {
    rebuildScenePanels(sg);
  }

  // Prefer the per-cell export (fast load + distance LOD). Fall back to the single
  // monolithic PLY when the run has no cells.json (legacy runs, or export disabled).
  els.loaderStatus.textContent = 'checking per-cell map…';
  els.loaderFill.style.width = '35%';
  let cellsManifest = null;
  try {
    cellsManifest = await fetchCellsManifest();
  } catch (e) {
    console.warn('[atlas] cells manifest fetch failed; using single PLY', e);
  }

  if (cellsManifest && cellsManifest.cells?.length) {
    els.loaderStatus.textContent =
      `loading map (${cellsManifest.cell_count} cells, ${cellsManifest.total_count.toLocaleString()} splats)…`;
    els.loaderFill.style.width = '50%';
    cellManager = new CellManager(viewer.app, viewer.pickLayer, {
      rotX: state.rotX,
      sizeScale: state.sizeScale,
      opacityScale: state.opacityScale,
      depthMode: state.depthMode,
    });
    cellManager.setManifest(cellsManifest);
    // Frame the whole map, then load the cells nearest to that initial target.
    const wb = cellManager.worldBoundsReplica();
    const b = aabbReplicaToPC(wb.min, wb.max);
    const initialTargetPC = [
      (b.min[0] + b.max[0]) / 2,
      (b.min[1] + b.max[1]) / 2,
      (b.min[2] + b.max[2]) / 2,
    ];
    await cellManager.loadInitial(initialTargetPC);
    els.loaderFill.style.width = '90%';
  } else {
    els.loaderStatus.textContent = 'fetching splats…';
    els.loaderFill.style.width = '35%';
    const buffer = await fetchPly(PLY_URL, (p) => {
      els.loaderFill.style.width = `${35 + (p * 40).toFixed(1)}%`;
    });

    els.loaderStatus.textContent = 'parsing splats…';
    els.loaderFill.style.width = '80%';
    const parsed = parsePly(buffer);
    els.loaderStatus.textContent = `${parsed.count.toLocaleString()} splats parsed · building scene…`;
    els.loaderFill.style.width = '90%';

    splatLayer = new SplatLayer(viewer.app, viewer.pickLayer);
    splatLayer.setParsed(parsed);
    splatLayer.rotation.x = state.rotX;
    await splatLayer.setDensity(state.density);
  }

  boxLayer = new BoxLayer(viewer.app, viewer.camera, viewer.orbit);
  boxLayer.setObjects(sg.objects);
  boxLayer.setAllById(state.objectMap); // so query-selected huge/hidden objects still draw
  boxLayer.setMode(state.boxMode);
  boxLayer.setShowGhost(state.ghost);
  boxLayer.setRoomFilter(state.roomFilter);

  const pickProxies = new PickProxies(viewer.app, viewer.pickLayer);
  pickProxies.setObjects(sg.objects);

  picker = new Picker(viewer.app, viewer.camera, els.canvas, viewer.pickLayer, pickProxies);
  picker.setObjects(sg.objects);

  labelLayer = new LabelLayer(viewer.app, viewer.camera, els.labelLayer, sg.objects);
  els.labelLayer.addEventListener('label-click', (e) => selectObject(e.detail.id, { frame: false }));

  await loadTrajectory();

  // Frame the whole map on load so the camera is sensibly placed for any scene
  // and any vertical axis (the old fixed default view pointed at a hardcoded
  // coordinate that was only right for one Replica scene).
  frameWholeScene();

  els.loaderFill.style.width = '100%';
  els.loaderStatus.textContent = 'ready';
  setTimeout(() => els.loader.classList.add('hidden'), 250);

  // Give the viewport keyboard focus so arrows/WASD drive the camera right away
  // (otherwise the auto-focused query box swallows the keys until you click).
  setTimeout(() => els.canvas.focus?.(), 300);

  updateSplatCount();

  // health → provider pill
  try {
    const meta = await fetchSceneMeta();
    const h = await (await fetch('/api/health')).json();
    els.providerPill.textContent = h.llm_provider
      ? `LLM ${h.llm_provider}` : 'LLM —';
  } catch (_) { /* ignore */ }

  wireCanvasEvents();
  wireFpsHud();
}

// ─── trajectory / PRM ────────────────────────────────────────────────────────

// Ground-plane axis indices of the data frame ([a, b] plane, h vertical).
function planeAxes() {
  return state.upAxis === 2 ? { a: 0, b: 1, h: 2 } : { a: 0, b: 2, h: 1 };
}

// Lift a robot/planner ground-plane (x, y) into a full data-frame point, taking
// the height of the nearest trajectory node so markers sit on the map floor.
function groundToData(x, y) {
  const { a, b, h } = planeAxes();
  let hVal = 0;
  if (trajData?.nodes?.length) {
    let bd = Infinity;
    for (const n of trajData.nodes) {
      const d = (n[a] - x) ** 2 + (n[b] - y) ** 2;
      if (d < bd) { bd = d; hVal = n[h]; }
    }
  }
  const p = [0, 0, 0];
  p[a] = x; p[b] = y; p[h] = hVal;
  return p;
}

async function loadTrajectory() {
  try {
    const t = await fetchTrajectory();
    if (!t) {
      navPanel?.setGraphInfo(null);
      return;
    }
    trajData = t;
    trajectoryLayer = new TrajectoryLayer(viewer.app);
    trajectoryLayer.setNodes(t.nodes.map(pointReplicaToPC));
    trajectoryLayer.setVisible(navPanel?.show ?? true);
    const prm = await fetchPrm(robot.thresh);
    trajectoryLayer.setProxEdges(prm.edges);
    navPanel?.setGraphInfo({ nodes: t.count, shortcuts: prm.count });
    console.info(`[atlas] trajectory: ${t.count} nodes, ${prm.count} shortcut edges`);
  } catch (e) {
    console.warn('[atlas] trajectory load failed', e);
    navPanel?.setGraphInfo(null);
  }
}

// Store → viewport: draw the planned route, the target marker, and the live
// robot pose whenever the robot store changes. Reads the current trajectoryLayer
// through the closure, so it survives scene reloads.
onRobot((r) => {
  if (!trajectoryLayer) return;
  if (r.plan?.path_nodes?.length) {
    const pc3 = r.plan.path_nodes.map(pointReplicaToPC);
    trajectoryLayer.setPath(pc3);
    trajectoryLayer.setTarget(pc3[pc3.length - 1]);
  } else {
    trajectoryLayer.setPath(null);
    trajectoryLayer.setTarget(null);
  }
  const bp = r.connected && r.reachable ? r.status?.base_pose : null;
  trajectoryLayer.setRobot(
    bp != null ? pointReplicaToPC(groundToData(Number(bp.x), Number(bp.y))) : null,
  );
});

function wireCanvasEvents() {
  els.canvas.onclick = (e) => {
    if (!picker) return;
    picker.invalidate();
    const id = picker.pick(e.clientX, e.clientY);
    selectObject(id, { frame: false });
  };
  let hoverTimer = null;
  let lastHoverEvent = null;
  els.canvas.onmousemove = (e) => {
    if (!picker) return;
    lastHoverEvent = { x: e.clientX, y: e.clientY };
    if (hoverTimer) return;
    hoverTimer = setTimeout(() => {
      hoverTimer = null;
      if (!lastHoverEvent) return;
      const id = picker.pick(lastHoverEvent.x, lastHoverEvent.y);
      setHover(id);
    }, 33);
  };
  els.canvas.onmouseleave = () => setHover(null);
}

function wireFpsHud() {
  // Avoid stacking multiple update listeners across scene reloads.
  if (wireFpsHud._wired) return;
  wireFpsHud._wired = true;
  let frames = 0;
  let lastT = performance.now();
  viewer.app.on('update', () => {
    frames++;
    const now = performance.now();
    // Distance-driven LOD: load/decimate/unload cells around the camera target.
    // CellManager.update is internally throttled, so calling every frame is cheap.
    if (cellManager) {
      const t = viewer.orbit.getTarget();
      cellManager.update([t.x, t.y, t.z], now);
    }
    if (now - lastT >= 500) {
      hud.setFps((frames * 1000) / (now - lastT));
      frames = 0;
      lastT = now;
      hud.setCam(viewer.camera.getPosition(), viewer.orbit.getTarget());
      if (cellManager) updateSplatCount();  // resident count drifts as cells load/unload
    }
  });
}

function rebuildScenePanels(sg) {
  // Rebuild only the panels that depend on the scene's objects.
  els.sceneGraphPanel.innerHTML = '';
  els.roomFilterPanel.innerHTML = '';
  sceneGraph = createSceneGraph(els.sceneGraphPanel, els.sgSearch, sg.objects, {
    onSelect: (id) => selectObject(id, { frame: false }),
    onHover: (id) => setHover(id),
  });
  createRoomFilter(els.roomFilterPanel, els.roomFilterCount, sg.objects, {
    onChange: (filter) => {
      state.roomFilter = filter;
      boxLayer?.setRoomFilter(filter);
      labelLayer?.setRoomFilter(filter);
    },
  });
}

// ─── one-shot panel construction (first scene) ───────────────────────────────

function buildPanels(sg) {
  const layers = createLayers(els.layersPanel, (key, value) => {
    state.layers[key] = value;
    if (key === 'splats') activeSplats()?.setVisible(value);
    if (key === 'boxes' && boxLayer) boxLayer.setVisible(value);
    if (key === 'labels' && labelLayer) labelLayer.setVisible(value);
    if (key === 'grid') {
      const g = els.viewport.querySelector('.grid-bg');
      if (g) g.style.display = value ? '' : 'none';
    }
  });
  layers.setMeta('boxes', `${sg.objects.length} objs`);
  layers.setMeta('labels', `≤24 vis`);

  sceneGraph = createSceneGraph(els.sceneGraphPanel, els.sgSearch, sg.objects, {
    onSelect: (id) => selectObject(id, { frame: false }),
    onHover: (id) => setHover(id),
  });

  createRoomFilter(els.roomFilterPanel, els.roomFilterCount, sg.objects, {
    onChange: (filter) => {
      state.roomFilter = filter;
      boxLayer?.setRoomFilter(filter);
      labelLayer?.setRoomFilter(filter);
    },
  });

  inspector = createInspector(els.inspectorPanel, els.inspectorSubtitle);
  inspector.onFrame((o) => frameBounds(viewer.camera, viewer.orbit, o.min, o.max, 2.4));

  queryPanel = createQuery(els.queryPanel, els.queryStatus, {
    onSelect: (id, opts) => {
      els.activateTab?.('query');
      selectObject(id, opts || { frame: false });
    },
    resolveCanonical: (id) => state.canonicalById.get(id) ?? id,
  });

  navPanel = createNav(els.navPanel, {
    onShow: (v) => trajectoryLayer?.setVisible(v),
    onThresh: async (v) => {
      if (!trajectoryLayer || !trajData) return;
      try {
        const prm = await fetchPrm(v);
        trajectoryLayer.setProxEdges(prm.edges);
        navPanel.setGraphInfo({ nodes: trajData.count, shortcuts: prm.count });
      } catch (e) {
        console.warn('[atlas] prm refresh failed', e);
      }
    },
  });

  createRobotBar(els.robotBar);

  createSettings(els.settingsPanel, {
    density: state.density,
    size: state.sizeScale,
    opacity: state.opacityScale,
    rotX: state.rotX,
    depthMode: state.depthMode,
    ghost: state.ghost,
    boxMode: state.boxMode,
  }, {
    onDensity: async (v) => {
      state.density = v;
      // Per-cell path manages density automatically via distance LOD; the manual
      // slider applies only to the single-PLY fallback.
      if (!splatLayer) return;
      els.loaderStatus.textContent = 're-packing…';
      els.loader.classList.remove('hidden');
      els.loaderFill.style.width = '20%';
      await splatLayer.setDensity(v);
      els.loaderFill.style.width = '100%';
      els.loaderStatus.textContent = 'ready';
      updateSplatCount();
      setTimeout(() => els.loader.classList.add('hidden'), 200);
    },
    onSize: (v) => { state.sizeScale = v; activeSplats()?.setSizeScale(v); rebakeAppearance(); },
    onOpacity: (v) => { state.opacityScale = v; activeSplats()?.setOpacityScale(v); rebakeAppearance(); },
    onRotX: (v) => { state.rotX = v; activeSplats()?.setRotationX(v); },
    onDepthMode: (v) => { state.depthMode = v; activeSplats()?.setDepthMode(v); },
    onGhost: (v) => { state.ghost = v; boxLayer?.setShowGhost(v); },
    onBoxMode: (v) => { state.boxMode = v; boxLayer?.setMode(v); },
  });

  createFloorStrip(els.floorStrip, sg.floors.length ? sg.floors : [0], 0, {
    onChange: () => { /* no-op for single floor */ },
  });

  createViewportTools(els.vpTools, {
    onTool: (t) => {
      if (t === 'frameAll') {
        frameWholeScene();
      } else if (t === 'frameSel') {
        const o = state.objectMap.get(state.selectedId);
        if (o) frameBounds(viewer.camera, viewer.orbit, o.min, o.max, 2.4);
      } else if (t === 'top') {
        viewer.orbit.setYawPitch(0, -Math.PI / 2 + 0.0001);
      } else if (t === 'side') {
        viewer.orbit.setYawPitch(Math.PI / 2, 0);
      }
    },
  });

  els.themeToggle.addEventListener('click', () => {
    state.theme = state.theme === 'night' ? 'day' : 'night';
    applyTheme(state.theme, viewer.camera);
    els.themeToggle.textContent = state.theme === 'night' ? '☾' : '☀';
  });
  els.themeToggle.textContent = state.theme === 'night' ? '☾' : '☀';

  els.frameAllBtn.addEventListener('click', frameWholeScene);
}

// Frame the whole map, from whichever splat source is active (cell AABBs in the
// per-cell path, parsed-PLY bounds in the single-PLY fallback). Replica→PC frame.
function frameWholeScene() {
  let raw = null;
  if (cellManager) raw = cellManager.worldBoundsReplica();
  else if (splatLayer?.parsed) raw = plyBoundingBox(splatLayer.parsed);
  if (!raw || !isFinite(raw.min[0])) return;
  const b = aabbReplicaToPC(raw.min, raw.max);
  frameBounds(viewer.camera, viewer.orbit, b.min, b.max, 1.3);
}

// ─── selection / hover ──────────────────────────────────────────────────────

function selectObject(id, opts = {}) {
  // External selections (CLIP / LLM query) may point at an absorbed id whose
  // box was deduped away — redirect to its canonical visible representative.
  const canonical = id == null ? null : (state.canonicalById.get(id) ?? id);
  state.selectedId = canonical;
  boxLayer?.setSelected(canonical);
  labelLayer?.setSelected(canonical);
  sceneGraph?.setSelected(canonical);
  const o = canonical != null ? state.objectMap.get(canonical) : null;
  inspector?.setSelected(o);
  statusbar.setSelected(o);
  if (opts.frame && o) frameBounds(viewer.camera, viewer.orbit, o.min, o.max, 2.4);

  // Nav: every selection (click, scene graph, CLIP/LLM query) becomes the shared
  // plan goal — projected to the data-frame ground plane the planner works in.
  // This only PREVIEWS a route; motion needs an explicit GO press.
  if (o) {
    const c = pointPCToReplica(o.center);
    const { a, b } = planeAxes();
    setRobotGoal({ x: c[a], y: c[b], label: `object #${o.id}` });
  } else {
    setRobotGoal(null);
  }
}

function setHover(id) {
  if (state.hoverId === id) return;
  state.hoverId = id;
  boxLayer?.setHover(id);
  labelLayer?.setHover(id);
  els.canvas.style.cursor = id != null ? 'pointer' : '';
}

function formatK(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

// Expose tab switcher to inner callbacks.
els.activateTab = els.activateTab; // already attached by shell.js