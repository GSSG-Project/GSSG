import { fetchPly, parsePly, buildCleanPly } from '../data/ply.js?v=3';
import { cellPlyUrl } from '../data/api.js?v=3';
import { pointReplicaToPC } from '../data/frame.js?v=3';
import { SplatLayer } from './splat.js?v=3';

// Per-cell splat loader with distance-driven level-of-detail.
//
// Instead of one monolithic multi-GB PLY, a finished map is exported as one PLY
// per ~12 m "viewer cell" (see SubmapManager.save_stable_ply_per_cell). This
// manager:
//   * fetches the nearest K cells to the camera target in parallel on load,
//   * keeps one SplatLayer (gsplat entity) per loaded cell,
//   * on every frame, re-evaluates each cell's distance to the camera target and
//       - loads cells that came within the load radius,
//       - sets near cells to full density, far ones to a decimated fraction,
//       - destroys cells past the unload radius (hysteresis: unload > load so
//         cells near the boundary don't thrash).
//
// All cell centers/AABBs in cells.json are in the capture (Replica Z-up) frame;
// the camera target is PC Y-up. We convert centers to PC once for comparison.

const NEAR_FULL_M = 8.0;     // within this -> full density
const FAR_MIN_M = 30.0;      // beyond this -> minimum density (clamped to DECIMATE_MIN)
const DECIMATE_MIN = 0.08;   // far-cell density floor
const LOAD_RADIUS_M = 45.0;  // load cells whose center is within this of the target
const UNLOAD_RADIUS_M = 60.0;// destroy cells past this (hysteresis vs LOAD_RADIUS_M)
const MAX_LOADED_CELLS = 64; // hard cap on concurrently resident cells
const INITIAL_K = 12;        // cells fetched in parallel on first load
const UPDATE_EVERY_MS = 250; // throttle LOD re-evaluation off the 60fps loop

export class CellManager {
  constructor(app, pickLayer, opts = {}) {
    this.app = app;
    this.pickLayer = pickLayer;
    this.manifest = null;
    this.cells = [];                 // [{cell_id, center, aabb, count, centerPC:[x,y,z]}]
    this.layers = new Map();         // cell_id -> SplatLayer
    this.loading = new Set();        // cell_ids with an in-flight fetch
    this.parsedCache = new Map();    // cell_id -> parsed PLY (kept while loaded for re-LOD)
    this.densityOf = new Map();      // cell_id -> current density fraction
    this.rotX = opts.rotX ?? 0;
    this.sizeScale = opts.sizeScale ?? 1.0;
    this.opacityScale = opts.opacityScale ?? 1.0;
    this.depthMode = opts.depthMode ?? 'soft';
    this.visible = true;
    this._disposed = false;
    this._lastUpdate = 0;
    this._target = [0, 0, 0];
  }

  // Total splats currently resident across all loaded cells (post-decimation).
  get currentCount() {
    let n = 0;
    for (const l of this.layers.values()) n += l.currentCount || 0;
    return n;
  }

  setManifest(manifest) {
    this.manifest = manifest;
    this.cells = (manifest.cells || []).map((c) => ({
      ...c,
      centerPC: pointReplicaToPC(c.center),
    }));
  }

  // World-frame AABB across all cells (Replica frame) for camera framing.
  worldBoundsReplica() {
    let mn = [Infinity, Infinity, Infinity];
    let mx = [-Infinity, -Infinity, -Infinity];
    for (const c of this.cells) {
      const [x0, y0, z0, x1, y1, z1] = c.aabb;
      mn = [Math.min(mn[0], x0), Math.min(mn[1], y0), Math.min(mn[2], z0)];
      mx = [Math.max(mx[0], x1), Math.max(mx[1], y1), Math.max(mx[2], z1)];
    }
    return { min: mn, max: mx };
  }

  _dist(centerPC, target) {
    const dx = centerPC[0] - target[0];
    const dy = centerPC[1] - target[1];
    const dz = centerPC[2] - target[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  _densityForDist(d) {
    if (d <= NEAR_FULL_M) return 1.0;
    if (d >= FAR_MIN_M) return DECIMATE_MIN;
    // linear ramp between near and far
    const t = (d - NEAR_FULL_M) / (FAR_MIN_M - NEAR_FULL_M);
    return Math.max(DECIMATE_MIN, 1.0 - t * (1.0 - DECIMATE_MIN));
  }

  // Initial load: fetch the nearest INITIAL_K cells to `targetPC` in parallel.
  async loadInitial(targetPC) {
    this._target = targetPC.slice();
    const sorted = this.cells
      .map((c) => ({ c, d: this._dist(c.centerPC, targetPC) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, INITIAL_K);
    await Promise.all(sorted.map(({ c, d }) => this._ensureCell(c, this._densityForDist(d))));
  }

  async _ensureCell(cell, density) {
    const id = cell.cell_id;
    if (this.layers.has(id)) {
      await this._setCellDensity(id, density);
      return;
    }
    if (this.loading.has(id)) return;
    if (this.layers.size >= MAX_LOADED_CELLS) return;
    this.loading.add(id);
    try {
      let parsed = this.parsedCache.get(id);
      if (!parsed) {
        const buffer = await fetchPly(cellPlyUrl(id));
        parsed = parsePly(buffer);
        this.parsedCache.set(id, parsed);
      }
      // The manager may have been disposed (scene switch) while fetching.
      if (this._disposed) { this.parsedCache.delete(id); return; }
      const layer = new SplatLayer(this.app, this.pickLayer);
      layer.setParsed(parsed);
      layer.rotation.x = this.rotX;
      layer.sizeScale = this.sizeScale;
      layer.opacityScale = this.opacityScale;
      layer.depthMode = this.depthMode;
      await layer.setDensity(density);
      layer.setVisible(this.visible);
      this.layers.set(id, layer);
      this.densityOf.set(id, density);
    } catch (e) {
      console.warn(`[atlas] cell ${id} load failed:`, e);
    } finally {
      this.loading.delete(id);
    }
  }

  async _setCellDensity(id, density) {
    const layer = this.layers.get(id);
    if (!layer) return;
    const prev = this.densityOf.get(id) ?? -1;
    // Only rebuild on a meaningful change (rebuild = destroy+reload of the gsplat asset).
    if (Math.abs(prev - density) < 0.05) return;
    this.densityOf.set(id, density);
    await layer.setDensity(density);
  }

  _unloadCell(id) {
    const layer = this.layers.get(id);
    if (layer?.entity) layer.entity.destroy();
    if (layer?.objectUrl) URL.revokeObjectURL(layer.objectUrl);
    if (layer?.asset) this.app.assets.remove(layer.asset);
    this.layers.delete(id);
    this.densityOf.delete(id);
    this.parsedCache.delete(id);  // free the parsed SoA too (re-fetch on re-entry)
  }

  // Drive LOD from the app update loop. `getTargetPC()` returns the current
  // camera target in PC frame. Throttled internally.
  update(targetPC, nowMs) {
    if (this._disposed) return;
    if (nowMs - this._lastUpdate < UPDATE_EVERY_MS) return;
    this._lastUpdate = nowMs;
    this._target = targetPC.slice();

    // Unload far cells first (frees the budget for nearer loads).
    for (const id of [...this.layers.keys()]) {
      const cell = this.cells.find((c) => c.cell_id === id);
      if (!cell) continue;
      const d = this._dist(cell.centerPC, targetPC);
      if (d > UNLOAD_RADIUS_M) this._unloadCell(id);
    }

    // Load / re-LOD cells within the load radius, nearest first.
    const inRange = this.cells
      .map((c) => ({ c, d: this._dist(c.centerPC, targetPC) }))
      .filter(({ d }) => d <= LOAD_RADIUS_M)
      .sort((a, b) => a.d - b.d);

    for (const { c, d } of inRange) {
      const density = this._densityForDist(d);
      if (this.layers.has(c.cell_id)) {
        // fire-and-forget density adjust (no await in the update loop)
        this._setCellDensity(c.cell_id, density);
      } else if (!this.loading.has(c.cell_id) && this.layers.size < MAX_LOADED_CELLS) {
        this._ensureCell(c, density);
      }
    }
  }

  // ---- bulk parameter passthroughs (settings panel) ----
  setVisible(v) {
    this.visible = v;
    for (const l of this.layers.values()) l.setVisible(v);
  }
  setSizeScale(s) {
    this.sizeScale = s;
    for (const l of this.layers.values()) l.setSizeScale(s);
  }
  setOpacityScale(s) {
    this.opacityScale = s;
    for (const l of this.layers.values()) l.setOpacityScale(s);
  }
  setRotationX(deg) {
    this.rotX = deg;
    for (const l of this.layers.values()) l.setRotationX(deg);
  }
  setDepthMode(mode) {
    this.depthMode = mode;
    for (const l of this.layers.values()) l.setDepthMode(mode);
  }

  dispose() {
    this._disposed = true;
    for (const id of [...this.layers.keys()]) this._unloadCell(id);
    this.layers.clear();
    this.parsedCache.clear();
    this.densityOf.clear();
  }
}
