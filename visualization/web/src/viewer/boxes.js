import * as pc from 'playcanvas';

// Renders AABB wireframes for the scene-graph objects every frame using
// PlayCanvas's drawWireAlignedBox immediate-mode helper.
//
// Declutter strategy (the readability fix for dense scenes):
//   - Non-selected boxes are drawn ONLY depth-tested, so the splat depth buffer
//     (depthMode 'soft'/'dither') hides boxes that sit behind walls/furniture.
//     No more see-through outlines piling up from every occluded object.
//   - Distance fade dims boxes far from the camera so near objects stand out.
//   - A "ghost" see-through outline is drawn ONLY for the selected/hovered box,
//     so the active object stays locatable even through walls.
//   - Display modes: 'all' | 'nearby' (around the camera target) | 'selected'.

const ROOM_COLORS = {
  null: new pc.Color(0.48, 0.5, 0.54, 0.9),
  1: new pc.Color(0.43, 0.76, 1.0, 0.95),
  2: new pc.Color(1.0, 0.48, 0.83, 0.95),
};
const SEL_COLOR = new pc.Color(1.0, 0.72, 0.3, 1.0);
const HOVER_COLOR = new pc.Color(1.0, 0.95, 0.55, 0.95);

function sq(x) { return x * x; }
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

export class BoxLayer {
  constructor(app, camera = null, orbit = null) {
    this.app = app;
    this.camera = camera;
    this.orbit = orbit;
    this.objects = [];
    this.visible = true;
    this.selectedId = null;
    this.hoverId = null;
    this.roomFilter = null; // null = all; otherwise Set<roomId|'none'>
    this.mode = 'all'; // 'all' | 'nearby' | 'selected'
    this.allById = null; // id -> object for ALL objects (incl. huge/hidden), for selection
    this.showGhost = true; // see-through outline for the selected/hovered box
    this.nearRadius = 6.0; // 'nearby' mode radius (PC units) around the target
    this.fadeStart = 8.0; // distance fade begins (PC units from camera)
    this.fadeEnd = 24.0; // fully faded-dim past this
    this._c = new pc.Color();
    this._g = new pc.Color();
    this._min = new pc.Vec3();
    this._max = new pc.Vec3();
    this._tickHandler = this.tick.bind(this);
    app.on('update', this._tickHandler);
  }

  setMode(m) { if (m === 'all' || m === 'nearby' || m === 'selected') this.mode = m; }
  setShowGhost(v) { this.showGhost = v; }
  setObjects(objects) { this.objects = objects; }
  setAllById(map) { this.allById = map; }
  setVisible(v) { this.visible = v; }
  setSelected(id) { this.selectedId = id; }
  setHover(id) { this.hoverId = id; }
  setRoomFilter(filter) { this.roomFilter = filter; }

  // Room-filtered base set.
  baseList() {
    if (!this.visible) return [];
    if (!this.roomFilter) return this.objects;
    return this.objects.filter((o) => this.roomFilter.has(o.roomId ?? 'none'));
  }

  displayList() {
    const base = this.baseList();
    let list;
    if (this.mode === 'selected') {
      list = base.filter((o) => o.id === this.selectedId || o.id === this.hoverId);
    } else if (this.mode === 'nearby' && this.orbit) {
      const t = this.orbit.getTarget();
      const r2 = this.nearRadius * this.nearRadius;
      list = base.filter(
        (o) =>
          o.id === this.selectedId ||
          o.id === this.hoverId ||
          sq(o.center[0] - t.x) + sq(o.center[1] - t.y) + sq(o.center[2] - t.z) <= r2,
      );
    } else {
      list = base;
    }
    // Always draw the selected/hovered box, even if it was hidden as a huge box
    // or deduped — otherwise a query result that lands on one shows nothing.
    const extra = [];
    for (const id of [this.selectedId, this.hoverId]) {
      if (id == null || list.some((o) => o.id === id)) continue;
      const o = this.allById && this.allById.get(id);
      if (o && o.min && o.max) extra.push(o);
    }
    return extra.length ? list.concat(extra) : list;
  }

  tick() {
    if (!this.visible) return;
    const list = this.displayList();
    const cam = this.camera ? this.camera.getPosition() : null;
    const min = this._min;
    const max = this._max;

    for (const o of list) {
      min.set(o.min[0], o.min[1], o.min[2]);
      max.set(o.max[0], o.max[1], o.max[2]);
      const emph = o.id === this.selectedId || o.id === this.hoverId;
      const src = o.id === this.selectedId
        ? SEL_COLOR
        : o.id === this.hoverId
          ? HOVER_COLOR
          : (ROOM_COLORS[o.roomId] || ROOM_COLORS.null);

      let a = src.a;
      if (!emph && cam) {
        const d = Math.sqrt(
          sq(o.center[0] - cam.x) + sq(o.center[1] - cam.y) + sq(o.center[2] - cam.z),
        );
        const f = clamp01(1 - (d - this.fadeStart) / (this.fadeEnd - this.fadeStart));
        a = src.a * (0.22 + 0.78 * f); // near = full, far = faint (never fully gone)
      }
      this._c.set(src.r, src.g, src.b, a);

      // Depth-tested: the splat depth buffer occludes boxes behind geometry.
      this.app.drawWireAlignedBox(min, max, this._c, true);

      if (emph) {
        // See-through ghost + halo so the active object reads through walls.
        if (this.showGhost) {
          this._g.set(src.r, src.g, src.b, 0.5);
          this.app.drawWireAlignedBox(min, max, this._g, false);
        }
        const pad = 0.02;
        min.x -= pad; min.y -= pad; min.z -= pad;
        max.x += pad; max.y += pad; max.z += pad;
        this.app.drawWireAlignedBox(min, max, src, false);
      }
    }
  }

  dispose() {
    this.app.off('update', this._tickHandler);
  }
}
