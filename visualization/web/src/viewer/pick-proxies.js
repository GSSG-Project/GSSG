import * as pc from 'playcanvas';

// Invisible solid-cube mesh entities, one per AABB. These are NOT in the
// camera's visible composition — they live only in pickLayer, which we
// hand to pc.Picker. Visible AABB rendering still uses immediate-mode
// wireframes from boxes.js, so per-frame selection coloring stays trivial.
//
// The pick-pass renders the splats in pickLayer first (with alpha-clip +
// depth-write, via the gsplat shader's PICK_PASS branch). Proxy cubes are
// then depth-tested against that depth, so cubes behind splats get
// rejected and never appear in the resulting pick map. Net effect: a
// hover/click only resolves to an object if its AABB is actually visible.

export class PickProxies {
  constructor(app, pickLayer) {
    this.app = app;
    this.pickLayer = pickLayer;
    this.entities = [];
    this.byMeshInstance = new Map(); // MeshInstance → object.id
    this.byId = new Map();           // object.id → Entity
    // Shared standard material — picker uses its own shader anyway, so the
    // visible material is irrelevant. We just need a real mesh + material.
    this._material = new pc.StandardMaterial();
    this._material.useLighting = false;
    this._material.update();
  }

  setObjects(objects) {
    this.dispose();
    for (const o of objects) {
      const e = new pc.Entity(`pick_${o.id}`);
      e.addComponent('render', {
        type: 'box',
        material: this._material,
        layers: [this.pickLayer.id],
        castShadows: false,
        receiveShadows: false,
      });
      const cx = (o.min[0] + o.max[0]) / 2;
      const cy = (o.min[1] + o.max[1]) / 2;
      const cz = (o.min[2] + o.max[2]) / 2;
      const sx = Math.max(o.max[0] - o.min[0], 0.001);
      const sy = Math.max(o.max[1] - o.min[1], 0.001);
      const sz = Math.max(o.max[2] - o.min[2], 0.001);
      e.setLocalPosition(cx, cy, cz);
      e.setLocalScale(sx, sy, sz);
      this.app.root.addChild(e);
      this.entities.push(e);
      this.byId.set(o.id, e);
      for (const mi of e.render.meshInstances) {
        this.byMeshInstance.set(mi, o.id);
      }
    }
  }

  idForMeshInstance(mi) {
    return this.byMeshInstance.get(mi) ?? null;
  }

  dispose() {
    for (const e of this.entities) e.destroy();
    this.entities = [];
    this.byMeshInstance.clear();
    this.byId.clear();
  }
}
