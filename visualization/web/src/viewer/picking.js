import * as pc from 'playcanvas';

// Depth-aware GPU picker.
//
// pc.Picker re-renders the supplied layer to a small offscreen RT using
// each material's PICK_PASS shader variant, then reads a pixel. The gsplat
// PICK_PASS does an alphaClip + depth-write per splat, so the depth buffer
// in the pick RT correctly reflects the front splat surface. Proxy cube
// meshes (rendered after, depth-tested) are rejected wherever splats are
// in front of them — so the pick only resolves to AABBs that are actually
// visible.
//
// We keep ray-vs-AABB as a fallback when GPU picking isn't available
// (e.g. some headless/software-WebGL contexts).

export class Picker {
  constructor(app, camera, canvas, pickLayer, proxies) {
    this.app = app;
    this.camera = camera;
    this.canvas = canvas;
    this.pickLayer = pickLayer;
    this.proxies = proxies;
    // 256² is small enough to be cheap, big enough to be precise around
    // the cursor. The Picker resizes internally too.
    this.pickRes = 256;
    this.gpu = null;
    this._tryInitGpu();
    // Ray-vs-AABB fallback bits.
    this._fallbackBoxes = [];
    this._ray = new pc.Ray();
    this._hitPoint = new pc.Vec3();
    this._preparedAt = 0;
    this._prepareCooldownMs = 30; // re-prepare at ≤ ~33 Hz
  }

  _tryInitGpu() {
    try {
      this.gpu = new pc.Picker(this.app, this.pickRes, this.pickRes);
    } catch (e) {
      console.warn('GPU picker init failed, falling back to ray-AABB:', e);
      this.gpu = null;
    }
  }

  setObjects(objects) {
    this._fallbackBoxes = objects.map((o) => {
      const min = new pc.Vec3(o.min[0], o.min[1], o.min[2]);
      const max = new pc.Vec3(o.max[0], o.max[1], o.max[2]);
      const half = new pc.Vec3((max.x - min.x) / 2, (max.y - min.y) / 2, (max.z - min.z) / 2);
      const center = new pc.Vec3((max.x + min.x) / 2, (max.y + min.y) / 2, (max.z + min.z) / 2);
      return { id: o.id, bbox: new pc.BoundingBox(center, half) };
    });
  }

  pick(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;

    if (this.gpu) {
      try {
        const id = this._pickGpu(sx, sy, rect.width, rect.height);
        return id;
      } catch (e) {
        console.warn('GPU pick failed, falling back to ray:', e);
        this.gpu = null;
      }
    }
    return this._pickRay(sx, sy);
  }

  _pickGpu(sx, sy, canvasW, canvasH) {
    // Re-prepare only when needed — prepare runs a full pick-layer render.
    const now = performance.now();
    if (now - this._preparedAt >= this._prepareCooldownMs) {
      this.gpu.prepare(this.camera.camera, this.app.scene, [this.pickLayer]);
      this._preparedAt = now;
    }
    // Map canvas coords → pick-RT coords (top-left origin both).
    const px = Math.floor((sx / canvasW) * this.pickRes);
    const py = Math.floor((sy / canvasH) * this.pickRes);
    if (px < 0 || py < 0 || px >= this.pickRes || py >= this.pickRes) return null;
    const selection = this.gpu.getSelection(px, py, 1, 1);
    if (!selection || !selection.length) return null;
    const hit = selection[0];
    // hit may be a MeshInstance (our proxy) or a GSplatComponent (the splat
    // was the topmost thing under the cursor → treat as "no object").
    if (hit instanceof pc.GSplatComponent) return null;
    return this.proxies.idForMeshInstance(hit) ?? null;
  }

  _pickRay(sx, sy) {
    const camComp = this.camera.camera;
    const nearW = new pc.Vec3();
    const farW = new pc.Vec3();
    camComp.screenToWorld(sx, sy, camComp.nearClip, nearW);
    camComp.screenToWorld(sx, sy, camComp.farClip, farW);
    this._ray.origin.copy(nearW);
    this._ray.direction.copy(farW).sub(nearW).normalize();
    let bestId = null;
    let bestT = Infinity;
    const origin = this._ray.origin;
    for (const { id, bbox } of this._fallbackBoxes) {
      if (bbox.intersectsRay(this._ray, this._hitPoint)) {
        const dx = this._hitPoint.x - origin.x;
        const dy = this._hitPoint.y - origin.y;
        const dz = this._hitPoint.z - origin.z;
        const t = dx * dx + dy * dy + dz * dz;
        if (t < bestT) { bestT = t; bestId = id; }
      }
    }
    return bestId;
  }

  invalidate() {
    // Force re-prepare on next pick (e.g. when scene contents change).
    this._preparedAt = 0;
  }

  dispose() {
    this.gpu?.destroy?.();
    this.gpu = null;
  }
}
