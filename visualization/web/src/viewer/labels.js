import * as pc from 'playcanvas';

// Projects object centers to screen space and positions absolute-positioned
// label callouts. Throttled to ~20 Hz.

export class LabelLayer {
  constructor(app, camera, container, objects) {
    this.app = app;
    this.camera = camera;
    this.container = container;
    this.objects = objects;
    this.visible = true;
    this.selectedId = null;
    this.hoverId = null;
    this.roomFilter = null;
    this.maxLabels = 24; // performance cap
    this._tmpWorld = new pc.Vec3();
    this._tmpScreen = new pc.Vec3();
    this._lastTick = 0;
    this._tick = this._tick.bind(this);
    app.on('update', this._tick);
    this.elements = new Map(); // id → div
    this._build();
  }

  _build() {
    this.container.innerHTML = '';
    this.elements.clear();
    for (const o of this.objects) {
      const el = document.createElement('div');
      el.className = 'lbl-callout';
      el.style.display = 'none';
      el.innerHTML = `<span class="lc-pin"></span><span class="lc-text">obj <b>#${o.id}</b><small>${o.roomId == null ? 'r—' : 'r' + o.roomId}</small></span>`;
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        const ev = new CustomEvent('label-click', { detail: { id: o.id } });
        this.container.dispatchEvent(ev);
      });
      this.container.appendChild(el);
      this.elements.set(o.id, el);
    }
  }

  setVisible(v) { this.visible = v; }
  setSelected(id) { this.selectedId = id; }
  setHover(id) { this.hoverId = id; }
  setRoomFilter(filter) { this.roomFilter = filter; }

  setObjects(objects) {
    this.objects = objects;
    this._build();
  }

  _shouldShow(o) {
    if (!this.visible) return false;
    if (this.roomFilter && !this.roomFilter.has(o.roomId ?? 'none')) return false;
    return true;
  }

  _tick(dt) {
    const now = performance.now();
    if (now - this._lastTick < 50) return;
    this._lastTick = now;

    const camComp = this.camera.camera;
    const camPos = this.camera.getPosition();
    const cw = this.container.clientWidth;
    const ch = this.container.clientHeight;

    // Pick the N nearest visible objects to the camera; hide everything else.
    const candidates = this.objects.filter((o) => this._shouldShow(o));
    candidates.sort((a, b) => {
      const da = sq(a.center[0] - camPos.x) + sq(a.center[1] - camPos.y) + sq(a.center[2] - camPos.z);
      const db = sq(b.center[0] - camPos.x) + sq(b.center[1] - camPos.y) + sq(b.center[2] - camPos.z);
      return da - db;
    });

    const visibleSet = new Set();
    let shown = 0;
    for (const o of candidates) {
      if (shown >= this.maxLabels) break;
      this._tmpWorld.set(o.center[0], o.center[1], o.center[2]);
      camComp.worldToScreen(this._tmpWorld, this._tmpScreen);
      const sx = this._tmpScreen.x;
      const sy = this._tmpScreen.y;
      const sz = this._tmpScreen.z;
      if (sz < 0 || sx < -50 || sy < -50 || sx > cw + 50 || sy > ch + 50) continue;
      visibleSet.add(o.id);
      const el = this.elements.get(o.id);
      if (!el) continue;
      el.style.left = `${sx}px`;
      el.style.top = `${sy}px`;
      el.style.display = '';
      el.classList.toggle('sel', o.id === this.selectedId);
      el.classList.toggle('hov', o.id === this.hoverId);
      shown++;
    }
    for (const [id, el] of this.elements) {
      if (!visibleSet.has(id)) el.style.display = 'none';
    }
  }

  dispose() {
    this.app.off('update', this._tick);
    this.container.innerHTML = '';
  }
}

function sq(x) { return x * x; }
