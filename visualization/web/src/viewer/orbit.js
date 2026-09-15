import * as pc from 'playcanvas';

// Minimal orbit camera: LMB orbit, RMB/Shift-LMB pan, wheel dolly.
// Yaw/pitch in radians; distance in world units.

export function createOrbitControls(app, cameraEntity, canvas) {
  let yaw = 0;
  let pitch = -0.35;
  let distance = 8;
  const target = new pc.Vec3(0, 0, 0);

  // Make the viewport focusable so clicking it takes keyboard focus away from
  // the query box — otherwise the isTyping() guard below swallows every key.
  canvas.tabIndex = -1;
  canvas.style.outline = 'none';

  let mode = null; // 'orbit' | 'pan' | null
  let lastX = 0;
  let lastY = 0;

  const update = () => {
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);
    const x = target.x + distance * cp * sy;
    const y = target.y + distance * sp;
    const z = target.z + distance * cp * cy;
    cameraEntity.setPosition(x, y, z);
    cameraEntity.lookAt(target);
  };

  canvas.addEventListener('pointerdown', (e) => {
    canvas.focus(); // grab keyboard focus from any text field
    canvas.setPointerCapture(e.pointerId);
    if (e.button === 2 || (e.button === 0 && e.shiftKey)) mode = 'pan';
    else if (e.button === 0) mode = 'orbit';
    else if (e.button === 1) mode = 'pan';
    lastX = e.clientX;
    lastY = e.clientY;
    e.preventDefault();
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!mode) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (mode === 'orbit') {
      yaw -= dx * 0.005;
      pitch -= dy * 0.005;
      const lim = Math.PI / 2 - 0.05;
      if (pitch > lim) pitch = lim;
      if (pitch < -lim) pitch = -lim;
    } else if (mode === 'pan') {
      // Move target along camera's right/up axes.
      const right = cameraEntity.right.clone();
      const up = cameraEntity.up.clone();
      const k = distance * 0.0018;
      target.x -= right.x * dx * k;
      target.y -= right.y * dx * k;
      target.z -= right.z * dx * k;
      target.x += up.x * dy * k;
      target.y += up.y * dy * k;
      target.z += up.z * dy * k;
    }
    update();
  });

  const endPointer = (e) => {
    mode = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  canvas.addEventListener('wheel', (e) => {
    const f = Math.exp(e.deltaY * 0.0015);
    distance = Math.max(0.4, Math.min(80, distance * f));
    update();
    e.preventDefault();
  }, { passive: false });

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  // ── keyboard navigation (antimatter15 splat-viewer scheme) ──────────────────
  //   arrows  : strafe (left/right) + move forward/back (up/down)
  //   space   : move up      shift : move down
  //   w/s,i/k : tilt (pitch)        a/d,j/l : turn (yaw)
  //   q/e     : zoom out/in (an orbit camera has no roll)
  // Movement scales with zoom distance and frame time; keys are ignored while a
  // text field is focused so the query box keeps working.
  const keys = new Set();
  const MOVE_KEYS = new Set([
    'arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' ', 'spacebar',
  ]);
  const isTyping = () => {
    const a = document.activeElement;
    return !!a && (a.tagName === 'INPUT' || a.tagName === 'TEXTAREA' || a.isContentEditable);
  };
  window.addEventListener('keydown', (e) => {
    if (isTyping()) return;
    const k = e.key.toLowerCase();
    keys.add(k);
    if (MOVE_KEYS.has(k)) e.preventDefault();
  });
  window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener('blur', () => keys.clear());

  const fwd = new pc.Vec3();
  const right = new pc.Vec3();
  app.on('update', (dt) => {
    if (keys.size === 0) return;
    if (mode) return; // a mouse drag is in progress (shift-pan etc.) — don't fight it
    const d = Math.min(dt || 0.016, 0.05);
    const move = distance * 0.5 * d;
    const rot = 0.8 * d;
    const has = (k) => keys.has(k);

    // Ground-plane camera basis (so forward/strafe walk the floor; space = up).
    fwd.copy(cameraEntity.forward); fwd.y = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1); else fwd.normalize();
    right.copy(cameraEntity.right); right.y = 0;
    if (right.lengthSq() < 1e-6) right.set(1, 0, 0); else right.normalize();

    let moved = false;
    if (has('arrowup')) { target.x += fwd.x * move; target.y += fwd.y * move; target.z += fwd.z * move; moved = true; }
    if (has('arrowdown')) { target.x -= fwd.x * move; target.y -= fwd.y * move; target.z -= fwd.z * move; moved = true; }
    if (has('arrowright')) { target.x += right.x * move; target.y += right.y * move; target.z += right.z * move; moved = true; }
    if (has('arrowleft')) { target.x -= right.x * move; target.y -= right.y * move; target.z -= right.z * move; moved = true; }
    if (has(' ') || has('spacebar')) { target.y += move; moved = true; }
    if (has('shift')) { target.y -= move; moved = true; }

    let dy = 0;
    let dp = 0;
    if (has('a') || has('j')) dy += rot;
    if (has('d') || has('l')) dy -= rot;
    if (has('w') || has('i')) dp += rot;
    if (has('s') || has('k')) dp -= rot;
    if (dy || dp) {
      yaw += dy;
      pitch += dp;
      const lim = Math.PI / 2 - 0.05;
      if (pitch > lim) pitch = lim;
      if (pitch < -lim) pitch = -lim;
      moved = true;
    }
    if (has('q')) { distance = Math.min(80, distance * Math.exp(0.7 * d)); moved = true; }
    if (has('e')) { distance = Math.max(0.4, distance * Math.exp(-0.7 * d)); moved = true; }

    if (moved) update();
  });

  update();

  return {
    setTarget(v) { target.copy(v); update(); },
    setDistance(d) { distance = d; update(); },
    setYawPitch(y, p) { yaw = y; pitch = p; update(); },
    getTarget() { return target.clone(); },
    getDistance() { return distance; },
  };
}
