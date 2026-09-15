import * as pc from 'playcanvas';
import { createOrbitControls } from './orbit.js?v=4';

export function createViewer(canvas) {
  // Note: we deliberately do NOT pass pc.Mouse/Keyboard — orbit controls use
  // raw pointer events on the canvas, and adding pc input handlers would
  // duplicate context-menu preventDefault and consume events we want.
  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: {
      antialias: true,
      alpha: false,
      preferWebGl2: true,
      powerPreference: 'high-performance',
    },
  });

  app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);

  // Picking layer: not part of the camera's visible composition. Holds the
  // splat (also assigned to World) plus invisible AABB proxy meshes. We
  // hand this layer to pc.Picker so its render pass rasterises the splats
  // (PICK_PASS shader, alpha-clip + depth) before the box proxies — boxes
  // occluded by splats get depth-rejected and never appear in the pick map.
  const pickLayer = new pc.Layer({
    name: 'pickLayer',
    opaqueSortMode: pc.SORTMODE_NONE,
    transparentSortMode: pc.SORTMODE_NONE,
  });
  app.scene.layers.push(pickLayer);

  // Camera entity — Y-up world.
  const camera = new pc.Entity('camera');
  camera.addComponent('camera', {
    clearColor: new pc.Color(0.04, 0.05, 0.06),
    nearClip: 0.05,
    farClip: 200,
    fov: 50,
    toneMapping: pc.TONEMAP_LINEAR,
    gammaCorrection: pc.GAMMA_SRGB,
  });
  app.root.addChild(camera);

  const orbit = createOrbitControls(app, camera, canvas);
  camera.setPosition(4, 3, 6);
  orbit.setTarget(new pc.Vec3(0, 0, 0));

  // Ambient + fill light so non-splat objects (boxes) look OK.
  app.scene.ambientLight = new pc.Color(0.4, 0.4, 0.45);

  // Resize handling — fillMode handles canvas size, but we want crispness on HiDPI.
  const resize = () => app.resizeCanvas();
  window.addEventListener('resize', resize);

  app.start();

  return { app, camera, orbit, pickLayer, dispose: () => { window.removeEventListener('resize', resize); app.destroy(); } };
}

// Smoothly move camera to look at a bounding box.
export function frameBounds(camera, orbit, min, max, padding = 1.4) {
  const cx = (min[0] + max[0]) / 2;
  const cy = (min[1] + max[1]) / 2;
  const cz = (min[2] + max[2]) / 2;
  const dx = max[0] - min[0];
  const dy = max[1] - min[1];
  const dz = max[2] - min[2];
  const diag = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const target = new pc.Vec3(cx, cy, cz);
  orbit.setTarget(target);
  orbit.setDistance(Math.max(0.6, diag * padding));
}

// Set an explicit camera position + look-at target. Coordinates are in PC
// world space (Y-up) — same frame the on-screen HUD reports. Derives
// yaw/pitch/distance from (position − target) for the orbit controls.
export function setView(orbit, position, target) {
  const dx = position[0] - target[0];
  const dy = position[1] - target[1];
  const dz = position[2] - target[2];
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (dist < 1e-4) return;
  const lim = Math.PI / 2 - 0.05;
  const yaw = Math.atan2(dx, dz);
  let pitch = Math.asin(Math.max(-1, Math.min(1, dy / dist)));
  if (pitch > lim) pitch = lim;
  if (pitch < -lim) pitch = -lim;
  orbit.setTarget(new pc.Vec3(target[0], target[1], target[2]));
  orbit.setDistance(dist);
  orbit.setYawPitch(yaw, pitch);
}
