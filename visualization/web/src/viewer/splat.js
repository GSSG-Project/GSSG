import * as pc from 'playcanvas';
import { buildCleanPly } from '../data/ply.js?v=3';
import { REPLICA_TO_PC_EULER_X } from '../data/frame.js?v=3';

// Wraps a single gsplat entity. Supports hot-swap of the asset when the
// density slider changes — we destroy the prior asset/entity and rebuild.

export class SplatLayer {
  constructor(app, pickLayer = null) {
    this.app = app;
    this.pickLayer = pickLayer;
    this.entity = null;
    this.asset = null;
    this.objectUrl = null;
    this.parsed = null;
    this.currentCount = 0;
    this.sizeScale = 1.0;
    this.opacityScale = 1.0;
    // Replica is Z-up; PlayCanvas is Y-up. Bring the splat into PC's frame
    // with a -90° X rotation. Object AABBs are pre-transformed in objects.js
    // so they sit in the same frame without extra ceremony.
    this.rotation = new pc.Vec3(REPLICA_TO_PC_EULER_X, 0, 0);
    // Depth modes:
    //   'blend'  — alpha-blended, no depthWrite. Beautiful splats, no line occlusion.
    //   'soft'   — alpha-blended + depthWrite=true. Smooth look, lines clipped behind splats.
    //   'dither' — stochastic alpha + depthWrite=true. Most accurate depth, gritty look.
    this.depthMode = 'soft';
    this.alphaClip = 0.3;
  }

  setParsed(parsed) {
    this.parsed = parsed;
  }

  async setDensity(fraction) {
    if (!this.parsed) return;
    const total = this.parsed.count;
    const want = Math.max(1000, Math.floor(total * fraction));
    if (want === this.currentCount) return;
    await this.rebuild(want);
  }

  async rebuild(numSplats) {
    if (!this.parsed) return;
    this.currentCount = numSplats;
    // Size/opacity are baked into the rebuilt PLY (the stock gsplat shader has
    // no uniform for them), so the sliders take effect on rebuild.
    const blob = buildCleanPly(this.parsed, numSplats, {
      sizeScale: this.sizeScale,
      opacityScale: this.opacityScale,
    });
    const newUrl = URL.createObjectURL(blob);
    const filename = `splats_${numSplats}.ply`;

    // Build new asset.
    const asset = new pc.Asset(filename, 'gsplat', { url: newUrl, filename });
    await new Promise((resolve, reject) => {
      asset.once('load', resolve);
      asset.once('error', (err) => reject(new Error(err || 'gsplat asset load error')));
      this.app.assets.add(asset);
      this.app.assets.load(asset);
    });

    // Build new entity and swap.
    const next = new pc.Entity('splats');
    next.addComponent('gsplat', { asset });
    // Put the splat in BOTH world (visible) and pick layers (so the GPU
    // picker can use it as a depth occluder).
    if (this.pickLayer) {
      next.gsplat.layers = [pc.LAYERID_WORLD, this.pickLayer.id];
    }
    next.setLocalEulerAngles(this.rotation.x, this.rotation.y, this.rotation.z);
    this.app.root.addChild(next);

    this.applyMaterialParams(next);

    if (this.entity) {
      this.entity.destroy();
    }
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    if (this.asset) this.app.assets.remove(this.asset);

    this.entity = next;
    this.asset = asset;
    this.objectUrl = newUrl;
  }

  applyMaterialParams(entity) {
    // entity.gsplat.instance.material only exists once the asset finishes
    // loading — the previous code applied once and gave up if it wasn't ready,
    // so the occlusion mode (depthWrite/blend) silently never applied. Retry on
    // subsequent frames until the material exists.
    const apply = () => {
      const mat = entity.gsplat?.instance?.material;
      if (!mat) return false;
      this.applyDepthMode(mat); // blendType + depthWrite — the real occlusion control
      mat.update?.();
      return true;
    };
    if (apply()) return;
    const onUpd = () => {
      if (!entity.gsplat || apply()) this.app.off('update', onUpd);
    };
    this.app.on('update', onUpd);
  }

  applyDepthMode(mat) {
    // Three render modes — see this.depthMode above for the trade-offs.
    if (this.depthMode === 'dither') {
      mat.setDefine('DITHER_NONE', null);
      mat.setDefine('DITHER_BLUENOISE', '');
      mat.blendType = pc.BLEND_NONE;
      mat.depthWrite = true;
    } else if (this.depthMode === 'soft') {
      mat.setDefine('DITHER_BLUENOISE', null);
      mat.setDefine('DITHER_NONE', '');
      mat.blendType = pc.BLEND_PREMULTIPLIED;
      // Alpha-blended splats are sorted back-to-front, so the closest
      // splat at each pixel wins the depth write. Lines drawn afterwards
      // get clipped correctly. Transparent splat edges DO write depth too;
      // for dense indoor scans the artifact is usually invisible.
      mat.depthWrite = true;
    } else {
      // 'blend' — original 3DGS look, no occlusion.
      mat.setDefine('DITHER_BLUENOISE', null);
      mat.setDefine('DITHER_NONE', '');
      mat.blendType = pc.BLEND_PREMULTIPLIED;
      mat.depthWrite = false;
    }
    mat.setParameter('alphaClip', this.alphaClip);
  }

  setDepthMode(mode) {
    if (mode !== 'blend' && mode !== 'soft' && mode !== 'dither') return;
    this.depthMode = mode;
    if (this.entity) this.applyMaterialParams(this.entity);
  }

  setAlphaClip(v) {
    this.alphaClip = v;
    if (this.entity) this.applyMaterialParams(this.entity);
  }

  setSizeScale(s) { this.sizeScale = s; }
  setOpacityScale(s) { this.opacityScale = s; }

  // Rebuild at the current density to bake in the latest size/opacity values.
  async refresh() {
    if (!this.parsed) return;
    await this.rebuild(this.currentCount || this.parsed.count);
  }

  setVisible(v) {
    if (this.entity) this.entity.enabled = v;
  }

  setRotationX(deg) {
    this.rotation.x = deg;
    if (this.entity) this.entity.setLocalEulerAngles(this.rotation.x, this.rotation.y, this.rotation.z);
  }
}
