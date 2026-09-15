// AABB IoU + greedy NMS + containment + max-volume filter.
//
// Used to clean up the rendered set of bounding boxes. The original objects
// list is preserved on the caller side (backend can still see all of them);
// this only produces:
//
//   visible        — the subset to draw / label / pick
//   canonicalById  — id → canonical-visible id. Every input id (visible,
//                    absorbed, AND huge) is keyed. Visible/huge map to
//                    themselves; absorbed ones map to their absorber.
//                    External callers (CLIP / LLM query results) use this
//                    to redirect a hidden id to the visible representative.
//   huge           — IDs that were filtered out for being too large (walls,
//                    floors, room-sized AABBs). Inspector can still look
//                    them up if they're in objectMap; their boxes are just
//                    not drawn.
//
// Filter order (important):
//   1. Drop objects with volume > maxVolume. These are usually walls/floors
//      and would, if used as NMS anchors, absorb essentially everything via
//      the containment check.
//   2. NMS over what remains, sorted by volume desc, id asc tiebreak.
//      An object is absorbed if any kept object either (a) has IoU ≥
//      iouThreshold with it, OR (b) fully contains its AABB.

export function aabbIoU(a, b) {
  const ix = Math.max(0, Math.min(a.max[0], b.max[0]) - Math.max(a.min[0], b.min[0]));
  const iy = Math.max(0, Math.min(a.max[1], b.max[1]) - Math.max(a.min[1], b.min[1]));
  const iz = Math.max(0, Math.min(a.max[2], b.max[2]) - Math.max(a.min[2], b.min[2]));
  const inter = ix * iy * iz;
  if (inter <= 0) return 0;
  const va = aabbVolume(a);
  const vb = aabbVolume(b);
  const u = va + vb - inter;
  return u > 0 ? inter / u : 0;
}

export function aabbVolume(o) {
  return Math.max(0, o.max[0] - o.min[0]) *
         Math.max(0, o.max[1] - o.min[1]) *
         Math.max(0, o.max[2] - o.min[2]);
}

// True if `outer.aabb` fully contains `inner.aabb`. A small slack (1 mm)
// absorbs floating-point noise from the world-to-PC frame transform.
export function aabbContains(outer, inner, slack = 1e-3) {
  return outer.min[0] - slack <= inner.min[0] &&
         outer.min[1] - slack <= inner.min[1] &&
         outer.min[2] - slack <= inner.min[2] &&
         outer.max[0] + slack >= inner.max[0] &&
         outer.max[1] + slack >= inner.max[1] &&
         outer.max[2] + slack >= inner.max[2];
}

export function dedupByIoU(objects, options = {}) {
  const iouThreshold = options.iouThreshold ?? 0.6;
  const maxVolume = options.maxVolume ?? Infinity;

  // 1. Pull out the huge ones — walls / floors / room-shells. They never
  // get drawn; if we used them as NMS anchors, the containment check below
  // would swallow every smaller object inside the room.
  const huge = [];
  const candidates = [];
  for (const o of objects) {
    if (aabbVolume(o) > maxVolume) huge.push(o);
    else candidates.push(o);
  }

  // 2. NMS over candidates, volume desc so larger anchors absorb smaller
  // overlapping / contained boxes (and not the other way around).
  const sorted = [...candidates].sort((a, b) => {
    const dv = aabbVolume(b) - aabbVolume(a);
    if (dv !== 0) return dv;
    return a.id - b.id;
  });

  const visible = [];
  const canonicalById = new Map();

  for (const obj of sorted) {
    let canonical = null;
    for (const kept of visible) {
      if (aabbIoU(obj, kept) >= iouThreshold || aabbContains(kept, obj)) {
        canonical = kept;
        break;
      }
    }
    if (canonical) {
      canonicalById.set(obj.id, canonical.id);
    } else {
      visible.push(obj);
      canonicalById.set(obj.id, obj.id);
    }
  }

  // Huge ones map to themselves: not drawn, but inspector + status bar
  // still work if something selects them by id (e.g. an LLM query).
  for (const o of huge) canonicalById.set(o.id, o.id);

  return { visible, canonicalById, huge };
}