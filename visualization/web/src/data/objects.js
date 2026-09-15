// Normalize a scene-graph JSON (from /api/scene/scene_graph) into the
// PlayCanvas Y-up frame so it aligns with the splat entity.

import { pointReplicaToPC, aabbReplicaToPC, sizeReplicaToPC } from './frame.js?v=3';
import { fetchSceneGraphJson } from './api.js?v=3';

export async function fetchSceneGraph() {
  const raw = await fetchSceneGraphJson();

  const objects = (raw.objects || [])
    .filter((o) => o && o.aabb && o.center)
    .map((o) => {
      const [x0, y0, z0, x1, y1, z1] = o.aabb;
      const rawMin = [Math.min(x0, x1), Math.min(y0, y1), Math.min(z0, z1)];
      const rawMax = [Math.max(x0, x1), Math.max(y0, y1), Math.max(z0, z1)];
      const rawSize = [Math.abs(x1 - x0), Math.abs(y1 - y0), Math.abs(z1 - z0)];
      const { min, max } = aabbReplicaToPC(rawMin, rawMax);
      return {
        id: o.id,
        roomId: o.room_id ?? null,
        center: pointReplicaToPC(o.center),
        min,
        max,
        size: sizeReplicaToPC(rawSize),
        vectorId: o.vector_id ?? null,
      };
    });

  const rooms = (raw.rooms || []).map((r) => ({
    id: r.id,
    floorId: r.floor_id ?? 0,
    polygon: r.polygon,
  }));

  const floors = Array.from(new Set(rooms.map((r) => r.floorId))).sort((a, b) => a - b);

  return { objects, rooms, floors };
}