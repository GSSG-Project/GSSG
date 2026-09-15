import math

import torch
import torch.nn.functional as F
from shapely.geometry import shape


class Object:
    def __init__(self, id, aabb, room_id, vector_id):
        self.id = id
        self.aabb = aabb
        self.room_id = room_id
        self.vector_id = vector_id

    @property
    def center(self):
        return (
            (self.aabb[0] + self.aabb[3]) / 2.0,
            (self.aabb[1] + self.aabb[4]) / 2.0,
            (self.aabb[2] + self.aabb[5]) / 2.0,
        )

    @property
    def volume(self):
        w = self.aabb[3] - self.aabb[0]
        h = self.aabb[4] - self.aabb[1]
        d = self.aabb[5] - self.aabb[2]
        return max(0.0, w * h * d)


class Room:
    def __init__(self, id, aabb=None, floor_id=None):
        self.id = id
        self.aabb = aabb
        self.floor_id = floor_id
        self.polygon = None
        self.embedding = None  # torch tensor


class Floor:
    def __init__(self, id, floor_height=None, room_ids=None):
        self.id = id
        self.floor_height = floor_height
        self.room_ids = room_ids or []


class SceneGraphInterface:
    def __init__(self, json_data, min_obj_volume=0.01, room_match_threshold=0.15):
        self.objects = []
        self.rooms = []
        self.floors = []
        self.min_obj_volume = min_obj_volume
        # Minimum cosine similarity for get_closest_room_id to accept a match.
        self.room_match_threshold = room_match_threshold
        self.setup(json_data)

    def get_object_by_id(self, object_id):
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def get_room_by_id(self, room_id):
        for room in self.rooms:
            if room.id == room_id:
                return room
        return None

    def get_closest_room_id(self, query_embedding):
        """Finds the room ID with the highest cosine similarity to the query embedding.
        Rooms without an embedding (barely visited / empty) are skipped."""
        embedded = [r for r in self.rooms if r.embedding is not None]
        if not embedded:
            return None

        room_embs = torch.stack([r.embedding for r in embedded])  # [N_rooms, Dim]

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.unsqueeze(0)

        scores = F.cosine_similarity(query_embedding, room_embs)
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()

        if best_score > self.room_match_threshold:
            return embedded[best_idx].id
        return None

    def _is_valid_aabb(self, aabb):
        return aabb is not None and isinstance(aabb, (list, tuple)) and len(aabb) == 6

    def setup(self, data):
        for f in data.get("floors", []):
            self.floors.append(Floor(f["id"], f.get("floor_height"), f.get("room_ids")))

        for r in data.get("rooms", []):
            # Rooms are polygon-first; the aabb is optional metadata (older exports had
            # none at all — never a reason to drop the room).
            aabb = r.get("aabb")
            room = Room(r["id"], aabb if self._is_valid_aabb(aabb) else None, r.get("floor_id"))

            if r.get("polygon"):
                # GeoJSON from shapely.mapping(); shape() also handles MultiPolygons.
                room.polygon = shape(r["polygon"])

            if "embedding" in r and r["embedding"]:
                # reshape(-1): older exports carry a leading batch dim.
                room.embedding = torch.tensor(r["embedding"], dtype=torch.float32).reshape(-1)

            self.rooms.append(room)

        for obj_data in data.get("objects", []):
            raw_aabb = obj_data.get("aabb")
            vector_id = obj_data.get("vector_id")
            if not self._is_valid_aabb(raw_aabb) or vector_id is None:
                continue

            w = raw_aabb[3] - raw_aabb[0]
            h = raw_aabb[4] - raw_aabb[1]
            d = raw_aabb[5] - raw_aabb[2]
            if w <= 0 or h <= 0 or d <= 0:
                continue

            volume = w * h * d
            if volume < self.min_obj_volume:
                continue

            new_obj = Object(obj_data["id"], raw_aabb, obj_data.get("room_id"), vector_id)
            self.objects.append(new_obj)


class VectorDBInterface:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def get_vector_and_object_id_by_vector_id(self, vector_id):
        return self.vector_db.get_vector(vector_id), self.vector_db.get_metadata(vector_id).get(
            "object_id"
        )

    def get_similar_vector_with_object_id_list(self, vector, k):
        if not isinstance(vector, torch.Tensor):
            vector = torch.from_numpy(vector)

        return [
            (vector_id, score, metadata.get("object_id"))
            for vector_id, score, metadata in self.vector_db.search(vector, k)
        ]


class SpatialIndexInterface:
    def __init__(self, spatial_index, scene_graph_interface):
        self.spatial_index = spatial_index
        self.scene_graph_interface = scene_graph_interface
        self.EPSILON = 0.15

    def get_min_distance_aabb(self, box1, box2):
        """Euclidean box-to-box distance between two AABBs (minx,miny,minz,maxx,maxy,maxz)."""
        dx = max(0, box1[0] - box2[3], box2[0] - box1[3])
        dy = max(0, box1[1] - box2[4], box2[1] - box1[4])
        dz = max(0, box1[2] - box2[5], box2[2] - box1[5])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def get_nearby_objects(self, object_id, radius: float, limit: int = 10):
        obj = self.scene_graph_interface.get_object_by_id(object_id)
        if not obj or not obj.aabb:
            return []

        x0, y0, z0, x1, y1, z1 = obj.aabb
        search_box = (x0 - radius, y0 - radius, z0 - radius, x1 + radius, y1 + radius, z1 + radius)

        results = []
        for cand_id in self.spatial_index.intersection(search_box):
            if cand_id == object_id:
                continue
            cand_obj = self.scene_graph_interface.get_object_by_id(cand_id)

            dist = self.get_min_distance_aabb(obj.aabb, cand_obj.aabb)

            if dist <= radius:
                results.append((cand_id, dist))

        return self._process_results(results, limit)

    def get_objects_on(self, support_object_id, limit: int = 10):
        support = self.scene_graph_interface.get_object_by_id(support_object_id)
        if not support or not support.aabb:
            return []

        sx_min, sy_min, sz_min, sx_max, sy_max, sz_max = support.aabb
        # Search the volume just above the support, up to 1 m.
        search_box = (
            sx_min - self.EPSILON,
            sy_min - self.EPSILON,
            sz_max - self.EPSILON,
            sx_max + self.EPSILON,
            sy_max + self.EPSILON,
            sz_max + 1.0,
        )
        results = []
        for cand_id in self.spatial_index.intersection(search_box):
            if cand_id == support_object_id:
                continue
            cand_obj = self.scene_graph_interface.get_object_by_id(cand_id)
            cx_min, cy_min, cz_min, cx_max, cy_max, cz_max = cand_obj.aabb

            vertical_dist = cz_min - sz_max
            is_vertically_aligned = abs(vertical_dist) < 0.2  # 20cm tolerance

            overlap_x = max(0, min(sx_max, cx_max) - max(sx_min, cx_min))
            overlap_y = max(0, min(sy_max, cy_max) - max(sy_min, cy_min))

            # "On" = vertically close to the support top and overlapping it in XY.
            if is_vertically_aligned and overlap_x > 0.05 and overlap_y > 0.05:
                results.append((cand_id, abs(vertical_dist)))

        return self._process_results(results, limit)

    def get_objects_inside(self, container_object_id, limit: int = 10):
        container = self.scene_graph_interface.get_object_by_id(container_object_id)
        if not container or not container.aabb:
            return []
        c_minx, c_miny, c_minz, c_maxx, c_maxy, c_maxz = container.aabb
        results = []

        for cand_id in self.spatial_index.intersection(container.aabb):
            if cand_id == container_object_id:
                continue
            cand = self.scene_graph_interface.get_object_by_id(cand_id)
            cx, cy, cz = cand.center
            if c_minx <= cx <= c_maxx and c_miny <= cy <= c_maxy and c_minz <= cz <= c_maxz:
                results.append((cand_id, 0.0))

        return self._process_results(results, limit)

    def get_objects_above(self, object_id, limit: int = 10):
        return self._get_vertical_relations(object_id, "above", limit)

    def get_objects_below(self, object_id, limit: int = 10):
        return self._get_vertical_relations(object_id, "below", limit)

    def get_distances_to_list(self, reference_object_id, candidate_object_ids, limit: int = 10):
        """Distances from the reference object to each candidate in the list."""
        ref_obj = self.scene_graph_interface.get_object_by_id(reference_object_id)
        if not ref_obj or not ref_obj.aabb:
            return []

        results = []
        for cand_id in candidate_object_ids:
            if cand_id == reference_object_id:
                continue

            cand_obj = self.scene_graph_interface.get_object_by_id(cand_id)
            if not cand_obj or not cand_obj.aabb:
                continue

            dist = self.get_min_distance_aabb(ref_obj.aabb, cand_obj.aabb)
            results.append((cand_id, dist))

        return self._process_results(results, limit)

    def _get_vertical_relations(self, object_id, direction, limit):
        ref = self.scene_graph_interface.get_object_by_id(object_id)
        if not ref or not ref.aabb:
            return []
        rx_min, ry_min, rz_min, rx_max, ry_max, rz_max = ref.aabb

        search_box = (rx_min, ry_min, -float("inf"), rx_max, ry_max, float("inf"))

        results = []
        for cand_id in self.spatial_index.intersection(search_box):
            if cand_id == object_id:
                continue
            cand = self.scene_graph_interface.get_object_by_id(cand_id)
            cx_min, cy_min, cz_min, cx_max, cy_max, cz_max = cand.aabb

            overlap_x = max(0, min(rx_max, cx_max) - max(rx_min, cx_min))
            overlap_y = max(0, min(ry_max, cy_max) - max(ry_min, cy_min))

            if overlap_x <= 0 or overlap_y <= 0:
                continue

            dist = 0
            if direction == "above" and cz_min >= rz_max:
                dist = cz_min - rz_max
                results.append((cand_id, dist))
            elif direction == "below" and cz_max <= rz_min:
                dist = rz_min - cz_max
                results.append((cand_id, dist))

        return self._process_results(results, limit)

    def _process_results(self, results, limit):
        results.sort(key=lambda x: x[1])
        if limit is not None and limit > 0:
            return results[:limit]
        return results
