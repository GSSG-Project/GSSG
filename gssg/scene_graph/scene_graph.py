import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from rtree import index
from shapely import Polygon
from shapely.geometry import Point

from gssg.scene_graph.scene_graph_layers import SGFloor, SGObject, SGRoom
from gssg.scene_graph.vector_db import FaissHNSWVectorDB
from gssg.utils.utils import to_numpy

# Image-embedding dim per known CLIP backbone, used to size the FAISS index at
# construction time (must match what encode_image returns).
CLIP_FEATURE_DIMS = {
    "ViT-B-16": 512,
    "ViT-B-32": 512,
    "ViT-L-14": 768,
    "ViT-H-14": 1024,
    "MobileCLIP-S0": 512,
    "MobileCLIP-S1": 512,
    "MobileCLIP-S2": 512,
    "MobileCLIP-B": 512,
    "MobileCLIP2-S0": 512,
    "MobileCLIP2-S1": 512,
    "MobileCLIP2-S2": 512,
    "MobileCLIP2-B": 512,
    "MobileCLIP-S3": 768,
    "MobileCLIP-S4": 768,
    "MobileCLIP2-S3": 768,
    "MobileCLIP2-S4": 768,
}


class SceneGraph:
    def __init__(self, args, scene="gssg"):
        self.args = args
        self.scene = scene
        # Explicit override > lookup table > default (1024 for H-14, else 768).
        dim = getattr(args, "clip_dim", None)
        if not dim:
            dim = CLIP_FEATURE_DIMS.get(args.clip_model)
        if not dim:
            dim = 1024 if "H-14" in args.clip_model else 768
        print(f"Vector DB dim: {dim}")
        self.db = FaissHNSWVectorDB(dim=dim, device="cpu")
        self.rtree = None
        self.floors = {}
        self.rooms = {}
        self.vertical_axis = args.vertical_axis
        self.near_wall_threshold = args.near_wall_threshold
        self.all_objects = {}

        self.features_by_object_names = {}  # {object_name: feature}
        self.new_detection_ids = None
        self.frame_embeddings = {}  # frame_id -> CPU embedding (FIFO-capped, room labeling)
        self.frame_camera_positions = {}  # frame_id -> world xyz (same FIFO cap)
        self._max_frame_embeddings = int(getattr(args, "max_frame_embeddings", 2000))
        # Room-embedding quality gates: a frame contributes to a room's embedding only if
        # its camera is inside the room and clear of walls (doorway frames see two rooms),
        # and a room needs a minimum number of contributing frames to get one at all.
        self.room_embed_wall_margin = float(getattr(args, "room_embed_wall_margin", 0.3))
        self.room_embed_min_frames = int(getattr(args, "room_embed_min_frames", 5))
        # Confidence-weighted multi-view embedding aggregation per object. When False, uses
        # last-write-wins overwrite plus a size-weighted merge mean.
        self.multiview_aggregation = bool(getattr(args, "multiview_aggregation", True))

        self.setup()

    def setup(self):
        self.rtree = index.Index(properties=index.Property(dimension=3))
        floor = SGFloor(self)
        self.floors[floor.id] = floor

    def add_object(self, object_id):
        if object_id not in self.all_objects:
            obj = SGObject(self, None, object_id)
            self.all_objects[object_id] = obj
        else:
            obj = self.all_objects[object_id]
        return obj

    def remove_object(self, object_id):
        obj = self.all_objects.get(object_id)
        if obj:
            self.rtree_delete_obj(obj)
            self.db.delete(obj.vector_id)
            del self.all_objects[object_id]

    def update_object_rooms(self, new_rooms, storeys=None):
        """Rebuild the room/floor layers from a segmentation result.

        new_rooms: {room_id: shapely geometry} or {room_id: room_segmentation.Room}
        (the dataclass carries storey/floor_height, which places each room on its
        SGFloor and gives it a vertical band). storeys: optional list of
        room_segmentation.Storey used to delimit the bands.
        """
        self.floors = {}
        self.rooms = {}

        # Storey index -> [low, high) vertical band. Bands partition the vertical axis
        # at each storey's floor height; the lowest/highest extend to +-inf.
        storey_heights = {}
        if storeys:
            for s in storeys:
                storey_heights[int(s.index)] = float(s.floor_height)
        for room in (new_rooms or {}).values():
            if hasattr(room, "storey") and room.floor_height is not None:
                storey_heights.setdefault(int(room.storey), float(room.floor_height))
        ordered = sorted(storey_heights.items(), key=lambda kv: kv[1])
        bands = {}
        for i, (idx, h) in enumerate(ordered):
            lo = float("-inf") if i == 0 else h
            hi = ordered[i + 1][1] if i + 1 < len(ordered) else float("inf")
            bands[idx] = (lo, hi)

        def _get_floor(storey_idx):
            if storey_idx not in self.floors:
                self.floors[storey_idx] = SGFloor(
                    sg=self, floor_id=storey_idx, floor_height=storey_heights.get(storey_idx)
                )
            return self.floors[storey_idx]

        if not new_rooms:
            huge_coords = [
                (-100000.0, -100000.0),
                (100000.0, -100000.0),
                (100000.0, 100000.0),
                (-100000.0, 100000.0),
            ]
            room_obj = SGRoom(self, floor=_get_floor(0), room_id=0)
            room_obj.polygon = Polygon(huge_coords)
            self.rooms[0] = room_obj
        else:
            for room_id, room in new_rooms.items():
                if hasattr(room, "polygon"):  # room_segmentation.Room dataclass
                    poly, storey_idx = room.polygon, int(room.storey)
                else:  # bare shapely geometry (legacy callers)
                    poly, storey_idx = room, 0
                room_obj = SGRoom(self, floor=_get_floor(storey_idx), room_id=room_id)
                room_obj.polygon = poly
                room_obj.height_band = bands.get(storey_idx)
                self.rooms[room_id] = room_obj

        for obj in self.all_objects.values():
            if obj.center is None:
                continue

            x, y, z = obj.center.tolist()

            room_id = self.query_position(x, y, z).get("room_id")

            should_check_poses = room_id == -1
            if not should_check_poses and room_id in self.rooms:
                if (
                    self.rooms[room_id].polygon.boundary.distance(self._floor_plane_point(x, y, z))
                    < self.near_wall_threshold
                ):
                    should_check_poses = True

            # Voting strategy for ambiguous objects.
            if should_check_poses and hasattr(obj, "camera_poses") and obj.camera_poses:
                votes = defaultdict(int)
                for pose in obj.camera_poses:
                    pose = np.asarray(pose)
                    if pose.shape[0] > 3 and pose.shape[1] > 3:
                        cx, cy, cz = pose[:3, 3]
                        if (c_id := self.query_position(cx, cy, cz).get("room_id")) != -1:
                            votes[c_id] += 1

                if votes:
                    room_id = max(votes, key=votes.get)

            if room_id in self.rooms:
                obj.room = self.rooms[room_id]

        self._update_room_embeddings()

    def _update_room_embeddings(self):
        """Room embedding = mean of frame embeddings whose camera stood inside the room,
        at least room_embed_wall_margin from any wall (doorway frames straddle two rooms
        and would contaminate both). Rooms with fewer than room_embed_min_frames
        contributing frames keep embedding=None rather than a noisy one."""
        room_embedding_map = defaultdict(list)
        for fid, emb in self.frame_embeddings.items():
            if emb is None:
                continue
            pos = self.frame_camera_positions.get(fid)
            if pos is None:
                continue
            q = self.query_position(float(pos[0]), float(pos[1]), float(pos[2]))
            if q["status"] != "in" or q["distance_to_wall"] < self.room_embed_wall_margin:
                continue
            room_id = q["room_id"]
            if room_id in self.rooms:
                self.rooms[room_id].frame_ids.add(fid)
                room_embedding_map[room_id].append(to_numpy(emb).reshape(-1))

        for room_id, embeddings_list in room_embedding_map.items():
            if len(embeddings_list) >= self.room_embed_min_frames:
                self.rooms[room_id].embedding = np.mean(embeddings_list, axis=0)

    def _floor_plane_point(self, x, y, z):
        if self.vertical_axis == 1:  # Y-up (HM3D): floor plane is X, Z
            return Point(x, z)
        if self.vertical_axis == 2:  # Z-up (Replica, ROS): floor plane is X, Y
            return Point(x, y)
        return Point(y, z)  # X-up

    def _vertical_coordinate(self, x, y, z):
        return (x, y, z)[self.vertical_axis]

    def query_position(self, x, y, z):
        p = self._floor_plane_point(x, y, z)
        v = self._vertical_coordinate(x, y, z)

        # Stacked storeys overlap in plan view, so 2D containment alone is ambiguous:
        # rooms whose storey band contains the point's height are preferred.
        contained = [
            room_id
            for room_id, room in self.rooms.items()
            if room.polygon and room.polygon.contains(p)
        ]
        in_band = [room_id for room_id in contained if self.rooms[room_id].contains_height(v)]
        candidates = in_band or contained

        if candidates:
            found_room_id = candidates[0]
            room = self.rooms[found_room_id]
            dist_to_wall = room.polygon.boundary.distance(p)  # boundary: MultiPolygon-safe
            status = "in"
        else:
            status = "out"
            pool = {
                room_id: room for room_id, room in self.rooms.items() if room.contains_height(v)
            } or self.rooms
            dists = {room_id: room.polygon.distance(p) for room_id, room in pool.items()}
            if not dists:
                dist_to_wall = 0.0
                found_room_id = -1
            else:
                nearest_room = min(dists, key=dists.get)
                dist_to_wall = dists[nearest_room]
                found_room_id = nearest_room

        return {"status": status, "room_id": found_room_id, "distance_to_wall": dist_to_wall}

    def process_semantic_results(self, frame_map, frame_id, c2w):
        semantic_results = frame_map["semantic_results"]
        frame_embedding = frame_map["frame_embedding"]
        # Keep the per-frame room-labeling embedding on the CPU (read only by
        # update_object_rooms, on CPU) to avoid an unbounded VRAM growth on long/online
        # runs. The dict is FIFO-capped so host RAM is bounded too; room labeling only
        # needs recent frames' embeddings.
        if frame_embedding is not None and hasattr(frame_embedding, "detach"):
            frame_embedding = frame_embedding.detach().cpu()
        self.frame_embeddings[frame_id] = frame_embedding
        pose = np.asarray(to_numpy(c2w))
        if pose.ndim == 2 and pose.shape[0] >= 3 and pose.shape[1] >= 4:
            self.frame_camera_positions[frame_id] = pose[:3, 3].astype(np.float64)
        if len(self.frame_embeddings) > self._max_frame_embeddings:
            oldest = next(iter(self.frame_embeddings))
            self.frame_embeddings.pop(oldest, None)
            self.frame_camera_positions.pop(oldest, None)
        self.new_detection_ids = set(semantic_results.keys())
        for semantic_id, (embeddings, confidence_score) in semantic_results.items():
            obj = self.add_object(semantic_id)
            obj.add_camera_pose(c2w, frame_id)
            if confidence_score:
                obj.confidence_score = confidence_score
            if self.multiview_aggregation:
                self.add_object_observation(semantic_id, embeddings, weight=confidence_score)
            else:
                self.set_object_features(semantic_id, embeddings, confidence_score)

    def add_object_observation(self, object_id, features, weight=None):
        """Accumulate one view's CLIP embedding into the object's confidence-weighted
        running mean, then store the normalized mean."""
        obj = self.get_object_by_id(object_id)
        if obj is None or features is None:
            return None
        feat = torch.nn.functional.normalize(features.detach().float().reshape(-1), dim=-1)
        w = float(weight) if weight and float(weight) > 0 else 1.0
        if obj.emb_sum is None:
            obj.emb_sum = w * feat
        else:
            obj.emb_sum = obj.emb_sum + w * feat.to(obj.emb_sum.device)
        obj.weight_sum += w
        obj.view_count += 1
        self.set_object_features(object_id, self._mean_embedding(obj))
        return True

    @staticmethod
    def _mean_embedding(obj):
        return torch.nn.functional.normalize(obj.emb_sum / max(obj.weight_sum, 1e-8), dim=-1)

    def clean_objects_without_points(self):
        objects_to_remove = []
        for obj_id, obj in self.all_objects.items():
            if not obj.xyz_set_once:
                objects_to_remove.append(obj_id)
            if obj.size < 20:
                objects_to_remove.append(obj_id)
        for obj_id in objects_to_remove:
            self.remove_object(obj_id)

    def merge_objects(self, source_id, target_id):
        source_obj = self.get_object_by_id(source_id)
        target_obj = self.get_object_by_id(target_id)

        if source_obj is None or target_obj is None:
            print(f"[WARN] Cannot merge objects: {source_id} or {target_id} not found")
            return False

        size_source = source_obj.size
        size_target = target_obj.size

        source_feat = source_obj.features
        target_feat = target_obj.features

        source_conf = (
            source_obj.confidence_score if source_obj.confidence_score is not None else 0.5
        )
        target_conf = (
            target_obj.confidence_score if target_obj.confidence_score is not None else 0.5
        )

        w_source = size_source * source_conf
        w_target = size_target * target_conf
        total_weight = w_source + w_target

        if self.multiview_aggregation and (
            source_obj.emb_sum is not None or target_obj.emb_sum is not None
        ):
            # Combine the confidence-weighted running sums additively so the merged object
            # is as if it had seen every view of both halves, with no view diluted by
            # point-count or merge order.
            if target_obj.emb_sum is None:
                target_obj.emb_sum = source_obj.emb_sum.clone()
                target_obj.weight_sum = source_obj.weight_sum
                target_obj.view_count = source_obj.view_count
            elif source_obj.emb_sum is not None:
                target_obj.emb_sum = target_obj.emb_sum + source_obj.emb_sum.to(
                    target_obj.emb_sum.device
                )
                target_obj.weight_sum += source_obj.weight_sum
                target_obj.view_count += source_obj.view_count
            self.set_object_features(target_id, self._mean_embedding(target_obj))
            if total_weight > 0:
                target_obj.confidence_score = (
                    w_source * source_conf + w_target * target_conf
                ) / total_weight
        elif (
            source_feat is not None and target_feat is not None and (size_source + size_target) > 0
        ):
            merged_feat = (w_source * source_feat + w_target * target_feat) / total_weight
            merged_feat = torch.nn.functional.normalize(merged_feat, dim=-1)
            self.set_object_features(target_id, merged_feat)
            target_obj.confidence_score = (
                w_source * source_conf + w_target * target_conf
            ) / total_weight
        elif source_feat is not None:
            self.set_object_features(target_id, source_feat)
        target_obj.size = size_source + size_target
        target_obj.camera_poses = np.unique(
            np.vstack([target_obj.camera_poses, source_obj.camera_poses]), axis=0
        ).tolist()
        target_obj.frame_ids |= source_obj.frame_ids
        self.remove_object(source_id)
        return True

    def assign_color_ids(self):
        for idx, obj in self.all_objects.items():
            obj.color_id = idx

        text = f"COLORS ASSIGNED. TOTAL OBJECTS {len(self.all_objects)}"
        colors = [
            "\033[91m",  # red
            "\033[93m",  # yellow
            "\033[92m",  # green
            "\033[96m",  # cyan
            "\033[94m",  # blue
            "\033[95m",  # magenta
        ]
        reset = "\033[0m"
        rainbow_text = ""
        for i, char in enumerate(text):
            rainbow_text += colors[i % len(colors)] + char
        rainbow_text += reset
        print(rainbow_text)

    def get_color_id(self, obj_id):
        obj = self.get_object_by_id(obj_id)
        if obj:
            return obj.color_id
        return 0

    def get_object_by_id(self, obj_id):
        obj = self.all_objects.get(obj_id)
        if not obj:
            # Normal: the id was merged away or cleaned (objects churn every frame) and
            # callers handle None.
            logging.debug("[SGRAPH] object #%s not found (merged/cleaned)", obj_id)
        return obj

    def rtree_rebuild(self):
        """Rebuild a fresh R-tree from all live objects (insert-only).

        We deliberately NEVER call libspatialindex's delete: deleting with
        bounds that don't exactly match the inserted ones corrupts its internal
        pages (InvalidPageException: Unknown page id), after which the next call
        ABORTS the whole process ("double free or corruption") -- an uncatchable
        glibc abort. Instead every mutation just flags the index stale and we
        rebuild it from the live objects before the next spatial query. Rebuild
        is O(N) and runs at most once per fusion pass, so it is cheap.
        """
        try:
            new_idx = index.Index(properties=index.Property(dimension=3))
        except Exception as e:
            logging.error(f"[RTREE] could not create index: {e}")
            self._rtree_dirty = True
            return
        objs = (
            self.all_objects.values()
            if isinstance(self.all_objects, dict)
            else self.all_objects
        )
        for o in list(objs):
            if not isinstance(o, SGObject):
                continue
            aabb = getattr(o, "aabb", None)
            if aabb is None:
                o._rtree_aabb = None
                continue
            try:
                new_idx.insert(o.id, aabb)
                o._rtree_aabb = list(aabb)
            except Exception:
                o._rtree_aabb = None
        self.rtree = new_idx
        self._rtree_dirty = False

    def rtree_insert_obj(self, obj):
        # Lazy: just flag the index stale; it is rebuilt before the next query.
        self._rtree_dirty = True

    def rtree_delete_obj(self, obj):
        # Lazy: never call the fragile rtree.delete; flag stale and rebuild later.
        self._rtree_dirty = True

    def rtree_find_nearby_objects(self, source_obj_id, threshold=1.0):
        source_obj = self.get_object_by_id(source_obj_id)
        if not source_obj or source_obj.aabb is None:
            return []
        if getattr(self, "_rtree_dirty", True):
            self.rtree_rebuild()
        min_x, min_y, min_z, max_x, max_y, max_z = source_obj.aabb
        query_box = (
            min_x - threshold,
            min_y - threshold,
            min_z - threshold,
            max_x + threshold,
            max_y + threshold,
            max_z + threshold,
        )
        try:
            results = self.rtree.intersection(query_box, objects=True)
            return [
                it for it in results
                if (it.id if hasattr(it, "id") else it) != source_obj.id
            ]
        except Exception as e:
            logging.error(f"[RTREE] intersection failed, using linear scan: {e}")
            return self._linear_find_nearby(query_box, source_obj.id)

    def _linear_find_nearby(self, query_box, exclude_id):
        """Reliable O(N) AABB-overlap fallback when the R-tree is unusable."""
        qx0, qy0, qz0, qx1, qy1, qz1 = query_box
        objs = (
            self.all_objects.values()
            if isinstance(self.all_objects, dict)
            else self.all_objects
        )
        out = []
        for o in objs:
            if not isinstance(o, SGObject) or o.id == exclude_id:
                continue
            a = getattr(o, "aabb", None)
            if a is None:
                continue
            if (
                a[0] <= qx1 and a[3] >= qx0
                and a[1] <= qy1 and a[4] >= qy0
                and a[2] <= qz1 and a[5] >= qz0
            ):
                out.append(o.id)
        return out

    def _update_geometry_via_index(self, active_pcd, object_ids, sem_index):
        """Set each object's geometry from its whole-map stable reduction (the index, which
        composes resident + evicted cells) folded with its active rows (always resident).
        Equals a gather over the full active+stable cloud."""
        import numpy as np

        a_sem = active_pcd._semantic.view(-1) if active_pcd.get_points_num > 0 else None
        for obj_id in object_ids:
            oid = int(obj_id)
            obj = self.get_object_by_id(oid)
            if obj is None:
                continue
            red = sem_index.reduction(oid)  # stable, whole-map: (count, sum, min, max) or None
            if a_sem is not None:
                am = a_sem == oid
                if am.any():
                    ap = active_pcd._xyz[am].detach().cpu().numpy().astype(np.float64)
                    a = (ap.shape[0], ap.sum(0), ap.min(0), ap.max(0))
                    if red is None:
                        red = a
                    else:
                        c, s, mn, mx = red
                        red = (c + a[0], s + a[1], np.minimum(mn, a[2]), np.maximum(mx, a[3]))
            if red is None or red[0] == 0:
                continue
            c, s, mn, mx = red
            obj.set_geometry(torch.tensor(s / c, dtype=torch.float32), list(mn) + list(mx), c)

    def update_object_geometry(self, active_pcd, stable_pcd, object_ids=None, sem_index=None):
        if sem_index is not None and object_ids is not None:
            self._update_geometry_via_index(active_pcd, object_ids, sem_index)
            return
        pcds = [p for p in [active_pcd, stable_pcd] if p.get_points_num > 0]
        if not pcds:
            return

        if object_ids is not None:
            # Gather only the changed objects' points from each cloud before concatenating,
            # so the whole map's xyz is never materialized.
            target_ids = torch.as_tensor(list(object_ids), device=pcds[0]._semantic.device)
            xyzs, sems = [], []
            for p in pcds:
                sem = p._semantic.squeeze(-1)
                m = torch.isin(sem, target_ids)
                if m.any():
                    xyzs.append(p._xyz[m])
                    sems.append(sem[m])
            if not xyzs:
                return
            cat_xyz, cat_sem = torch.cat(xyzs), torch.cat(sems)
        else:
            cat_xyz = torch.cat([p._xyz for p in pcds])
            cat_sem = torch.cat([p._semantic for p in pcds]).squeeze()

        sort_idx = torch.argsort(cat_sem)
        sorted_xyz, sorted_sem = cat_xyz[sort_idx], cat_sem[sort_idx]

        unique_ids, counts = torch.unique_consecutive(sorted_sem, return_counts=True)
        point_groups = torch.split(sorted_xyz, counts.cpu().tolist())
        for obj_id, points in zip(unique_ids.tolist(), point_groups, strict=False):
            obj = self.get_object_by_id(int(obj_id))
            if obj:
                obj.update_geometry(points)

    def get_object_embedding_and_confidence(self, object_id, device=None):
        obj = self.get_object_by_id(object_id)
        if obj is None:
            return None, None
        if device:
            if obj.features is None:
                logging.debug("[SGRAPH] Object has no features yet")
                return None, None
            return obj.features.to(device), obj.confidence_score
        return obj.features, obj.confidence_score

    def set_object_features(self, object_id, features, confidence_score=None):
        obj = self.get_object_by_id(object_id)
        if obj is None:
            return None
        obj.vector_id = self.db.insert(
            features,
            vec_id=obj.vector_id if obj.vector_id else None,
            metadata={"object_id": object_id},
        )
        if confidence_score:
            obj.confidence_score = confidence_score
        return True

    def export_objects(self, export_path="data.json"):
        output_dir = os.path.dirname(export_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        serialized_rooms = []
        for room in self.rooms.values():
            serialized_rooms.append(room.serialize())

        objects = self.all_objects.values()
        serialized_objects = []
        for obj in objects:
            serialized_objects.append(obj.serialize())

        export_result = {
            "objects": serialized_objects,
            "rooms": serialized_rooms,
            "floors": [floor.serialize() for floor in self.floors.values()],
        }
        with open(export_path, "w") as f:
            json.dump(export_result, f, indent=4)

    def __repr__(self):
        return f"<SceneGraph scene={self.scene} floors={len(self.floors)}>"
