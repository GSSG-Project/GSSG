import logging

import torch
from shapely.geometry import Polygon, mapping

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class SGFloor:
    def __init__(self, sg, floor_id: int = 0, floor_height: float | None = None):
        self.sg = sg
        self.id = floor_id
        self.floor_height = floor_height
        self.rooms = set()

    def add_room(self, room):
        self.rooms.add(room)

    def __repr__(self):
        return f"<SGFloor id={self.id} rooms={len(self.rooms)}>"

    def serialize(self):
        return {
            "id": self.id,
            "floor_height": self.floor_height,
            "room_ids": sorted(r.id for r in self.rooms),
        }


class SGRoom:
    def __init__(self, sg, floor: SGFloor, room_id: int = 0):
        self.sg = sg
        self.floor = floor
        self.id = room_id
        self.polygon: Polygon | None = None
        self.frame_ids = set()
        self.embedding = None
        # Vertical extent [low, high) of the room's storey in world coordinates along
        # sg.vertical_axis; None when the segmentation carried no storey information.
        self.height_band = None
        floor.add_room(self)

    def __repr__(self):
        return f"<SGRoom id={self.id} floor={self.floor.id}>"

    def contains_height(self, v: float) -> bool:
        if self.height_band is None:
            return True
        lo, hi = self.height_band
        return lo <= v < hi

    @property
    def aabb(self):
        """World-frame (minx,miny,minz,maxx,maxy,maxz) from the plan polygon and the
        storey height band (a 3 m default ceiling when the band is open-ended)."""
        if self.polygon is None:
            return None
        px0, py0, px1, py1 = self.polygon.bounds
        if self.height_band is not None:
            lo, hi = self.height_band
            if lo == float("-inf"):
                lo = self.floor.floor_height if self.floor.floor_height is not None else 0.0
            if hi == float("inf"):
                hi = lo + 3.0
        else:
            lo, hi = 0.0, 0.0
        axis = getattr(self.sg, "vertical_axis", 2)
        if axis == 1:  # Y-up: plan coords are (x, z)
            return [px0, lo, py0, px1, hi, py1]
        if axis == 2:  # Z-up: plan coords are (x, y)
            return [px0, py0, lo, px1, py1, hi]
        return [lo, px0, py0, hi, px1, py1]  # X-up: plan coords are (y, z)

    def serialize(self):
        return {
            "id": self.id,
            "floor_id": self.floor.id if self.floor is not None else None,
            "polygon": mapping(self.polygon) if self.polygon is not None else None,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "aabb": self.aabb,
            "frame_count": len(self.frame_ids),
        }


class SGObject:
    def __init__(self, sg, room: SGRoom, object_id: int):
        self.sg = sg
        self.room = room
        self.id = object_id  # segmentation id
        self.center = None
        self.aabb = None
        self.vector_id = None
        self.object_name = None
        self.color_id = None
        self.size = 0
        self.camera_poses = []
        self.confidence_score = None
        # Confidence-weighted multi-view embedding accumulator:
        # emb_sum = sum_i w_i * normalize(view_i); the stored db vector is
        # normalize(emb_sum / weight_sum). Additive, so it is order-invariant and
        # composable across fusion merges.
        self.emb_sum = None
        self.weight_sum = 0.0
        self.view_count = 0
        self._rtree_aabb = None  # bounds currently in the R-tree (delete-before-insert)
        self.xyz_set_once = False
        self.xyz_rem_called = False
        self.frame_id = None
        self.frame_ids = set()

    @property
    def features(self):
        if self.vector_id is None:
            return None
        return self.sg.db.get_vector(self.vector_id)

    def update_object_name(self):
        self.object_name = self.sg.get_closest_object_name(self.features)

    def update_geometry(self, pcd):
        self.update_center(pcd)
        self.update_aabb(pcd)
        self.size = len(pcd)
        self.sg.rtree_insert_obj(self)
        self.xyz_set_once = True

    def set_geometry(self, center, aabb, size):
        """Set geometry from precomputed components instead of a gathered point cloud."""
        self.center = center
        self.aabb = aabb
        self.size = int(size)
        self.sg.rtree_insert_obj(self)
        self.xyz_set_once = True

    def add_camera_pose(self, camera_pose, frame_id):
        if camera_pose not in self.camera_poses:
            self.camera_poses.append(camera_pose)
            self.frame_ids.add(frame_id)
        self.frame_id = frame_id

    def update_center(self, pcd):
        if pcd is None or len(pcd) == 0:
            self.center = None
            return
        self.center = torch.mean(pcd, dim=0)

    def update_aabb(self, pcd):
        if pcd.shape[0] == 0:
            return None
        c_min = pcd.min(axis=0).values.tolist()
        c_max = pcd.max(axis=0).values.tolist()
        # AABB is (min_x, min_y, min_z, max_x, max_y, max_z)
        self.aabb = c_min + c_max

    def serialize(self):
        return {
            "id": self.id,
            "center": self.center.tolist() if self.center is not None else None,
            "room_id": self.room.id if self.room is not None else None,
            "aabb": self.aabb,
            "vector_id": self.vector_id,
        }

    def __repr__(self):
        return f"<SGObject id={self.id} room={self.room.id}>"
