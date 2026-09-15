"""Wall-first, self-calibrating room segmentation for the Gaussian map.

Pipeline:
  1. classify every splat floor/ceiling/wall/clutter from its normal   (classify.py)
  2. detect floor heights automatically, one per storey                (floors.py)
  3. rasterize walls only, in a floor-relative band                    (grid.py)
  4. pick the room count via a wall-inflation sweep, then flood         (partition.py)
  5. merge undersized regions into their neighbors                     (partition.py)
  6. trace room polygons                                               (vectorize.py)

The only external input is `vertical_axis`. Per-splat normals are recomputed from
scales/rotations (gaussian_pointcloud.get_normal); the PLY stores nx/ny/nz as zeros but
keeps scale_*/rot_*, so the same normals regenerate offline from a saved *_stable.ply.
"""

import os

import numpy as np

from .classify import HORIZONTAL, WALL, classify_splats
from .constants import RESOLUTION_M, STOREY_BELOW_FLOOR_M
from .debug_viz import save_room_vectors, save_storey_masks
from .floors import detect_floor_heights
from .grid import BEVGrid, free_space_mask, wall_mask
from .partition import merge_small_regions, partition_rooms
from .types import Room, RoomSegmentationResult, Storey
from .vectorize import vectorize_rooms


class StructuralRoomSegmenter:
    def __init__(self, vertical_axis, output_dir=None):
        self.vertical_axis = int(vertical_axis)
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def segment(self, points, normals, opacities=None):
        """points/normals: (N,3) torch tensors or numpy arrays; opacities: (N,) or (N,1)."""
        result = RoomSegmentationResult()
        splats = classify_splats(points, normals, opacities, self.vertical_axis)
        if len(splats) == 0:
            return result

        floors = detect_floor_heights(splats.h[splats.label == HORIZONTAL], splats.h)
        next_room_id = 1
        for index, floor_h in enumerate(floors):
            ceiling = floors[index + 1] if index + 1 < len(floors) else np.inf
            storey_splats = splats.subset(
                (splats.h >= floor_h - STOREY_BELOW_FLOOR_M)
                & (splats.h < ceiling - STOREY_BELOW_FLOOR_M)
            )
            if len(storey_splats) == 0:
                continue

            grid = BEVGrid.from_points(storey_splats.u, storey_splats.v, RESOLUTION_M)
            is_wall = storey_splats.label == WALL
            wall = wall_mask(
                grid,
                storey_splats.u[is_wall],
                storey_splats.v[is_wall],
                storey_splats.h[is_wall],
                floor_h,
            )
            free = free_space_mask(grid, storey_splats.u, storey_splats.v, wall)

            labels = partition_rooms(free, wall, grid.resolution)
            labels = merge_small_regions(labels, grid.resolution)
            polygons = vectorize_rooms(labels, grid)

            storey = Storey(index=index, floor_height=float(floor_h))
            for local_id in sorted(polygons):
                room = Room(
                    id=next_room_id,
                    polygon=polygons[local_id],
                    storey=index,
                    floor_height=float(floor_h),
                    area_m2=float(polygons[local_id].area),
                )
                result.rooms[room.id] = room
                storey.room_ids.append(room.id)
                next_room_id += 1
            result.storeys.append(storey)

            if self.output_dir:
                save_storey_masks(self.output_dir, index, wall, free, labels)

        if self.output_dir and not result.is_empty:
            save_room_vectors(
                os.path.join(self.output_dir, "rooms_vector.png"), result.room_polygons()
            )
        return result
