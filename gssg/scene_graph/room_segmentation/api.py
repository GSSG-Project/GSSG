"""Reusable, Mapping-free entrypoints for room segmentation.

All methods take plain arrays (numpy or torch), so the live mapper and an offline
saved-map run share one code path. The PLY stores nx/ny/nz as zeros, but per-splat
normals are a deterministic function of the serialized `scale_*`/`rot_*` fields, so a
saved `*_stable.ply` regenerates the exact normals the live map uses. Use the run's
final/global stable PLY for the cleanest result.

CLI: ``python -m gssg.scene_graph.room_segmentation --input <stable.ply>
[--method ours|ours_v2|hydra|hovsg] [--vertical-axis N] [--out DIR]``.
"""

from __future__ import annotations

import os

from .classify import as_numpy
from .cloud import SplatCloud, load_splat_cloud
from .methods import segment
from .types import RoomSegmentationResult


def segment_rooms_from_arrays(
    points,
    normals=None,
    opacities=None,
    *,
    scales=None,
    rotations=None,
    method="ours",
    vertical_axis=2,
    output_dir=None,
) -> RoomSegmentationResult:
    """Segment rooms from raw per-splat arrays — numpy or torch, points (N,3), normals
    (N,3), opacities (N,), scales (N,3), rotations (N,4) wxyz. Shared by the live mapper
    and the offline CLI. Point-based methods need only `points`; `ours` needs normals;
    `ours_v2` needs scales/rotations/opacities."""
    cloud = SplatCloud(
        xyz=as_numpy(points),
        normals=as_numpy(normals),
        opacities=as_numpy(opacities),
        scales=as_numpy(scales),
        rotations=as_numpy(rotations),
    )
    return segment(cloud, method=method, vertical_axis=vertical_axis, output_dir=output_dir)


def segment_rooms_from_ply(
    ply_path, *, method="ours", vertical_axis=2, output_dir=None
) -> RoomSegmentationResult:
    """Load a saved Gaussian PLY (CUDA-free) and segment its rooms in one call."""
    cloud = load_splat_cloud(ply_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(ply_path)), "rooms")
    return segment(cloud, method=method, vertical_axis=vertical_axis, output_dir=output_dir)
