"""Room segmentation: four swappable methods behind one interface.

`methods.segment` (or the `segment_rooms_from_*` helpers) maps a splat cloud to a
`RoomSegmentationResult`; the method switch is `room_seg_method` in configs and
`--method` on the CLI. `ours` is the wall-first, self-calibrating structural segmenter;
`ours_v2` the splat-native transmittance floorplan; `hydra` and `hovsg` are reimplemented
baselines. Saved `*_stable.ply` maps re-segment offline via
`python -m gssg.scene_graph.room_segmentation --input <ply> --method <name>`.
"""

from .api import segment_rooms_from_arrays, segment_rooms_from_ply
from .cloud import SplatCloud, load_splat_cloud
from .floorplan import write_floorplan
from .methods import METHODS, segment
from .segmenter import StructuralRoomSegmenter
from .types import Room, RoomSegmentationResult, Storey

__all__ = [
    "METHODS",
    "Room",
    "RoomSegmentationResult",
    "SplatCloud",
    "Storey",
    "StructuralRoomSegmenter",
    "load_splat_cloud",
    "segment",
    "segment_rooms_from_arrays",
    "segment_rooms_from_ply",
    "write_floorplan",
]
