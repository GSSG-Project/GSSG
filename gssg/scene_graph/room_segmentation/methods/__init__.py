"""Room-segmentation method registry.

Every method maps a SplatCloud to a RoomSegmentationResult:

- ``ours``    — wall-first structural segmenter (normals + inflation sweep)
- ``ours_v2`` — splat-native transmittance floorplan (covariance + opacity)
- ``hydra``   — free-space dilation sweep, after Hughes et al. (RSS 2022)
- ``hovsg``   — BEV histogram + watershed, after Werby et al. (RSS 2024)

Method modules import lazily so the live pipeline only pays for what it uses.
"""

from ..types import RoomSegmentationResult


def _ours(cloud, vertical_axis, output_dir):
    from ..segmenter import StructuralRoomSegmenter

    if cloud.normals is None:
        raise ValueError("method 'ours' needs normals (PLY missing scale_*/rot_* fields)")
    return StructuralRoomSegmenter(vertical_axis, output_dir).segment(
        cloud.xyz, cloud.normals, cloud.opacities
    )


def _ours_v2(cloud, vertical_axis, output_dir):
    from .transmittance import segment

    return segment(cloud, vertical_axis, output_dir)


def _hydra(cloud, vertical_axis, output_dir):
    from .hydra import segment

    return segment(cloud, vertical_axis, output_dir)


def _hovsg(cloud, vertical_axis, output_dir):
    from .hovsg import segment

    return segment(cloud, vertical_axis, output_dir)


METHODS = {"ours": _ours, "ours_v2": _ours_v2, "hydra": _hydra, "hovsg": _hovsg}


def segment(cloud, *, method="ours", vertical_axis=2, output_dir=None) -> RoomSegmentationResult:
    if method not in METHODS:
        raise ValueError(f"unknown room_seg_method {method!r}; options: {sorted(METHODS)}")
    return METHODS[method](cloud, int(vertical_axis), output_dir)
