"""HOV-SG floor + room segmentation (Werby et al., RSS 2024).

Reimplemented from the released code (hovsg/graph/graph.py: segment_floors L262-376,
segment_rooms L378-577; hovsg/utils/graph_utils.py: distance_transform), generalized from
their fixed Y-up frame to `vertical_axis`. Point-based: uses splat centers only, with
low-opacity splats dropped (their input is a fused RGB-D cloud; a raw GS map is not).

Faithful choices kept from the original, quirks included: 90th-percentile peak gate,
eps=1 m peak chaining with top-1/top-2 selection per cluster, consecutive peak pairing,
0.25*max wall threshold, 10 px border padding, Otsu seeds on the normalized distance
transform with the (11,1) sigma=10 blur, watershed over the binary obstacle map, no
post-watershed region merging. Deviations: exact bin edges anchored at the point extents
(the original int-truncates the grid size, drifting cells on large floors), and a
single-storey fallback when fewer than two histogram peaks survive (the original finds
no rooms at all in that case).
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..classify import _PLANE_AXES, as_numpy
from ..constants import MIN_OPACITY
from ..debug_viz import save_storey_masks
from ..grid import BEVGrid
from ..types import Room, RoomSegmentationResult, Storey
from ..vectorize import vectorize_rooms

DOWNSAMPLE_VOXEL_M = 0.05
HEIGHT_BIN_M = 0.01
PEAK_SMOOTH_SIGMA_BINS = 2
PEAK_MIN_SEPARATION_M = 0.2
PEAK_HEIGHT_PERCENTILE = 90
PEAK_CLUSTER_EPS_M = 1.0

GRID_RES_M = 0.05
WALL_BAND_ABOVE_FLOOR_M = 1.5
WALL_BAND_BELOW_CEILING_M = 0.3
FOOTPRINT_BELOW_CEILING_M = 0.2
PAD_PX = 10
WALL_THRESHOLD_FRAC = 0.25
MIN_SEED_SIDE_M = 0.5


def _downsample(xyz, voxel):
    _, idx = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_index=True)
    return xyz[idx]


def _floor_slabs(heights_ds, h_min, h_max):
    """Their segment_floors: histogram peaks -> 1 m chains -> top-1/top-2 -> pairs."""
    n_bins = max(int((h_max - h_min) / HEIGHT_BIN_M), 1)
    counts, edges = np.histogram(heights_ds, bins=n_bins)
    smooth = gaussian_filter1d(counts.astype(np.float64), PEAK_SMOOTH_SIGMA_BINS)
    distance = max(int(PEAK_MIN_SEPARATION_M / HEIGHT_BIN_M), 1)
    peaks, _ = find_peaks(
        smooth, distance=distance, height=np.percentile(smooth, PEAK_HEIGHT_PERCENTILE)
    )
    positions = edges[peaks]
    strengths = smooth[peaks]

    clusters = []
    for i in np.argsort(positions):
        if clusters and positions[i] - clusters[-1][-1][0] <= PEAK_CLUSTER_EPS_M:
            clusters[-1].append((positions[i], strengths[i]))
        else:
            clusters.append([(positions[i], strengths[i])])

    kept = []
    for c, members in enumerate(clusters):
        top = 1 if c in (0, len(clusters) - 1) else 2
        kept += [p for p, _ in sorted(members, key=lambda m: -m[1])[:top]]
    kept = sorted(kept)

    if len(kept) < 2:
        return [[float(h_min), float(h_max)]]
    slabs = [[float(kept[i]), float(kept[i + 1])] for i in range(0, len(kept) - 1, 2)]
    slabs[0][0] = (slabs[0][0] + float(h_min)) / 2
    slabs[-1][1] = (slabs[-1][1] + float(h_max)) / 2
    return slabs


def _normalized_u8(hist):
    return cv2.normalize(hist.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _pad(img):
    return cv2.copyMakeBorder(img, PAD_PX, PAD_PX, PAD_PX, PAD_PX, cv2.BORDER_CONSTANT, value=0)


def _segment_storey(u, v, h, z0, storey_height, grid, output_dir, storey_index):
    u_edges = grid.u_min + np.arange(grid.width + 1) * grid.resolution
    v_edges = grid.v_min + np.arange(grid.height + 1) * grid.resolution

    in_wall_band = (h >= z0 + WALL_BAND_ABOVE_FLOOR_M) & (
        h < z0 + storey_height - WALL_BAND_BELOW_CEILING_M
    )
    wall_hist, _, _ = np.histogram2d(v[in_wall_band], u[in_wall_band], bins=[v_edges, u_edges])
    walls = _normalized_u8(wall_hist)
    walls = cv2.GaussianBlur(walls, (5, 5), 1)
    walls = ((walls > WALL_THRESHOLD_FRAC * walls.max()) * 255).astype(np.uint8)
    walls = _pad(walls)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, cross)

    below_ceiling = h < z0 + storey_height - FOOTPRINT_BELOW_CEILING_M
    full_hist, _, _ = np.histogram2d(v[below_ceiling], u[below_ceiling], bins=[v_edges, u_edges])
    footprint = _normalized_u8(full_hist)
    footprint = cv2.GaussianBlur(footprint, (21, 21), 2)
    footprint = ((footprint > 0) * 255).astype(np.uint8)
    footprint = _pad(footprint)
    rect5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    footprint = cv2.morphologyEx(footprint, cv2.MORPH_CLOSE, rect5, iterations=3)
    contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    footprint = np.zeros_like(footprint)
    cv2.drawContours(footprint, contours, -1, 255, thickness=-1)

    full_map = cv2.bitwise_or(walls, cv2.bitwise_not(footprint))
    rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, rect3, iterations=2)

    dist = cv2.distanceTransform(cv2.bitwise_not(full_map), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist = cv2.GaussianBlur(dist, (11, 1), 10)
    _, seeds = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_seed_px = (MIN_SEED_SIDE_M / grid.resolution) ** 2
    contours = [c for c in contours if cv2.contourArea(c) > min_seed_px]

    markers = np.zeros(full_map.shape, dtype=np.int32)
    for i, contour in enumerate(contours):
        cv2.drawContours(markers, [contour], -1, i + 1, thickness=-1)
    cv2.circle(markers, (3, 3), 1, len(contours) + 1, -1)
    cv2.watershed(cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR), markers)

    labels = np.zeros(full_map.shape, dtype=np.int32)
    for i in range(len(contours)):
        labels[markers == i + 1] = i + 1
    labels = labels[PAD_PX:-PAD_PX, PAD_PX:-PAD_PX]

    if output_dir:
        save_storey_masks(
            output_dir,
            storey_index,
            walls[PAD_PX:-PAD_PX, PAD_PX:-PAD_PX] > 0,
            full_map[PAD_PX:-PAD_PX, PAD_PX:-PAD_PX] == 0,
            labels,
        )
    return labels


def segment(cloud, vertical_axis=2, output_dir=None) -> RoomSegmentationResult:
    result = RoomSegmentationResult()
    xyz = as_numpy(cloud.xyz).reshape(-1, 3)
    if cloud.opacities is not None:
        xyz = xyz[as_numpy(cloud.opacities).reshape(-1) > MIN_OPACITY]
    if len(xyz) == 0:
        return result

    u_axis, v_axis = _PLANE_AXES[int(vertical_axis)]
    u, v, h = xyz[:, u_axis], xyz[:, v_axis], xyz[:, vertical_axis]

    heights_ds = _downsample(xyz, DOWNSAMPLE_VOXEL_M)[:, vertical_axis]
    slabs = _floor_slabs(heights_ds, h.min(), h.max())

    next_room_id = 1
    for index, (lo, hi) in enumerate(slabs):
        in_storey = (h >= lo) & (h <= hi)
        if not in_storey.any():
            continue
        su, sv, sh = u[in_storey], v[in_storey], h[in_storey]
        z0 = float(sh.min())
        storey_height = hi - z0
        grid = BEVGrid.from_points(su, sv, GRID_RES_M, padding=0.0)

        labels = _segment_storey(su, sv, sh, z0, storey_height, grid, output_dir, index)
        polygons = vectorize_rooms(labels, grid)

        storey = Storey(index=index, floor_height=z0)
        for local_id in sorted(polygons):
            room = Room(
                id=next_room_id,
                polygon=polygons[local_id],
                storey=index,
                floor_height=z0,
                area_m2=float(polygons[local_id].area),
            )
            result.rooms[room.id] = room
            storey.room_ids.append(room.id)
            next_room_id += 1
        result.storeys.append(storey)
    return result
