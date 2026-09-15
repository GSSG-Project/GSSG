"""Self-tuning room partition.

To choose the room count, virtually inflate the walls over a sweep of radii. At small
radii the free space stays one blob, at large radii it shatters; the component count that
stays stable across the sweep (the median) is the room count. Seeds then flood the free
space, and undersized regions are merged into their largest-border neighbor.
"""

import cv2
import numpy as np

from .constants import (
    FALLBACK_MIN_CORE_M,
    MIN_FREE_AREA_M2,
    MIN_ROOM_AREA_M2,
    MIN_SEED_AREA_M2,
    SWEEP_RADII_M,
    SWEEP_STEPS,
)


def _strip_filaments(labels):
    """Drop 1-px leak trails the flood leaves along wall rims (they make a room's outer
    contour wrap around its neighbor) and keep each room's largest connected piece."""
    out = np.zeros_like(labels)
    kernel = np.ones((3, 3), np.uint8)
    for uid in np.unique(labels[labels > 0]):
        mask = cv2.morphologyEx((labels == uid).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, components = cv2.connectedComponents(mask)
        if n <= 1:
            continue
        areas = np.bincount(components.ravel())
        areas[0] = 0
        out[components == areas.argmax()] = uid
    return out


def _relabel_by_area(labels):
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    out = np.zeros_like(labels)
    for new_id, old_id in enumerate(ids[np.argsort(-counts)], start=1):
        out[labels == old_id] = new_id
    return out


def _seed_components(dist, radius_px, min_seed_px):
    seeds = (dist > radius_px).astype(np.uint8)
    _, components = cv2.connectedComponents(seeds)
    areas = np.bincount(components.ravel())
    valid = [i for i in range(1, len(areas)) if areas[i] >= min_seed_px]
    return components, valid


def partition_rooms(free, wall, resolution):
    """free/wall: bool grids. Returns an int32 label grid (0 = not a room, 1..N = rooms)."""
    labels = np.zeros(free.shape, dtype=np.int32)
    if free.sum() * resolution**2 < MIN_FREE_AREA_M2:
        return labels

    dist = cv2.distanceTransform((free * 255).astype(np.uint8), cv2.DIST_L2, 5)
    min_seed_px = MIN_SEED_AREA_M2 / resolution**2

    radii = np.linspace(SWEEP_RADII_M[0], SWEEP_RADII_M[1], SWEEP_STEPS)
    sweep = [_seed_components(dist, r / resolution, min_seed_px) for r in radii]
    counts = [len(valid) for _, valid in sweep]

    # vote over radii that still see the scene; zero-seed radii are beyond its scale
    voting = [c for c in counts if c > 0]
    room_count = int(round(float(np.median(voting)))) if voting else 0
    if room_count < 1:
        # scan too small for the sweep: one room if there is any real core at all
        if dist.max() * resolution < FALLBACK_MIN_CORE_M:
            return labels
        components = np.zeros(free.shape, dtype=np.int32)
        components[np.unravel_index(int(dist.argmax()), dist.shape)] = 1
        valid = [1]
    else:
        matching = [i for i, c in enumerate(counts) if c == room_count]
        idx = matching[0] if matching else int(np.argmin(np.abs(np.array(counts) - room_count)))
        components, valid = sweep[idx]

    remap = np.zeros(int(components.max()) + 1, dtype=np.int32)
    for k, comp_id in enumerate(valid):
        remap[comp_id] = k + 1
    seeds = np.where(free, remap[components], 0)

    labels = _geodesic_flood(seeds, free)
    return _relabel_by_area(_strip_filaments(labels))


def _geodesic_flood(labels, free):
    """Equal-speed multi-source flood: every free cell joins the geodesically nearest seed,
    so room fronts meet at the doorway midline and never wrap around a neighbor's rim
    (cv2.watershed on a near-flat priority image does exactly that)."""
    labels = labels.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max(labels.shape)):
        grown = cv2.dilate(labels, kernel)
        new = free & (labels == 0) & (grown > 0)
        if not new.any():
            break
        labels[new] = grown[new]
    return labels.astype(np.int32)


def merge_small_regions(labels, resolution, min_area_m2=MIN_ROOM_AREA_M2):
    """Merge regions under min_area_m2 into the neighbor sharing the longest border;
    isolated small islands are dropped."""
    min_px = min_area_m2 / resolution**2
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(int(labels.max()) + 2):
        areas = np.bincount(labels[labels > 0].ravel(), minlength=int(labels.max()) + 1)
        small = [i for i in range(1, len(areas)) if 0 < areas[i] < min_px]
        if not small:
            break
        sid = min(small, key=lambda i: areas[i])
        mask = (labels == sid).astype(np.uint8)
        ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
        neighbors = labels[ring]
        neighbors = neighbors[neighbors > 0]
        labels[mask > 0] = np.bincount(neighbors).argmax() if neighbors.size else 0
    return _relabel_by_area(labels)
