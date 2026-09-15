"""Bird's-eye-view grid: wall raster and observed free space.

Walls are rasterized from wall-labeled splats only. A cell counts as wall when wall splats
occupy at least 2 of 3 height sub-bands of the wall band — a real wall spans the band,
a sofa back or cabinet side occupies only the lowest sub-band.

Free space is "observed and not wall": the observed mask (any splat projects there)
confines rooms to the scanned area, so the building exterior never becomes a room.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import (
    MIN_WALL_CELLS,
    OBSERVED_CLOSE_M,
    PADDING_M,
    RESOLUTION_M,
    ROBUST_BOUNDS_PERCENTILE,
    WALL_BAND_ABOVE_FLOOR_M,
    WALL_CLOSE_M,
    WALL_SUBBAND_MIN_VOTES,
    WALL_SUBBANDS,
    WALL_THICKEN_M,
)


def _odd_kernel_px(size_m, resolution):
    px = max(int(round(size_m / resolution)), 1)
    return px if px % 2 == 1 else px + 1


@dataclass
class BEVGrid:
    u_min: float
    v_min: float
    width: int
    height: int
    resolution: float = RESOLUTION_M

    @classmethod
    def from_points(cls, u, v, resolution=RESOLUTION_M, padding=PADDING_M):
        p = ROBUST_BOUNDS_PERCENTILE
        u_min = float(np.percentile(u, p)) - padding
        u_max = float(np.percentile(u, 100 - p)) + padding
        v_min = float(np.percentile(v, p)) - padding
        v_max = float(np.percentile(v, 100 - p)) + padding
        width = max(int(np.ceil((u_max - u_min) / resolution)), 1)
        height = max(int(np.ceil((v_max - v_min) / resolution)), 1)
        return cls(u_min=u_min, v_min=v_min, width=width, height=height, resolution=resolution)

    def bin(self, u, v):
        """Returns (ix, iy, valid). Outlier points beyond the robust bounds are dropped."""
        ix = np.floor((u - self.u_min) / self.resolution).astype(np.int64)
        iy = np.floor((v - self.v_min) / self.resolution).astype(np.int64)
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        return ix, iy, valid

    def occupancy(self, u, v, min_count=1):
        ix, iy, valid = self.bin(u, v)
        flat = np.bincount(iy[valid] * self.width + ix[valid], minlength=self.height * self.width)
        return flat.reshape(self.height, self.width) >= min_count

    def pixel_to_world(self, ix, iy):
        return self.u_min + ix * self.resolution, self.v_min + iy * self.resolution

    def area_m2(self, n_pixels):
        return n_pixels * self.resolution**2


def _drop_small_islands(mask_u8, min_cells):
    n, components = cv2.connectedComponents((mask_u8 > 0).astype(np.uint8))
    if n <= 1:
        return mask_u8
    areas = np.bincount(components.ravel())
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas[1:] >= min_cells
    return (keep[components] * 255).astype(np.uint8)


def wall_mask(grid, wall_u, wall_v, wall_h, floor_height):
    band_lo = floor_height + WALL_BAND_ABOVE_FLOOR_M[0]
    band_hi = floor_height + WALL_BAND_ABOVE_FLOOR_M[1]
    in_band = (wall_h >= band_lo) & (wall_h <= band_hi)
    u, v, h = wall_u[in_band], wall_v[in_band], wall_h[in_band]

    votes = np.zeros((grid.height, grid.width), dtype=np.uint8)
    sub = np.clip(
        ((h - band_lo) / max(band_hi - band_lo, 1e-6) * WALL_SUBBANDS).astype(np.int64),
        0,
        WALL_SUBBANDS - 1,
    )
    for s in range(WALL_SUBBANDS):
        votes += grid.occupancy(u[sub == s], v[sub == s]).astype(np.uint8)
    wall = ((votes >= WALL_SUBBAND_MIN_VOTES) * 255).astype(np.uint8)
    wall = _drop_small_islands(wall, MIN_WALL_CELLS)

    k_close = _odd_kernel_px(WALL_CLOSE_M, grid.resolution)
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    k_thick = _odd_kernel_px(WALL_THICKEN_M, grid.resolution)
    wall = cv2.dilate(wall, np.ones((k_thick, k_thick), np.uint8), iterations=1)
    return wall > 0


def free_space_mask(grid, all_u, all_v, wall):
    observed = (grid.occupancy(all_u, all_v) * 255).astype(np.uint8)
    k = _odd_kernel_px(OBSERVED_CLOSE_M, grid.resolution)
    observed = cv2.morphologyEx(observed, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return (observed > 0) & ~wall
