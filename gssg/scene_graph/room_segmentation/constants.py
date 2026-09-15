"""Internal constants of the room segmenter.

These describe properties of buildings, doors and the reconstruction rather than of
datasets, so they are constants rather than config knobs. The only external input the
segmenter takes is `vertical_axis`.
"""

# --- splat filtering / classification ---
MIN_OPACITY = 0.2  # below this a splat is a half-converged ghost, not a surface
FLOOR_MAX_TILT_DEG = 20.0  # surface within 20 deg of horizontal -> floor/ceiling candidate
WALL_MAX_TILT_DEG = 20.0  # normal within 20 deg of horizontal -> vertical surface (wall-like)

# --- floor (storey) detection ---
FLOOR_BIN_M = 0.05  # height-histogram bin, ~ floor reconstruction noise
FLOOR_PEAK_MIN_FRAC = 0.15  # a histogram peak must reach 15% of the tallest peak
PEAK_NMS_SEPARATION_M = 0.3  # collapse double-peaks of the same physical surface
MIN_STOREY_SEPARATION_M = 2.0  # ceilings/next floors are never closer to a floor than this
SLAB_PAIR_MAX_M = 1.0  # ceiling and the next storey's floor sit within one slab thickness
FLOOR_BAND_MIN_MASS = 0.03  # a real floor has >=3% of all splats in its wall band
MIN_HORIZONTAL_SPLATS = 50  # fewer -> fall back to a height percentile for the floor
FLOOR_FALLBACK_PERCENTILE = 2.0

# --- wall rasterization ---
RESOLUTION_M = 0.03  # BEV cell size; 3 cm resolves interior walls (>= 7 cm thick)
PADDING_M = 0.5  # grid margin around the scanned extent
ROBUST_BOUNDS_PERCENTILE = 0.5  # ignore the most extreme 0.5% outlier splats for grid extent
WALL_BAND_ABOVE_FLOOR_M = (0.3, 2.0)  # skirting/furniture below, door lintels above
WALL_SUBBANDS = 3  # wall cell = occupied in >=2 of 3 height sub-bands (kills furniture sides)
WALL_SUBBAND_MIN_VOTES = 2
MIN_WALL_CELLS = 5  # a real wall fragment spans >=5 connected cells (~15 cm); smaller
#   isolated blobs are floater splats with random normals
WALL_CLOSE_M = 0.15  # bridge sampling gaps in walls; far below any door width (~0.7 m)
WALL_THICKEN_M = 0.09  # thicken walls so the watershed cannot tunnel diagonally
OBSERVED_CLOSE_M = 0.21  # fill sampling holes in the observed-area mask
STOREY_BELOW_FLOOR_M = 0.5  # a storey's splats start slightly below its detected floor

# --- room partition (wall-inflation dilation sweep) ---
SWEEP_RADII_M = (0.45, 1.2)  # half a doorway .. radius of a small room
SWEEP_STEPS = 8
MIN_SEED_AREA_M2 = 0.8  # a room core smaller than this is noise
FALLBACK_MIN_CORE_M = 0.3  # no sweep seeds at all -> still seed one room if this much core
MIN_ROOM_AREA_M2 = 5.0  # smaller regions merge into their largest-border neighbor
MIN_FREE_AREA_M2 = 1.0  # less observed free space than this -> no rooms

# --- vectorization ---
POLY_SIMPLIFY_FRAC = 0.005  # approxPolyDP epsilon as a fraction of contour length
