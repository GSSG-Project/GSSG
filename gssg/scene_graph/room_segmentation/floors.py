"""Automatic floor-height detection (one per storey).

Histogram the heights of horizontal splats: floors and ceilings show up as sharp peaks.
Splat normals carry no reliable sign, so floors and ceilings are told apart by structure:
walking up from the lowest peak, a peak >= 2 m above the current floor is a ceiling; if the
very next peak sits within one slab thickness above it, that next peak is the floor of the
storey above. A peak only counts as a floor when the wall band above it actually contains
scene mass — a top ceiling has nothing above and is rejected.
"""

import numpy as np

from .constants import (
    FLOOR_BAND_MIN_MASS,
    FLOOR_BIN_M,
    FLOOR_FALLBACK_PERCENTILE,
    FLOOR_PEAK_MIN_FRAC,
    MIN_HORIZONTAL_SPLATS,
    MIN_STOREY_SEPARATION_M,
    PEAK_NMS_SEPARATION_M,
    SLAB_PAIR_MAX_M,
    WALL_BAND_ABOVE_FLOOR_M,
)


def _band_mass(all_heights, floor_h):
    lo, hi = WALL_BAND_ABOVE_FLOOR_M
    return int(np.count_nonzero((all_heights > floor_h + lo) & (all_heights < floor_h + hi)))


def _histogram_peaks(heights):
    """Sorted heights of prominent local maxima, double-peaks collapsed."""
    lo = float(np.percentile(heights, 0.5))
    hi = float(np.percentile(heights, 99.5))
    n_bins = max(int(np.ceil((hi - lo) / FLOOR_BIN_M)), 1)
    hist, edges = np.histogram(heights, bins=n_bins, range=(lo, lo + n_bins * FLOOR_BIN_M))
    smooth = np.convolve(hist.astype(np.float64), np.ones(3) / 3.0, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2.0

    threshold = FLOOR_PEAK_MIN_FRAC * smooth.max()
    candidates = [
        i
        for i in range(len(smooth))
        if smooth[i] >= threshold and smooth[i] == smooth[max(0, i - 1) : i + 2].max()
    ]
    selected = []
    for i in sorted(candidates, key=lambda i: -smooth[i]):
        if all(abs(centers[i] - centers[j]) >= PEAK_NMS_SEPARATION_M for j in selected):
            selected.append(i)
    return sorted(float(centers[i]) for i in selected)


def detect_floor_heights(horizontal_heights, all_heights):
    """Returns sorted floor heights, one per detected storey (>= 1 if any points exist)."""
    if all_heights.size == 0:
        return []
    if horizontal_heights.size < MIN_HORIZONTAL_SPLATS:
        return [float(np.percentile(all_heights, FLOOR_FALLBACK_PERCENTILE))]

    peaks = _histogram_peaks(horizontal_heights)
    if not peaks:
        return [float(np.percentile(all_heights, FLOOR_FALLBACK_PERCENTILE))]

    min_mass = FLOOR_BAND_MIN_MASS * all_heights.size
    floors = [peaks[0]]
    i = 1
    while i < len(peaks):
        if peaks[i] - floors[-1] < MIN_STOREY_SEPARATION_M:
            i += 1  # same-storey surface (table, bed, counter)
            continue
        if i + 1 < len(peaks) and peaks[i + 1] - peaks[i] <= SLAB_PAIR_MAX_M:
            candidate, i = peaks[i + 1], i + 2  # ceiling + slab -> next storey's floor
        else:
            candidate, i = peaks[i], i + 1  # lone peak: next floor only if scene mass above
        if _band_mass(all_heights, candidate) >= min_mass:
            floors.append(candidate)
    return floors
