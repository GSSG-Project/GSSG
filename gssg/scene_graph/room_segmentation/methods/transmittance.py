"""Splat-native transmittance floorplan (ours_v2).

Works on the Gaussian parameterization itself instead of a derived point cloud. Each
splat's 3D covariance is marginalized analytically along the vertical axis: the line
integral of an anisotropic Gaussian is a 2D Gaussian with the in-plane covariance
sub-block, weighted by opacity times the conditional vertical extent. Accumulated over
a knee-to-lintel height band and normalized by the band height, this gives per BEV cell
the alpha-weighted fraction of the band that is solid — a continuous occupancy field in
which floors are automatically transparent (thin vertical extent), walls automatically
opaque (they span the band), and furniture lands below one half (it fills only the
bottom of the band). No surface normals, no binarization of points, no morphology.

A column is blocked when the solid fraction exceeds 1/2 (a majority statement, not a
tuned threshold). Rooms are the persistence-filtered maxima of the free-space distance
field (h-maxima with h = half a standard door width) flooded by watershed: a region is
a separate room exactly when its interior clearance exceeds the clearance of its tightest
connection by more than half a doorway. The constants below are properties of buildings
and doors, not per-scene knobs.
"""

import numpy as np
from scipy import ndimage

from ..classify import _PLANE_AXES, HORIZONTAL, as_numpy, classify_splats
from ..cloud import band_fractions, normals_from_shape, vertical_marginals
from ..constants import OBSERVED_CLOSE_M, RESOLUTION_M, STOREY_BELOW_FLOOR_M
from ..debug_viz import save_storey_masks
from ..floors import detect_floor_heights
from ..grid import BEVGrid, _odd_kernel_px
from ..partition import _relabel_by_area, merge_small_regions
from ..types import Room, RoomSegmentationResult, Storey
from ..vectorize import vectorize_rooms

BAND_ABOVE_FLOOR_M = (0.5, 2.0)  # knee height .. door lintel
BAND_BELOW_CEILING_M = 0.2
BLOCKED_FRACTION = 0.5
PERSISTENCE_M = 0.35  # half a standard door width
MIN_ROOM_AREA_M2 = 1.0
MAX_KERNEL_RADIUS_PX = 34
MIN_WEIGHT = 1e-3


def _splat_bev(cu, cv, cov_a, cov_b, cov_c, weights, grid):
    """Accumulate w_i * exp(-0.5 * d^2) anisotropic kernels into the grid (torch,
    CUDA when available). Covariances are the (u,v) marginals in world units."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = grid.resolution

    def tensor(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)

    a, b, c = tensor(cov_a / res**2), tensor(cov_b / res**2), tensor(cov_c / res**2)
    det = torch.clamp(a * c - b * b, min=1e-8)
    ia, ib, ic = c / det, -b / det, a / det

    eig_max = (a + c) / 2 + torch.sqrt(((a - c) / 2) ** 2 + b * b)
    radius = torch.ceil(3.0 * torch.sqrt(eig_max)).long().clamp(1, MAX_KERNEL_RADIUS_PX)
    cx = tensor((cu - grid.u_min) / res)
    cy = tensor((cv - grid.v_min) / res)
    w = tensor(weights)

    image = torch.zeros(grid.height * grid.width, device=device, dtype=torch.float32)
    order = torch.argsort(radius)
    budget = 40_000_000
    start = 0
    while start < len(order):
        end = min(len(order), start + max(1, budget // int(2 * radius[order[-1]] + 1) ** 2))
        idx = order[start:end]
        r = int(radius[idx].max())
        offsets = torch.arange(-r, r + 1, device=device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dx, dy = dx.reshape(-1).float(), dy.reshape(-1).float()

        px = torch.round(cx[idx])[:, None] + dx[None, :]
        py = torch.round(cy[idx])[:, None] + dy[None, :]
        fx = px - cx[idx][:, None]
        fy = py - cy[idx][:, None]
        q = ia[idx][:, None] * fx * fx + 2 * ib[idx][:, None] * fx * fy + ic[idx][:, None] * fy * fy
        val = w[idx][:, None] * torch.exp(-0.5 * q)
        keep = (q <= 9.0) & (px >= 0) & (px < grid.width) & (py >= 0) & (py < grid.height)
        flat = (py * grid.width + px).long()
        image.scatter_add_(0, flat[keep], val[keep])
        start = end
    return image.reshape(grid.height, grid.width).cpu().numpy()


def _solid_fraction(u, v, h, sigma_h, sigma_cond, opacity, cov2d, band, grid):
    lo, hi = band
    fractions = band_fractions(h, sigma_h, lo, hi)
    weights = opacity * np.sqrt(2 * np.pi) * sigma_cond * fractions / max(hi - lo, 1e-6)
    keep = weights > MIN_WEIGHT
    if not keep.any():
        return np.zeros((grid.height, grid.width), dtype=np.float32)
    return _splat_bev(
        u[keep], v[keep], cov2d[0][keep], cov2d[1][keep], cov2d[2][keep], weights[keep], grid
    )


def _partition(free, resolution):
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    dist = ndimage.distance_transform_edt(free) * resolution
    markers, n = ndimage.label(h_maxima(dist, PERSISTENCE_M))
    if n == 0:
        if dist.max() <= 0:
            return np.zeros(free.shape, dtype=np.int32)
        markers = np.zeros(free.shape, dtype=np.int32)
        markers[np.unravel_index(int(dist.argmax()), dist.shape)] = 1
    labels = watershed(-dist, markers, mask=free, connectivity=1).astype(np.int32)
    labels = merge_small_regions(labels, resolution, min_area_m2=MIN_ROOM_AREA_M2)
    return _relabel_by_area(labels)


def segment(cloud, vertical_axis=2, output_dir=None) -> RoomSegmentationResult:
    result = RoomSegmentationResult()
    if cloud.scales is None or cloud.rotations is None or cloud.opacities is None:
        raise ValueError("method 'ours_v2' needs full splats (scale_*/rot_*/opacity fields)")
    xyz = as_numpy(cloud.xyz).reshape(-1, 3)
    if len(xyz) == 0:
        return result

    u_axis, v_axis = _PLANE_AXES[int(vertical_axis)]
    u, v, h = xyz[:, u_axis], xyz[:, v_axis], xyz[:, vertical_axis]
    opacity = as_numpy(cloud.opacities).reshape(-1)
    cov2d, sigma_h, sigma_cond = vertical_marginals(
        as_numpy(cloud.scales), as_numpy(cloud.rotations), u_axis, v_axis, int(vertical_axis)
    )

    normals = cloud.normals
    if normals is None:
        normals = normals_from_shape(as_numpy(cloud.scales), as_numpy(cloud.rotations))
    splats = classify_splats(xyz, normals, cloud.opacities, vertical_axis)
    floors = detect_floor_heights(splats.h[splats.label == HORIZONTAL], splats.h)

    next_room_id = 1
    for index, floor_h in enumerate(floors):
        ceiling = floors[index + 1] if index + 1 < len(floors) else np.inf
        in_storey = (h >= floor_h - STOREY_BELOW_FLOOR_M) & (h < ceiling - STOREY_BELOW_FLOOR_M)
        if not in_storey.any():
            continue
        su, sv = u[in_storey], v[in_storey]
        grid = BEVGrid.from_points(su, sv, RESOLUTION_M)
        band = (
            floor_h + BAND_ABOVE_FLOOR_M[0],
            min(floor_h + BAND_ABOVE_FLOOR_M[1], ceiling - BAND_BELOW_CEILING_M),
        )
        solid = _solid_fraction(
            su,
            sv,
            h[in_storey],
            sigma_h[in_storey],
            sigma_cond[in_storey],
            opacity[in_storey],
            tuple(c[in_storey] for c in cov2d),
            band,
            grid,
        )
        blocked = solid >= BLOCKED_FRACTION

        observed = (grid.occupancy(su, sv) * 255).astype(np.uint8)
        k = _odd_kernel_px(OBSERVED_CLOSE_M, grid.resolution)
        observed = ndimage.binary_closing(observed > 0, structure=np.ones((k, k), dtype=bool))
        free = observed & ~blocked

        labels = _partition(free, grid.resolution)
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

        if output_dir:
            save_storey_masks(output_dir, index, blocked, free, labels)
    return result
