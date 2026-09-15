"""Fixed-radius neighbor test via a voxel hash grid (pure torch, O(N+M)).

Quantizes reference points into cubic cells with edge >= the largest test radius,
sorts by linearized cell key, then for each query point binary-searches the 27
neighboring cells and does exact distance-vs-radius checks on the gathered
candidates only.
"""

from __future__ import annotations

import torch

# Cell coordinates are offset to non-negative and packed into one int64 key:
# 21 bits per axis (+-2^20 cells; at ~6 cm cells that is a +-60 km scene bound).
_COORD_BITS = 21
_COORD_OFFSET = 1 << (_COORD_BITS - 1)


def _cell_keys(coords: torch.Tensor) -> torch.Tensor:
    c = coords + _COORD_OFFSET
    return (c[:, 0] << (2 * _COORD_BITS)) | (c[:, 1] << _COORD_BITS) | c[:, 2]


@torch.no_grad()
def radius_inside_mask(
    query_xyz: torch.Tensor,
    ref_xyz: torch.Tensor,
    ref_radius: torch.Tensor,
    max_radius: float,
) -> torch.Tensor:
    """Boolean [Nq]: query point lies within `ref_radius[j]` of any reference point j.

    `max_radius` must upper-bound every entry of `ref_radius` (it sets the cell
    size; radii above it would need a wider neighborhood than the 27 cells searched).
    """
    nq, nr = query_xyz.shape[0], ref_xyz.shape[0]
    if nq == 0 or nr == 0:
        return torch.zeros(nq, dtype=torch.bool, device=query_xyz.device)

    cell = max(float(max_radius), 1e-6)
    ref_cells = torch.floor(ref_xyz / cell).long()
    ref_keys = _cell_keys(ref_cells)
    order = torch.argsort(ref_keys)
    ref_keys_sorted = ref_keys[order]
    ref_xyz_sorted = ref_xyz[order]
    ref_radius_sorted = ref_radius.view(-1)[order]

    q_cells = torch.floor(query_xyz / cell).long()

    # 27-neighborhood offsets, applied to every query cell at once.
    rng = torch.arange(-1, 2, device=query_xyz.device)
    offsets = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1).reshape(-1, 3)
    nbr_keys = _cell_keys((q_cells.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3))  # [Nq*27]

    starts = torch.searchsorted(ref_keys_sorted, nbr_keys, right=False)
    ends = torch.searchsorted(ref_keys_sorted, nbr_keys, right=True)
    counts = ends - starts
    total = int(counts.sum().item())
    inside = torch.zeros(nq, dtype=torch.bool, device=query_xyz.device)
    if total == 0:
        return inside

    # Expand each (query, cell) range into flat candidate pairs.
    pair_q = torch.repeat_interleave(
        torch.arange(nq, device=query_xyz.device).repeat_interleave(27), counts
    )
    cum = torch.cumsum(counts, dim=0) - counts  # exclusive prefix sum
    pair_r = (
        torch.repeat_interleave(starts, counts)
        + torch.arange(total, device=query_xyz.device)
        - torch.repeat_interleave(cum, counts)
    )

    d2 = ((query_xyz[pair_q] - ref_xyz_sorted[pair_r]) ** 2).sum(dim=-1)
    hit = d2 < ref_radius_sorted[pair_r] ** 2
    inside.index_put_((pair_q[hit],), torch.ones_like(pair_q[hit], dtype=torch.bool))
    return inside
