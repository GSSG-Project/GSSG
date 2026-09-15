"""Correctness and speed tests for the voxel-grid seed filter (gssg/map/grid_filter.py).

Correctness is checked against an exact brute-force all-pairs reference; the speed
section compares against the pytorch3d knn_points path at the point counts
temp_points_filter sees.
"""

import time

import torch

from gssg.map.grid_filter import radius_inside_mask

MAX_RADIUS = 0.10  # matches base config max_radius


def brute_force_mask(query, ref, radius):
    d2 = torch.cdist(query, ref) ** 2
    return (d2 < radius.view(1, -1) ** 2).any(dim=1)


def _rand_cloud(n, scale, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand((n, 3), generator=g, device="cuda") * scale


def test_correctness():
    for nq, nr, scale, seed in [
        (1000, 5000, 2.0, 0),  # dense
        (2000, 500, 10.0, 1),  # sparse
        (1, 1, 1.0, 2),  # minimal
        (500, 3000, 0.3, 3),  # very dense (many candidates per cell)
    ]:
        q = _rand_cloud(nq, scale, seed)
        r = _rand_cloud(nr, scale, seed + 100)
        g = torch.Generator(device="cuda").manual_seed(seed + 200)
        radius = torch.rand((nr,), generator=g, device="cuda") * MAX_RADIUS
        got = radius_inside_mask(q, r, radius, max_radius=MAX_RADIUS)
        want = brute_force_mask(q, r, radius)
        assert torch.equal(got, want), (
            f"mismatch nq={nq} nr={nr} scale={scale}: {(got != want).sum().item()} rows differ"
        )
    # empties
    e = torch.empty((0, 3), device="cuda")
    assert radius_inside_mask(e, e, torch.empty(0, device="cuda"), MAX_RADIUS).shape[0] == 0
    print("  [ok] grid filter matches brute-force all-pairs on 4 regimes + empties")


def test_negative_coords():
    q = _rand_cloud(800, 4.0, 7) - 2.0  # straddles the origin (negative cells)
    r = _rand_cloud(4000, 4.0, 8) - 2.0
    radius = torch.full((4000,), 0.05, device="cuda")
    got = radius_inside_mask(q, r, radius, max_radius=MAX_RADIUS)
    want = brute_force_mask(q, r, radius)
    assert torch.equal(got, want), "mismatch with negative coordinates"
    print("  [ok] negative-coordinate cells handled")


def bench():
    from pytorch3d.ops import knn_points

    # Realistic temp_points_filter sizes: ~5-15k seeds vs 30-150k active gaussians.
    for nq, nr in [(5000, 30000), (10000, 100000), (15000, 200000)]:
        q = _rand_cloud(nq, 5.0, 11)
        r = _rand_cloud(nr, 5.0, 12)
        radius = torch.full((nr,), 0.06, device="cuda")

        def run_grid(q=q, r=r, radius=radius):
            return radius_inside_mask(q, r, radius, max_radius=MAX_RADIUS)

        def run_knn(q=q, r=r, radius=radius):
            nn_dist, nn_idx, _ = knn_points(q[None], r[None], norm=2, K=3, return_nn=True)
            return (torch.sqrt(nn_dist).squeeze(0) < radius[nn_idx.squeeze(0)]).any(dim=-1)

        for name, fn in [("grid", run_grid), ("knn", run_knn)]:
            fn()  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 10 * 1e3
            print(f"  [bench] {nq:6d} x {nr:7d}  {name:4s}: {ms:7.2f} ms")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print("Running grid-filter tests...")
    test_correctness()
    test_negative_coords()
    bench()
    print("All grid-filter tests passed.")
