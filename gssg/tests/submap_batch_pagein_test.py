"""Tests that a single batched page-in of several cells reconstructs the original cloud exactly and accounts for every cell."""

import torch

from gssg.map.submap import SubmapManager
from gssg.tests.submap_test import _args, _cloud_on_grid, _snapshot


def test_batched_pagein_is_lossless_and_counted():
    sm = SubmapManager(_args(submap_evict_tier="host"))
    cells = ((0, 0), (5, 0), (0, 5), (5, 5))
    cloud = _cloud_on_grid(sm.args if hasattr(sm, "args") else _args(), cells=cells)
    before = _snapshot(cloud)

    # Evict three of the four cells (leave one resident), then page all three back in one call.
    cell_ids = []
    for u, v in cells[:3]:
        probe = torch.tensor([[u * 2.0 + 1.0, v * 2.0 + 1.0, 0.25]], device="cuda")
        cid = int(sm.cell_of(probe)[0].item())
        m = sm.cell_of(cloud.get_xyz) == cid
        sm._evict(cloud, cid, m)
        cell_ids.append(cid)

    assert len(sm._evicted) == 3, f"expected 3 evicted cells, got {len(sm._evicted)}"
    n0 = sm.n_pageins
    sm._page_in_many(cloud, cell_ids)

    assert len(sm._evicted) == 0, "evicted cells remained after batched page-in"
    assert sm.n_pageins == n0 + 3, f"n_pageins should count each cell: {sm.n_pageins} != {n0 + 3}"
    after = _snapshot(cloud)
    assert before.shape == after.shape, f"point count changed: {before.shape} -> {after.shape}"
    assert torch.allclose(before, after, atol=1e-4), "batched page-in did not reconstruct the cloud"
    print(f"  [ok] batched page-in: 3 cells in one cat, lossless ({after.shape[0]} pts), counted")


def test_empty_batch_is_noop():
    sm = SubmapManager(_args())
    cloud = _cloud_on_grid(_args(), cells=((0, 0),))
    n0 = sm.n_pageins
    sm._page_in_many(cloud, [])
    sm._page_in_many(cloud, [999])  # not evicted -> filtered out
    assert sm.n_pageins == n0, "empty/invalid batch must be a no-op"
    print("  [ok] empty / non-evicted batch is a no-op")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    test_batched_pagein_is_lossless_and_counted()
    test_empty_batch_is_noop()
    print("\nAll batched page-in tests passed.")
