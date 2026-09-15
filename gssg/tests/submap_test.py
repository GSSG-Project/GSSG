"""Correctness tests for submapping (gssg/map/submap.py).

Guarantees checked:
  1. cell_of is axis-aware (ground plane = the two non-vertical axes).
  2. evict -> page-in is lossless (the stable cloud's row set is preserved exactly).
  3. make_full_resident reconstructs the whole cloud after multiple evictions.
  4. residency keeps the camera's cell and drops far cells; counts include evicted.
"""

from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from gssg.map.gaussian_pointcloud import GaussianPointCloud
from gssg.map.submap import SubmapManager


def _args(**kw):
    base = dict(
        init_opacity=0.99,
        scale_factor=0.5,
        min_radius=0.01,
        max_radius=0.10,
        max_sh_degree=4,
        active_sh_degree=-1,
        xyz_factor=[1, 1, 1],
        vertical_axis=2,
        max_depth=6.0,
        submap_cell_size=2.0,
        submap_radius=0.0,
        submap_evict_tier="host",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _cloud_on_grid(args, n_per_cell=40, cells=((0, 0), (5, 0), (0, 5), (5, 5))):
    """Stable cloud with n_per_cell points planted in each named (u,v) ground cell
    (vertical_axis=2 -> plane is x,y; z is vertical)."""
    c = GaussianPointCloud(args, scene_graph=None, name="stable")
    cs = args.submap_cell_size
    xs, ns, cols, sems = [], [], [], []
    g = torch.Generator(device="cuda").manual_seed(0)
    for ci, (u, v) in enumerate(cells):
        p = torch.rand((n_per_cell, 3), generator=g, device="cuda")
        p[:, 0] = u * cs + p[:, 0] * cs  # x in cell u
        p[:, 1] = v * cs + p[:, 1] * cs  # y in cell v
        p[:, 2] = p[:, 2] * 0.5  # z (vertical) small
        xs.append(p)
        ns.append(
            torch.nn.functional.normalize(
                torch.rand((n_per_cell, 3), generator=g, device="cuda"), dim=-1
            )
        )
        cols.append(torch.rand((n_per_cell, 3), generator=g, device="cuda"))
        sems.append(torch.full((n_per_cell, 1), float(ci), device="cuda"))
    c.add_empty_points(
        torch.cat(xs), torch.cat(ns), torch.cat(cols), time=1, semantic=torch.cat(sems)
    )
    return c


def _snapshot(cloud):
    """Sorted full-attribute fingerprint, order-independent (page-in reorders rows)."""
    keyset = GaussianPointCloud._BLOB_ATTRS
    cat = torch.cat(
        [getattr(cloud, a).reshape(cloud.get_points_num, -1).float() for a in keyset], dim=1
    )
    order = torch.argsort(cat[:, 0] * 1e6 + cat[:, 1] * 1e3 + cat[:, 2])
    return cat[order].cpu()


def test_cell_of_axis_aware():
    # Z-up: moving along z (axis 2) must NOT change the cell; moving x/y must.
    for vax, vert in [(2, 2), (1, 1), (0, 0)]:
        sm = SubmapManager(_args(vertical_axis=vax))
        base = torch.tensor([[1.0, 1.0, 1.0]], device="cuda")
        moved_vert = base.clone()
        moved_vert[0, vert] += 100.0  # large move along the vertical axis
        assert sm.cell_of(base).item() == sm.cell_of(moved_vert).item(), (
            f"vertical_axis={vax}: cell changed when moving along the vertical axis"
        )
        a, b = sm._plane_axes()
        moved_plane = base.clone()
        moved_plane[0, a] += 100.0
        assert sm.cell_of(base).item() != sm.cell_of(moved_plane).item(), (
            f"vertical_axis={vax}: cell unchanged when moving in the ground plane"
        )
    print("  [ok] cell_of is axis-aware (ground plane = non-vertical axes)")


def test_evict_pagein_lossless():
    args = _args()
    cloud = _cloud_on_grid(args)
    sm = SubmapManager(args)
    before = _snapshot(cloud)
    n0 = cloud.get_points_num

    cells = sm.cell_of(cloud.get_xyz)
    target = int(torch.unique(cells)[0].item())
    mask = cells == target
    n_cell = int(mask.sum().item())

    sm._evict(cloud, target, mask)
    assert cloud.get_points_num == n0 - n_cell, "evict did not remove the cell's rows"
    assert sm.total_stable_count(cloud.get_points_num) == n0, (
        "evicted points missing from total count"
    )
    assert target in sm._evicted

    sm._page_in(cloud, target)
    assert cloud.get_points_num == n0, "page-in did not restore the row count"
    after = _snapshot(cloud)
    assert torch.allclose(before, after, atol=1e-6), "evict->page-in changed the cloud (lossy!)"
    print(f"  [ok] evict -> page-in is lossless ({n_cell} rows round-tripped, set identical)")


def test_full_resident_roundtrip():
    args = _args()
    cloud = _cloud_on_grid(args)
    sm = SubmapManager(args)
    before = _snapshot(cloud)
    n0 = cloud.get_points_num

    # Evict every cell, then restore.
    for cell_t in torch.unique(sm.cell_of(cloud.get_xyz)).tolist():
        m = sm.cell_of(cloud.get_xyz) == cell_t
        sm._evict(cloud, int(cell_t), m)
    assert cloud.get_points_num == 0, "not all cells evicted"
    assert sm.total_stable_count(0) == n0

    sm.make_full_resident(SimpleNamespace(stable_gaussians=cloud))
    assert cloud.get_points_num == n0 and not sm._evicted
    assert torch.allclose(before, _snapshot(cloud), atol=1e-6), "full residency changed the cloud"
    print("  [ok] make_full_resident reconstructs the whole cloud after evicting all cells")


def test_disk_tier_roundtrip():
    import tempfile

    args = _args(submap_evict_tier="disk", submap_page_dir=tempfile.mkdtemp())
    cloud = _cloud_on_grid(args)
    sm = SubmapManager(args)
    before = _snapshot(cloud)
    cell_t = int(torch.unique(sm.cell_of(cloud.get_xyz))[0].item())
    sm._evict(cloud, cell_t, sm.cell_of(cloud.get_xyz) == cell_t)
    assert "path" in sm._evicted[cell_t], "disk tier did not write a blob path"
    sm._page_in(cloud, cell_t)
    assert torch.allclose(before, _snapshot(cloud), atol=1e-6), "disk evict->page-in lossy"
    print("  [ok] disk-tier evict -> page-in lossless (torch.save blob)")


class _StubCam:
    def __init__(self, pos):
        self._c2w = torch.eye(4, device="cuda")
        self._c2w[:3, 3] = torch.tensor(pos, device="cuda", dtype=torch.float32)

    @property
    def get_c2w(self):
        return self._c2w


def test_residency_keeps_near_drops_far():
    args = _args(submap_cell_size=2.0, max_depth=6.0)  # radius auto = 8.0
    cloud = _cloud_on_grid(args, cells=((0, 0), (50, 0)))  # one cell at origin, one ~100 m away
    sm = SubmapManager(args)
    mapping = SimpleNamespace(
        stable_gaussians=cloud, processed_frames=deque(), keyframe_list=[], global_keyframe_num=3
    )
    frame = _StubCam([1.0, 1.0, 0.0])  # camera in the origin cell

    n0 = cloud.get_points_num
    far_cell = sm.cell_of(torch.tensor([[100.0, 0.5, 0.0]], device="cuda")).item()
    near_cell = sm.cell_of(torch.tensor([[1.0, 1.0, 0.0]], device="cuda")).item()

    sm.update_residency(mapping, frame)  # camera in the near cell
    assert far_cell in sm._evicted, "far cell (~100 m) was not evicted"
    assert near_cell not in sm._evicted, "near cell (under camera) was wrongly evicted"
    assert cloud.get_points_num < n0, "nothing evicted"
    assert sm.total_stable_count(cloud.get_points_num) == n0, "count lost points on eviction"

    # Move the camera to the far cell -> it pages back; the now-far near cell evicts.
    sm.update_residency(mapping, _StubCam([100.0, 0.5, 0.0]))
    assert far_cell not in sm._evicted, "far cell did not page back in on approach"
    assert near_cell in sm._evicted, "departed cell did not evict"
    assert sm.total_stable_count(cloud.get_points_num) == n0, "count drifted after page-in"
    print("  [ok] residency evicts far cells, pages them back on approach, count preserved")


def test_hybrid_spill_lru():
    """Hybrid tier keeps cells in RAM until over budget, then spills LRU cells to disk;
    the whole set still pages back losslessly."""
    import tempfile

    args = _args(submap_evict_tier="hybrid", submap_page_dir=tempfile.mkdtemp())
    cloud = _cloud_on_grid(args)  # 4 cells
    sm = SubmapManager(args)
    sm._host_budget = 1  # 1 byte -> every cell exceeds budget and must spill
    before = _snapshot(cloud)
    n0 = cloud.get_points_num

    for cell_t in torch.unique(sm.cell_of(cloud.get_xyz)).tolist():
        m = sm.cell_of(cloud.get_xyz) == cell_t
        sm._evict(cloud, int(cell_t), m)

    on_disk = [c for c, b in sm._evicted.items() if isinstance(b, dict) and "path" in b]
    assert sm.n_spills >= 1 and len(on_disk) >= 1, "hybrid did not spill to disk under budget"
    assert sm._host_bytes <= sm._host_budget, "host bytes exceed budget after spill"
    assert sm.total_stable_count(cloud.get_points_num) == n0, "count lost on spill"

    sm.make_full_resident(SimpleNamespace(stable_gaussians=cloud))
    assert cloud.get_points_num == n0 and not sm._evicted
    assert torch.allclose(before, _snapshot(cloud), atol=1e-6), "hybrid round-trip lossy"
    print(f"  [ok] hybrid tier spills LRU cells to disk under budget ({sm.n_spills}), lossless")


def test_streaming_export_roundtrip():
    """save_full_stable_ply streams resident + evicted cells to one PLY without full
    residency; the exported vertex set equals the whole map."""
    import os
    import tempfile

    from plyfile import PlyData

    args = _args()
    cloud = _cloud_on_grid(args)
    sm = SubmapManager(args)
    n0 = cloud.get_points_num
    orig_xyz = cloud.get_xyz.detach().cpu().numpy()

    cells = torch.unique(sm.cell_of(cloud.get_xyz)).tolist()
    for cell_t in cells[: max(1, len(cells) // 2)]:  # leave some resident
        m = sm.cell_of(cloud.get_xyz) == cell_t
        sm._evict(cloud, int(cell_t), m)
    assert sm.n_evicted() > 0 and cloud.get_points_num > 0, "need both resident and evicted"

    path = os.path.join(tempfile.mkdtemp(), "stable_full.ply")
    n = sm.save_full_stable_ply(cloud, path, include_confidence=True)
    assert n == n0, f"streamed {n} verts, expected {n0}"

    verts = PlyData.read(path)["vertex"]
    assert len(verts) == n0, "PLY vertex count mismatch"
    px = np.stack([verts["x"], verts["y"], verts["z"]], axis=1)

    def keysort(a):
        return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]

    assert np.allclose(keysort(px), keysort(orig_xyz), atol=1e-4), "exported xyz != original"
    print(f"  [ok] streaming export round-trip: {n} verts ({sm.evicted_points()} evicted)")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print("Running submapping tests...")
    test_cell_of_axis_aware()
    test_evict_pagein_lossless()
    test_full_resident_roundtrip()
    test_disk_tier_roundtrip()
    test_residency_keeps_near_drops_far()
    test_hybrid_spill_lru()
    test_streaming_export_roundtrip()
    print("All submapping tests passed.")
