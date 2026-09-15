"""Tests the two primitives behind the per-cell final pass: SubmapManager.set_resident_cells (make a chosen set of cells resident, the rest evicted, losslessly) and Mapping._keyframes_observing (select the keyframes whose frustum sees a cell's splats)."""

from types import SimpleNamespace

import torch

from gssg.map.mapper import Mapping
from gssg.map.submap import SubmapManager
from gssg.tests.submap_test import _args, _cloud_on_grid, _snapshot


def test_set_resident_cells_keeps_exactly_target():
    sm = SubmapManager(_args(submap_evict_tier="host"))
    cells = ((0, 0), (5, 0), (0, 5), (5, 5))
    cloud = _cloud_on_grid(_args(), cells=cells)
    mapping = SimpleNamespace(stable_gaussians=cloud)
    before = _snapshot(cloud)

    all_ids = sm.all_cell_ids(mapping)
    assert len(all_ids) == 4, all_ids
    keep = all_ids[:2]

    sm.set_resident_cells(mapping, keep)
    assert sorted(sm.resident_cell_ids(mapping)) == sorted(keep), "resident set != target"
    assert sorted(int(c) for c in sm._evicted) == sorted(all_ids[2:]), "wrong cells evicted"
    print(f"  [ok] set_resident_cells keeps exactly the target ({len(keep)} cells resident)")

    sm.set_resident_cells(mapping, all_ids)
    assert len(sm._evicted) == 0
    after = _snapshot(cloud)
    assert before.shape == after.shape and torch.allclose(before, after, atol=1e-4), (
        "round-trip through partial residency was not lossless"
    )
    print("  [ok] paging the full set back is lossless")


class _Frame:
    """Minimal stand-in exposing exactly what _stable_frustum_mask reads."""

    def __init__(self, origin, fx=500.0, W=640, H=480):
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor(origin, dtype=torch.float32)  # looks down +z from `origin`
        self._w2c = torch.linalg.inv(c2w).cuda()
        self._K = torch.tensor(
            [[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=torch.float32
        ).cuda()
        self.znear = 0.01
        self.image_width = W
        self.image_height = H

    def get_w2c(self):
        return self._w2c

    @property
    def get_intrinsic(self):
        return self._K


class _FakeMapper:
    _stable_frustum_mask = Mapping._stable_frustum_mask
    _keyframes_observing = Mapping._keyframes_observing

    def __init__(self, frames):
        self.keyframe_list = frames
        self.get_keyframe_num = len(frames)


def test_keyframes_observing_selects_viewers():
    # kf0 at origin looking +z sees the cluster ahead; kf1 sits 10 m away and looks past it.
    near = _Frame(origin=(0.0, 0.0, 0.0))
    far = _Frame(origin=(10.0, 10.0, 0.0))
    fm = _FakeMapper([near, far])
    xyz = torch.rand(500, 3, device="cuda") * 0.4 + torch.tensor([-0.2, -0.2, 1.5], device="cuda")

    obs = fm._keyframes_observing(xyz)
    assert obs == [0], f"expected only kf0 to observe the cluster, got {obs}"

    behind = torch.rand(500, 3, device="cuda") * 0.4 + torch.tensor([0.0, 0.0, -3.0], device="cuda")
    assert fm._keyframes_observing(behind) == [], "points behind every camera should be unobserved"
    assert fm._keyframes_observing(torch.empty(0, 3, device="cuda")) == [], "empty -> no keyframes"
    print("  [ok] _keyframes_observing selects only the cameras whose frustum sees the cell")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    test_set_resident_cells_keeps_exactly_target()
    test_keyframes_observing_selects_viewers()
    print("\nAll per-cell final-pass building-block tests passed.")
