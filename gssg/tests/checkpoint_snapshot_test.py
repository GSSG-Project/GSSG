"""Tests that GaussianPointCloud.clone_to_cpu yields an independent host snapshot and that save_model_ply works on it."""

import os
import tempfile
from types import SimpleNamespace

import torch
from plyfile import PlyData

from gssg.map.gaussian_pointcloud import GaussianPointCloud


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
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _cloud(n=128):
    c = GaussianPointCloud(_args(), scene_graph=None, name="stable")
    g = torch.Generator(device="cuda").manual_seed(0)
    xyz = torch.rand((n, 3), generator=g, device="cuda")
    nrm = torch.nn.functional.normalize(torch.rand((n, 3), generator=g, device="cuda"), dim=-1)
    col = torch.rand((n, 3), generator=g, device="cuda")
    sem = torch.randint(0, 5, (n, 1), generator=g, device="cuda").float()
    c.add_empty_points(xyz, nrm, col, time=1, semantic=sem)
    return c


def test_clone_is_independent_cpu_snapshot():
    c = _cloud()
    snap = c.clone_to_cpu()
    for name, val in vars(snap).items():
        if torch.is_tensor(val):
            assert val.device.type == "cpu", f"snapshot tensor {name} is not on cpu"
    assert snap.get_points_num == c.get_points_num
    before = snap._xyz.clone()
    # Mutate the live cloud in place and by reassignment; the snapshot must not move.
    c._xyz.add_(1.0)
    c._xyz = torch.zeros_like(c._xyz)
    assert torch.allclose(snap._xyz, before), "snapshot aliased the live tensor (in-place leak)"
    assert not torch.allclose(snap._xyz.cuda(), c._xyz), "snapshot tracked a live reassignment"
    print("  [ok] clone_to_cpu: independent host snapshot, immune to live mutation")


def test_snapshot_writes_valid_ply():
    c = _cloud(n=200)
    snap = c.clone_to_cpu()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "snap.ply")
        snap.save_model_ply(p, include_confidence=True, include_anchor=True)
        assert os.path.exists(p), "snapshot PLY not written"
        ply = PlyData.read(p)
        assert ply["vertex"].count == 200, f"wrong vertex count: {ply['vertex'].count}"
    print("  [ok] save_model_ply on CPU snapshot -> valid PLY (200 verts)")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs CUDA (matches the other gssg tests)"
    test_clone_is_independent_cpu_snapshot()
    test_snapshot_writes_valid_ply()
    print("\nAll checkpoint-snapshot tests passed.")
