"""Tests adaptive iteration scheduling and the version-keyed render-param cache."""

from types import SimpleNamespace

import torch

from gssg.map.gaussian_pointcloud import GaussianPointCloud
from gssg.map.mapper import Mapping, adaptive_iter_count


def test_adaptive_iter_count():
    base = 20
    # High novelty (many new points) -> full budget
    assert adaptive_iter_count(base, 2000, 5000, 0.0, 0.25) == base
    # High novelty via transmission ratio alone -> full budget
    assert adaptive_iter_count(base, 0, 5000, 0.15, 0.25) == base
    # Zero novelty -> floor at min_ratio
    assert adaptive_iter_count(base, 0, 5000, 0.0, 0.25) == 5
    # Mid novelty -> between floor and full
    mid = adaptive_iter_count(base, 750, 5000, 0.0, 0.25)
    assert 5 < mid < base, f"expected mid-range, got {mid}"
    # Hard floor of 4 even with tiny base
    assert adaptive_iter_count(5, 0, 5000, 0.0, 0.1) == 4
    print("  [ok] adaptive_iter_count scaling and floors")


def _cloud_args():
    return SimpleNamespace(
        init_opacity=0.99,
        scale_factor=0.5,
        min_radius=0.01,
        max_radius=0.10,
        max_sh_degree=3,
        active_sh_degree=0,
        xyz_factor=[1, 1, 1],
    )


def _filled_cloud(n, seed=0):
    c = GaussianPointCloud(_cloud_args(), scene_graph=None, name="t")
    g = torch.Generator(device="cuda").manual_seed(seed)
    xyz = torch.rand((n, 3), generator=g, device="cuda")
    normal = torch.nn.functional.normalize(torch.rand((n, 3), generator=g, device="cuda"), dim=-1)
    color = torch.rand((n, 3), generator=g, device="cuda")
    sem = torch.zeros((n, 1), device="cuda")
    c.add_empty_points(xyz, normal, color, time=0, semantic=sem)
    return c


def test_params_cache():
    m = object.__new__(Mapping)  # cache logic only; skip full __init__
    cloud = _filled_cloud(500)
    m.stable_gaussians = cloud

    p1 = m._cached_params(cloud, "_stable_params_cache")
    p2 = m._cached_params(cloud, "_stable_params_cache")
    assert p1 is p2, "unchanged cloud must serve the cached dict"

    # Mutation via cat invalidates
    extra = _filled_cloud(100, seed=1)
    cloud.cat(extra.remove(torch.ones(100, dtype=torch.bool, device="cuda")))
    p3 = m._cached_params(cloud, "_stable_params_cache")
    assert p3 is not p2, "cat must invalidate the cache"
    assert p3["xyz"].shape[0] == 600

    # Mutation via delete invalidates
    mask = torch.zeros(600, dtype=torch.bool, device="cuda")
    mask[:50] = True
    cloud.delete(mask)
    p4 = m._cached_params(cloud, "_stable_params_cache")
    assert p4 is not p3 and p4["xyz"].shape[0] == 550

    # While parametrized: always fresh, never cached
    cloud.parametrize(
        SimpleNamespace(
            position_lr=0.01, feature_lr=0.01, opacity_lr=0.01, scaling_lr=0.01, rotation_lr=0.01
        )
    )
    f1 = m._cached_params(cloud, "_stable_params_cache")
    f2 = m._cached_params(cloud, "_stable_params_cache")
    assert f1 is not f2, "parametrized cloud must be served fresh each call"
    cloud.detach()
    d1 = m._cached_params(cloud, "_stable_params_cache")
    d2 = m._cached_params(cloud, "_stable_params_cache")
    assert d1 is d2, "after detach the cache must be re-enabled"

    # Direct-assign mutation paths must bump explicitly (history_merge pattern)
    cloud._xyz = cloud._xyz + 1.0
    cloud.bump_version()
    d3 = m._cached_params(cloud, "_stable_params_cache")
    assert d3 is not d2 and torch.allclose(d3["xyz"], d2["xyz"] + 1.0)
    print("  [ok] param cache: hit, cat/delete invalidation, parametrize bypass, bump")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print("Running mapper optimization tests...")
    test_adaptive_iter_count()
    test_params_cache()
    print("All mapper optimization tests passed.")
