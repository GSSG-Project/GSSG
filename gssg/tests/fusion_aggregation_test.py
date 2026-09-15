"""Tests that the confidence-weighted running mean (SceneGraph.add_object_observation and merge_objects) is order-invariant and yields the same normalized object embedding regardless of view or merge order."""

import sys
from types import SimpleNamespace

import numpy as np
import torch

from gssg.scene_graph.scene_graph import SceneGraph

DIM = 64
TOL = 1e-5


def _args():
    return SimpleNamespace(
        clip_model="custom",
        clip_dim=DIM,
        vertical_axis=2,
        near_wall_threshold=0.5,
        max_frame_embeddings=2000,
        multiview_aggregation=True,
    )


def _stored(sg, oid):
    v = sg.get_object_by_id(oid).features.float().reshape(-1)
    return torch.nn.functional.normalize(v, dim=-1)


def _expected(views):
    s = torch.zeros(DIM)
    W = 0.0
    for v, w in views:
        s = s + w * torch.nn.functional.normalize(v.float(), dim=-1)
        W += w
    return torch.nn.functional.normalize(s / W, dim=-1)


def _make_views(n, rng):
    return [
        (torch.from_numpy(rng.normal(size=DIM)).float(), float(rng.uniform(0.3, 1.0)))
        for _ in range(n)
    ]


def test_running_mean_matches_closed_form():
    rng = np.random.default_rng(0)
    views = _make_views(8, rng)
    sg = SceneGraph(_args())
    sg.add_object(1)
    for v, w in views:
        sg.add_object_observation(1, v, weight=w)
    assert torch.allclose(_stored(sg, 1), _expected(views), atol=TOL)
    assert sg.get_object_by_id(1).view_count == 8
    print("PASS running_mean_matches_closed_form")


def test_view_order_invariance():
    rng = np.random.default_rng(1)
    views = _make_views(10, rng)
    embs = []
    for perm_seed in (10, 20, 30):
        order = list(np.random.default_rng(perm_seed).permutation(len(views)))
        sg = SceneGraph(_args())
        sg.add_object(1)
        for i in order:
            sg.add_object_observation(1, views[i][0], weight=views[i][1])
        embs.append(_stored(sg, 1))
    for e in embs[1:]:
        assert torch.allclose(e, embs[0], atol=TOL)
    assert torch.allclose(embs[0], _expected(views), atol=TOL)
    print("PASS view_order_invariance")


def test_merge_associativity():
    """Splitting views across two objects then merging equals observing them all on one."""
    rng = np.random.default_rng(2)
    views = _make_views(12, rng)
    a_views, b_views = views[:7], views[7:]

    sg = SceneGraph(_args())
    sg.add_object(1)
    sg.add_object(2)
    sg.get_object_by_id(1).size = len(a_views)
    sg.get_object_by_id(2).size = len(b_views)
    sg.get_object_by_id(1).add_camera_pose(np.eye(4), 0)
    sg.get_object_by_id(2).add_camera_pose(2 * np.eye(4), 1)
    for v, w in a_views:
        sg.add_object_observation(1, v, weight=w)
    for v, w in b_views:
        sg.add_object_observation(2, v, weight=w)
    sg.merge_objects(2, 1)

    assert torch.allclose(_stored(sg, 1), _expected(views), atol=TOL)
    assert sg.get_object_by_id(1).view_count == 12
    assert sg.get_object_by_id(2) is None
    print("PASS merge_associativity")


def test_disabled_flag_uses_legacy_overwrite():
    rng = np.random.default_rng(3)
    a = _args()
    a.multiview_aggregation = False
    sg = SceneGraph(a)
    sg.add_object(1)
    v1 = torch.from_numpy(rng.normal(size=DIM)).float()
    v2 = torch.from_numpy(rng.normal(size=DIM)).float()
    sg.set_object_features(1, v1, 0.9)
    sg.set_object_features(1, v2, 0.9)  # overwrites
    assert torch.allclose(_stored(sg, 1), torch.nn.functional.normalize(v2, dim=-1), atol=TOL)
    print("PASS disabled_flag_uses_legacy_overwrite")


if __name__ == "__main__":
    test_running_mean_matches_closed_form()
    test_view_order_invariance()
    test_merge_associativity()
    test_disabled_flag_uses_legacy_overwrite()
    print("\nAll fusion_aggregation tests passed.")
    sys.exit(0)
