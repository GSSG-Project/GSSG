"""Tests gssg.scene_graph.room_segmentation on synthetic apartments with known geometry,
checking storey detection, room count, polygon areas and point-in-room assignment across
all methods."""

import json
import os
import sys
import tempfile

import numpy as np
from shapely.geometry import Point

from gssg.scene_graph.room_segmentation import (
    StructuralRoomSegmenter,
    segment_rooms_from_arrays,
    write_floorplan,
)

# canonical scene is z-up: u->x, v->y, h->z; remap places (u, v, h) on the right axes
_PLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

WALL_H = 2.5
FLOOR_Z = -2.75


def _wall_segment(u0, v0, u1, v1, normal_uv, h0, h1, door_v=None, door_top=2.0, step=0.025):
    length = float(np.hypot(u1 - u0, v1 - v0))
    t = np.arange(0, length, step) / max(length, 1e-9)
    u = u0 + t * (u1 - u0)
    v = v0 + t * (v1 - v0)
    heights = np.arange(0.0, h1 - h0, 0.05)
    uu = np.repeat(u, len(heights))
    vv = np.repeat(v, len(heights))
    hh = h0 + np.tile(heights, len(u))
    if door_v is not None:
        in_door = (np.abs(vv - door_v) < 0.45) & (hh - h0 < door_top)
        uu, vv, hh = uu[~in_door], vv[~in_door], hh[~in_door]
    n = np.tile(np.array([normal_uv[0], normal_uv[1], 0.0]), (len(uu), 1))
    return np.stack([uu, vv, hh], axis=1), n


def _horizontal_patch(u0, v0, u1, v1, h, step=0.05):
    u, v = np.meshgrid(np.arange(u0, u1, step), np.arange(v0, v1, step))
    pts = np.stack([u.ravel(), v.ravel(), np.full(u.size, float(h))], axis=1)
    n = np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
    return pts, n


def _box(u0, v0, u1, v1, h0, h1):
    """Furniture: 4 vertical sides + horizontal top."""
    parts = [
        _wall_segment(u0, v0, u1, v0, (0, 1), h0, h1),
        _wall_segment(u0, v1, u1, v1, (0, -1), h0, h1),
        _wall_segment(u0, v0, u0, v1, (1, 0), h0, h1),
        _wall_segment(u1, v0, u1, v1, (-1, 0), h0, h1),
        _horizontal_patch(u0, v0, u1, v1, h1),
    ]
    return parts


def make_apartment(floor_z=FLOOR_Z, rng=None):
    """7x4 m two-room flat (4x4 + 3x4) split at u=4 with a 0.9 m door. Returns (pts, nrm)
    in canonical z-up coordinates with the floor at `floor_z`."""
    parts = [
        _wall_segment(0, 0, 7, 0, (0, 1), floor_z, floor_z + WALL_H),
        _wall_segment(0, 4, 7, 4, (0, -1), floor_z, floor_z + WALL_H),
        _wall_segment(0, 0, 0, 4, (1, 0), floor_z, floor_z + WALL_H),
        _wall_segment(7, 0, 7, 4, (-1, 0), floor_z, floor_z + WALL_H),
        _wall_segment(4, 0, 4, 4, (1, 0), floor_z, floor_z + WALL_H, door_v=2.0),
        _horizontal_patch(0, 0, 7, 4, floor_z),
        _horizontal_patch(0, 0, 7, 4, floor_z + WALL_H),
    ]
    parts += _box(1, 1, 2.8, 1.8, floor_z, floor_z + 0.5)  # sofa: short sides, not walls
    parts += _box(6.2, 0.2, 6.8, 0.8, floor_z, floor_z + 2.2)  # wardrobe: tall, wall-like
    if rng is not None:
        n_noise = 800
        pts = np.stack(
            [
                rng.uniform(0, 7, n_noise),
                rng.uniform(0, 4, n_noise),
                rng.uniform(floor_z, floor_z + WALL_H, n_noise),
            ],
            axis=1,
        )
        nrm = rng.normal(size=(n_noise, 3))
        parts.append((pts, nrm))
    points = np.concatenate([p for p, _ in parts])
    normals = np.concatenate([n for _, n in parts])
    return points, normals


def remap_axes(points, normals, vertical_axis):
    u_axis, v_axis = _PLANE_AXES[vertical_axis]
    out_p = np.zeros_like(points)
    out_n = np.zeros_like(normals)
    out_p[:, u_axis], out_p[:, v_axis], out_p[:, vertical_axis] = (
        points[:, 0],
        points[:, 1],
        points[:, 2],
    )
    out_n[:, u_axis], out_n[:, v_axis], out_n[:, vertical_axis] = (
        normals[:, 0],
        normals[:, 1],
        normals[:, 2],
    )
    return out_p, out_n


def room_at(result, u, v):
    for room_id, room in result.rooms.items():
        if room.polygon.contains(Point(u, v)):
            return room_id
    return None


def with_ghosts(points, normals, rng):
    """Append low-opacity ghost splats that must be filtered out."""
    n_ghost = 3000
    ghosts = np.stack(
        [
            rng.uniform(-1, 8, n_ghost),
            rng.uniform(-1, 5, n_ghost),
            rng.uniform(FLOOR_Z - 0.5, FLOOR_Z + 3.0, n_ghost),
        ],
        axis=1,
    )
    ghost_n = rng.normal(size=(n_ghost, 3))
    opacities = np.concatenate([np.full(len(points), 0.95), np.full(n_ghost, 0.05)])
    return np.concatenate([points, ghosts]), np.concatenate([normals, ghost_n]), opacities


def test_two_rooms_zup(output_dir):
    rng = np.random.default_rng(0)
    points, normals = make_apartment(rng=rng)
    points, normals, opacities = with_ghosts(points, normals, rng)

    seg = StructuralRoomSegmenter(vertical_axis=2, output_dir=output_dir)
    result = seg.segment(points, normals, opacities)

    assert len(result.storeys) == 1, f"expected 1 storey, got {len(result.storeys)}"
    floor = result.storeys[0].floor_height
    assert abs(floor - FLOOR_Z) < 0.15, f"floor {floor:.2f} vs true {FLOOR_Z}"
    assert len(result.rooms) == 2, f"expected 2 rooms, got {len(result.rooms)}"

    room_a = room_at(result, 1.0, 3.5)
    room_b = room_at(result, 6.0, 2.0)
    assert room_a is not None and room_b is not None, "test points not inside any room"
    assert room_a != room_b, "door leak: both probe points landed in the same room"

    area_a = result.rooms[room_a].area_m2
    area_b = result.rooms[room_b].area_m2
    assert 11.0 < area_a < 18.0, f"room A area {area_a:.1f} out of range (true ~15.5)"
    assert 7.5 < area_b < 14.0, f"room B area {area_b:.1f} out of range (true ~11.5)"
    print(f"PASS two_rooms_zup  floor={floor:.2f}  areas=({area_a:.1f}, {area_b:.1f}) m2")


def test_two_rooms_yup():
    rng = np.random.default_rng(1)
    points, normals = make_apartment(rng=rng)
    points, normals = remap_axes(points, normals, vertical_axis=1)

    seg = StructuralRoomSegmenter(vertical_axis=1)
    result = seg.segment(points, normals)

    assert len(result.rooms) == 2, f"expected 2 rooms, got {len(result.rooms)}"
    assert room_at(result, 1.0, 3.5) != room_at(result, 6.0, 2.0)
    print("PASS two_rooms_yup")


def test_two_storeys():
    rng = np.random.default_rng(2)
    p1, n1 = make_apartment(floor_z=FLOOR_Z, rng=rng)
    p2, n2 = make_apartment(floor_z=FLOOR_Z + 3.0, rng=rng)  # 0.5 m slab above ceiling
    points = np.concatenate([p1, p2])
    normals = np.concatenate([n1, n2])

    seg = StructuralRoomSegmenter(vertical_axis=2)
    result = seg.segment(points, normals)

    assert len(result.storeys) == 2, f"expected 2 storeys, got {len(result.storeys)}"
    f0, f1 = (s.floor_height for s in result.storeys)
    assert abs(f0 - FLOOR_Z) < 0.15 and abs(f1 - (FLOOR_Z + 3.0)) < 0.15, f"floors {f0}, {f1}"
    per_storey = [len(s.room_ids) for s in result.storeys]
    assert per_storey == [2, 2], f"rooms per storey {per_storey}, expected [2, 2]"
    print(f"PASS two_storeys  floors=({f0:.2f}, {f1:.2f})")


def test_dense_noisy():
    """SLAM-scale check: jitter-duplicated splats plus floaters with random normals
    must neither fabricate walls nor lose the room split."""
    rng = np.random.default_rng(4)
    base_p, base_n = make_apartment(rng=rng)
    reps = int(np.ceil(500_000 / len(base_p)))
    points = np.concatenate([base_p + rng.normal(0, 0.01, base_p.shape) for _ in range(reps)])
    normals = np.concatenate([base_n] * reps)
    opacities = rng.uniform(0.3, 1.0, len(points))

    seg = StructuralRoomSegmenter(vertical_axis=2)
    result = seg.segment(points, normals, opacities)

    assert len(result.rooms) == 2, f"expected 2 rooms, got {len(result.rooms)}"
    areas = sorted(r.area_m2 for r in result.rooms.values())
    assert 7.5 < areas[0] < 14.0 and 11.0 < areas[1] < 18.0, f"areas {areas}"
    print(f"PASS dense_noisy  {len(points):,} splats, areas={[round(a, 1) for a in areas]} m2")


def test_degenerate_inputs():
    seg = StructuralRoomSegmenter(vertical_axis=2)
    empty = seg.segment(np.zeros((0, 3)), np.zeros((0, 3)))
    assert empty.is_empty and empty.room_polygons() == {}

    rng = np.random.default_rng(3)
    tiny = rng.uniform(0, 0.5, (20, 3))
    result = seg.segment(tiny, rng.normal(size=(20, 3)))
    assert isinstance(result.rooms, dict)  # must not crash; rooms may be empty
    print("PASS degenerate_inputs")


def test_legacy_slice_comparison():
    """An absolute height slice [0.5, 2.0] captures nothing when the floor is at -2.75 m,
    while a floor-relative slice captures the walls."""
    points, _ = make_apartment()
    heights = points[:, 2]
    legacy = np.count_nonzero((heights >= 0.5) & (heights <= 2.0))
    relative = np.count_nonzero((heights >= FLOOR_Z + 0.5) & (heights <= FLOOR_Z + 2.0))
    assert legacy == 0, "legacy slice unexpectedly caught points"
    assert relative > 0.3 * len(points)
    print(
        f"PASS legacy_slice_comparison  absolute slice: {legacy} pts, "
        f"floor-relative: {relative} pts ({100 * relative / len(points):.0f}%)"
    )


def test_reusable_api():
    """segment_rooms_from_arrays reproduces direct StructuralRoomSegmenter usage."""
    points, normals = make_apartment()
    direct = StructuralRoomSegmenter(vertical_axis=2).segment(points, normals)
    via_api = segment_rooms_from_arrays(points, normals, vertical_axis=2)
    assert not via_api.is_empty
    assert len(via_api.rooms) == len(direct.rooms)
    assert len(via_api.storeys) == len(direct.storeys)
    print(f"PASS reusable_api  rooms: {len(via_api.rooms)} (matches direct)")


def _splats_from_surfels(points, normals, rng):
    """Turn surface points into synthetic splats: thin axis along the normal."""
    from scipy.spatial.transform import Rotation

    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    helper = np.where(np.abs(n[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    t1 = np.cross(n, helper)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-8)
    t2 = np.cross(n, t1)
    R = np.stack([n, t1, t2], axis=2)
    quats = Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
    scales = np.tile([0.01, 0.05, 0.05], (len(points), 1))
    opacities = rng.uniform(0.85, 0.99, len(points))
    return quats.astype(np.float32), scales.astype(np.float32), opacities.astype(np.float32)


def _assert_two_rooms(result, method):
    assert len(result.rooms) >= 2, f"[{method}] expected >=2 rooms, got {len(result.rooms)}"
    room_a = room_at(result, 1.0, 3.5)
    room_b = room_at(result, 6.0, 2.0)
    assert room_a is not None and room_b is not None, f"[{method}] probe points unassigned"
    assert room_a != room_b, f"[{method}] door leak: probes landed in the same room"
    for rid in (room_a, room_b):
        assert result.rooms[rid].area_m2 > 5.0, f"[{method}] probe room too small"


def test_point_methods_two_rooms():
    """The reimplemented point-based baselines split the synthetic flat at the door."""
    rng = np.random.default_rng(5)
    points, normals = make_apartment(rng=rng)
    for method in ("hydra", "hovsg"):
        result = segment_rooms_from_arrays(points, normals, method=method, vertical_axis=2)
        _assert_two_rooms(result, method)
        print(f"PASS point_methods_two_rooms[{method}]  rooms: {len(result.rooms)}")


def test_ours_v2_two_rooms():
    """ours_v2 on synthetic splats: walls block, floors/sofa stay transparent."""
    rng = np.random.default_rng(6)
    points, normals = make_apartment(rng=None)
    quats, scales, opacities = _splats_from_surfels(points, normals, rng)
    result = segment_rooms_from_arrays(
        points,
        normals,
        opacities,
        scales=scales,
        rotations=quats,
        method="ours_v2",
        vertical_axis=2,
    )
    assert len(result.storeys) == 1, f"expected 1 storey, got {len(result.storeys)}"
    assert len(result.rooms) == 2, f"expected 2 rooms, got {len(result.rooms)}"
    _assert_two_rooms(result, "ours_v2")
    areas = sorted(r.area_m2 for r in result.rooms.values())
    assert 7.5 < areas[0] < 14.0 and 11.0 < areas[1] < 18.0, f"areas {areas}"
    print(f"PASS ours_v2_two_rooms  areas={[round(a, 1) for a in areas]} m2")


def test_floorplan_output(output_dir):
    rng = np.random.default_rng(7)
    points, normals = make_apartment(rng=rng)
    result = segment_rooms_from_arrays(points, normals, vertical_axis=2)
    path = write_floorplan(result, output_dir, method="ours", vertical_axis=2)
    with open(path) as f:
        data = json.load(f)
    assert data["format"] == "gssg-floorplan-v1"
    assert len(data["rooms"]) == len(result.rooms)
    assert all(len(r["polygon"]) >= 1 for r in data["rooms"])
    assert os.path.exists(os.path.join(output_dir, "floorplan.png"))
    print(f"PASS floorplan_output  {len(data['rooms'])} rooms -> {path}")


if __name__ == "__main__":
    out = os.path.join(tempfile.gettempdir(), "room_segmentation_test")
    os.makedirs(out, exist_ok=True)
    test_two_rooms_zup(out)
    test_two_rooms_yup()
    test_two_storeys()
    test_dense_noisy()
    test_degenerate_inputs()
    test_legacy_slice_comparison()
    test_reusable_api()
    test_point_methods_two_rooms()
    test_ours_v2_two_rooms()
    test_floorplan_output(out)
    print(f"\nAll room_segmentation tests passed. Debug images: {out}")
    sys.exit(0)
