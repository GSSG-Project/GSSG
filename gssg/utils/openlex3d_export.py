import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Gaussian-splat -> dense surface point cloud (mathematically grounded).
#
# Each stable gaussian is a 3D normal N(mu, Sigma) with
#     Sigma = R S S^T R^T ,   S = diag(s0, s1, s2) (per-axis std-devs),
#     R = rotation from the unit quaternion.
# Let L = R S  (so Sigma = L L^T). A point uniformly distributed inside the
# k-sigma confidence ellipsoid  E_k = { x : (x-mu)^T Sigma^-1 (x-mu) <= k^2 }
# is obtained by
#     x = mu + k * L * b ,     b ~ Uniform(unit ball B^3),
# because the affine map b -> k L b sends B^3 onto E_k with constant Jacobian
# (|k L| = k^3 s0 s1 s2), so uniform stays uniform. For a flat surfel (one tiny
# scale = thickness) this is a near-flat disk of points lying on the surface the
# splat models -- exactly the coverage the OpenLex3D metric needs (a GT point is
# "missing" if no predicted point lies within 5 cm of it).
#
# We draw a number of samples per splat proportional to its footprint AREA
# (pi * (k s_a)(k s_b), the two largest scales = tangent extents) so the output
# has a roughly uniform surface density of ~1 point per `spacing` metres.
# ---------------------------------------------------------------------------


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _quat_wxyz_to_R(q):
    """(N,4) unit quaternions (w,x,y,z) -> (N,3,3) rotation matrices.

    Matches gssg.utils.general_utils.build_rotation.
    """
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _sample_unit_ball(n, rng):
    """n points drawn uniformly from the unit ball B^3."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    r = rng.random(n) ** (1.0 / 3.0)  # radial CDF for uniform-in-ball
    return d * r[:, None]


def _densify_gaussians(
    xyz, log_scale, quat, rows, spacing, k_sigma, k_min, k_max, seed=0
):
    """Sample a dense, surface-conforming point cloud from the object gaussians.

    Returns (points (M,3) float32, index (M,) int32 -> embedding row).
    Falls back to plain centers if scale/rotation are unusable.
    """
    s = np.exp(log_scale.astype(np.float64))  # std-devs (linear), always > 0
    s = np.clip(s, 1e-4, None)
    finite = (
        np.isfinite(xyz).all(1)
        & np.isfinite(s).all(1)
        & np.isfinite(quat).all(1)
        & (np.linalg.norm(quat, axis=1) > 1e-6)
    )
    if not finite.any():
        return xyz.astype(np.float32), rows.astype(np.int32)
    xyz, s, quat, rows = xyz[finite], s[finite], quat[finite], rows[finite]

    R = _quat_wxyz_to_R(quat)               # (M0,3,3)
    L = R * s[:, None, :]                    # L = R @ diag(s):  L[:,i,j] = R[:,i,j]*s_j

    # samples per gaussian proportional to footprint area (two largest scales)
    s_sorted = np.sort(s, axis=1)[:, ::-1]
    area = np.pi * (k_sigma ** 2) * s_sorted[:, 0] * s_sorted[:, 1]
    K = np.clip(np.rint(area / (spacing ** 2)), k_min, k_max).astype(np.int64)

    # Hard safety cap on total points (huge multi-room maps must not blow up RAM).
    TOTAL_CAP = 6_000_000
    total = int(K.sum())
    if total > TOTAL_CAP:
        K = np.maximum(1, np.floor(K * (TOTAL_CAP / total))).astype(np.int64)

    parent = np.repeat(np.arange(len(xyz)), K)      # (M,)
    rng = np.random.default_rng(seed)
    b = _sample_unit_ball(len(parent), rng)         # (M,3)
    # x = mu + k * L @ b
    pts = xyz[parent] + k_sigma * np.einsum("mij,mj->mi", L[parent], b)
    return pts.astype(np.float32), rows[parent].astype(np.int32)


def export_for_openlex3d(
    point_cloud,
    scene_graph,
    output_dir="output",
    densify=True,
    densify_spacing=0.025,   # ~2.5cm: the eval voxel-downsamples to 5cm, so denser is wasted
    densify_k_sigma=2.0,
    densify_min=2,
    densify_max=8,           # cap points/gaussian -> bounded cloud size on huge multi-room HM3D maps
):
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(point_cloud, "_xyz"):
        xyz = point_cloud._xyz
    else:
        xyz = getattr(point_cloud, "xyz", None)
        if xyz is None:
            raise AttributeError("point_cloud object missing '_xyz' or 'xyz'")
    xyz = _to_numpy(xyz)

    if xyz.shape[0] == 0:
        print("ERROR: Point cloud _xyz is empty (0 points). Check your input data.")
        return

    if hasattr(point_cloud, "_semantic"):
        semantic_ids = point_cloud._semantic
    else:
        semantic_ids = getattr(point_cloud, "semantic", None)
        if semantic_ids is None:
            raise AttributeError("point_cloud object missing '_semantic'")
    # int cast: a float dtype here mismatches the index files and zeroes them out
    semantic_ids = _to_numpy(semantic_ids).astype(np.int32).flatten()

    # gaussian shape params (for densification); optional -> graceful fallback
    log_scale = getattr(point_cloud, "_scaling", None)
    quat = getattr(point_cloud, "_rotation", None)
    have_shape = log_scale is not None and quat is not None
    if have_shape:
        log_scale = _to_numpy(log_scale)
        quat = _to_numpy(quat)
        have_shape = log_scale.shape[0] == xyz.shape[0] == quat.shape[0]

    print(f"DEBUG: Processing Point Cloud with {xyz.shape[0]} points.")

    raw_objects = scene_graph.all_objects
    object_list = []
    if isinstance(raw_objects, dict):
        for obj_id, obj in raw_objects.items():
            object_list.append((obj_id, obj))
    else:
        for obj in raw_objects.values():
            if hasattr(obj, "id"):
                object_list.append((obj.id, obj))
    object_list.sort(key=lambda x: x[0])

    embeddings = []
    id_to_index_map = {}
    for obj_id, _obj in object_list:
        try:
            emb_vector, _ = scene_graph.get_object_embedding_and_confidence(obj_id)
        except Exception:
            continue
        if emb_vector is None:
            continue
        if isinstance(emb_vector, torch.Tensor):
            emb_vector = emb_vector.detach().cpu().numpy()
        # OpenLex3D scores by cosine vs text features, so each object embedding must be
        # L2-normalized (the stored db vector is the raw input to insert, which is not
        # guaranteed unit-norm).
        v = emb_vector.flatten().astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        embeddings.append(v)
        id_to_index_map[int(obj_id)] = len(embeddings) - 1

    if not embeddings:
        raise ValueError("Scene Graph processing resulted in 0 valid embeddings.")
    embeddings_np = np.vstack(embeddings).astype(np.float32)
    print(f"DEBUG: Generated Embeddings with shape {embeddings_np.shape}.")

    # Per-gaussian embedding-row, -1 for background / embedding-less points.
    # (These previously defaulted to row 0 = the first object's vector, which the
    # eval's nearest-neighbour search then mis-attributed -> inflated Incorrect.)
    row_full = np.full(semantic_ids.shape, -1, dtype=np.int32)
    for obj_id, row_index in id_to_index_map.items():
        row_full[semantic_ids == obj_id] = row_index
    keep = row_full >= 0
    n_total = int(keep.shape[0])
    n_obj = int(keep.sum())

    if n_obj == 0:
        raise ValueError("No gaussian carries a real object embedding.")

    xyz_o = xyz[keep]
    rows_o = row_full[keep]

    if densify and have_shape:
        try:
            out_xyz, out_index = _densify_gaussians(
                xyz_o,
                log_scale[keep],
                quat[keep],
                rows_o,
                spacing=float(densify_spacing),
                k_sigma=float(densify_k_sigma),
                k_min=int(densify_min),
                k_max=int(densify_max),
            )
            print(
                f"DEBUG: densified {n_obj} object gaussians -> {len(out_xyz)} surface "
                f"points (x{len(out_xyz) / max(n_obj, 1):.1f}); dropped "
                f"{n_total - n_obj} background ({100.0 * (n_total - n_obj) / max(n_total, 1):.1f}%)."
            )
        except Exception as e:  # never let densification break the export
            print(f"WARN: densification failed ({e}); exporting gaussian centers.")
            out_xyz, out_index = xyz_o.astype(np.float32), rows_o.astype(np.int32)
    else:
        out_xyz, out_index = xyz_o.astype(np.float32), rows_o.astype(np.int32)
        print(
            f"DEBUG: exporting {n_obj} object-gaussian centers (densify={densify}, "
            f"have_shape={have_shape}); dropped {n_total - n_obj} background."
        )

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings_np)
    np.save(os.path.join(output_dir, "index.npy"), out_index)
    save_ply(os.path.join(output_dir, "input.ply"), out_xyz)
    print("Export successfully finished.")


def save_ply(filepath, points):
    num_points = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.astype(np.float32).tobytes())
