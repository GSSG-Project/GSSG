"""SplatCloud: the common input to every room-segmentation method.

Carries the full Gaussian parameterization (not just points), so methods can be
point-based (hydra, hovsg), normal-based (ours) or splat-native (ours_v2).
`load_splat_cloud` reads a saved 3DGS PLY with plain numpy — no CUDA required —
applying the standard activations (sigmoid opacity, exp scales, normalized wxyz
quaternions) and recomputing normals as the ellipsoid axis of smallest scale,
exactly as the live map's `GaussianPointCloud.get_normal` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plyfile import PlyData


@dataclass
class SplatCloud:
    xyz: np.ndarray  # (N,3)
    normals: np.ndarray | None = None  # (N,3) unit, sign-ambiguous
    opacities: np.ndarray | None = None  # (N,) in [0,1]
    scales: np.ndarray | None = None  # (N,3) world units
    rotations: np.ndarray | None = None  # (N,4) unit quaternions, wxyz

    def __len__(self):
        return len(self.xyz)


def rotation_matrices(quats):
    """(N,4) wxyz quaternions -> (N,3,3) rotation matrices (normalizes first)."""
    q = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float32)
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


def normals_from_shape(scales, rotations):
    """Normal = ellipsoid axis with the smallest scale (the splat's thin direction)."""
    R = rotation_matrices(rotations)
    idx = scales.argmin(axis=1)
    n = R[np.arange(len(R)), :, idx]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def vertical_marginals(scales, rotations, u_axis, v_axis, vertical_axis):
    """Per-splat quantities for vertical line integrals of the 3D Gaussian density.

    Returns ((a, b, c), sigma_h, sigma_cond) where (a, b, c) is the in-plane 2x2 marginal
    covariance [[a, b], [b, c]] (the (u,v) sub-block of Sigma = R diag(s^2) R^T), sigma_h
    the marginal vertical std, and sigma_cond the conditional vertical std — the effective
    thickness the line integral sees: integral along h = sqrt(2*pi)*sigma_cond * G_2d(u,v).
    """
    R = rotation_matrices(rotations)
    cov = np.einsum("nij,nkj->nik", R * (scales[:, None, :] ** 2), R)
    a = cov[:, u_axis, u_axis]
    b = cov[:, u_axis, v_axis]
    c = cov[:, v_axis, v_axis]
    d = cov[:, u_axis, vertical_axis]
    e = cov[:, v_axis, vertical_axis]
    f = cov[:, vertical_axis, vertical_axis]
    det = np.maximum(a * c - b * b, 1e-12)
    sigma_cond_sq = np.maximum(f - (c * d * d - 2 * b * d * e + a * e * e) / det, 1e-12)
    return (a, b, c), np.sqrt(np.maximum(f, 1e-12)), np.sqrt(sigma_cond_sq)


def band_fractions(mu_h, sigma_h, lo, hi):
    """Fraction of each splat's vertical mass inside the height band [lo, hi]."""
    from scipy.special import erf

    s = np.maximum(sigma_h, 1e-6) * np.sqrt(2.0)
    return 0.5 * (erf((hi - mu_h) / s) - erf((lo - mu_h) / s))


def load_splat_cloud(ply_path) -> SplatCloud:
    v = PlyData.read(ply_path).elements[0]
    names = {p.name for p in v.properties}
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    opacities = scales = rotations = normals = None
    if "opacity" in names:
        opacities = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    if {"scale_0", "scale_1", "scale_2"} <= names:
        scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)).astype(
            np.float32
        )
    if {"rot_0", "rot_1", "rot_2", "rot_3"} <= names:
        rotations = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
    if scales is not None and rotations is not None:
        normals = normals_from_shape(scales, rotations)

    return SplatCloud(
        xyz=xyz, normals=normals, opacities=opacities, scales=scales, rotations=rotations
    )
