"""Structural classification of splats from their surface normals.

A splat whose normal is near-vertical lies on a horizontal surface (floor/ceiling/table);
one whose normal is near-horizontal lies on a vertical surface (wall, cabinet side).
Everything else is clutter.
"""

from dataclasses import dataclass

import numpy as np

from .constants import FLOOR_MAX_TILT_DEG, MIN_OPACITY, WALL_MAX_TILT_DEG

CLUTTER = 0
HORIZONTAL = 1
WALL = 2

# vertical_axis -> the two in-plane axes (u, v)
_PLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


@dataclass
class ClassifiedSplats:
    u: np.ndarray
    v: np.ndarray
    h: np.ndarray
    label: np.ndarray  # CLUTTER / HORIZONTAL / WALL per splat

    def __len__(self):
        return len(self.h)

    def subset(self, mask):
        return ClassifiedSplats(self.u[mask], self.v[mask], self.h[mask], self.label[mask])


def as_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float32)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


def classify_splats(points, normals, opacities=None, vertical_axis=2):
    points = as_numpy(points).reshape(-1, 3)
    normals = as_numpy(normals).reshape(-1, 3)
    if opacities is not None:
        keep = as_numpy(opacities).reshape(-1) > MIN_OPACITY
        points, normals = points[keep], normals[keep]

    u_axis, v_axis = _PLANE_AXES[vertical_axis]
    vertical = np.abs(normals[:, vertical_axis])
    vertical = vertical / np.maximum(np.linalg.norm(normals, axis=1), 1e-8)

    label = np.full(len(points), CLUTTER, dtype=np.uint8)
    label[vertical >= np.cos(np.radians(FLOOR_MAX_TILT_DEG))] = HORIZONTAL
    label[vertical <= np.sin(np.radians(WALL_MAX_TILT_DEG))] = WALL

    return ClassifiedSplats(
        u=points[:, u_axis],
        v=points[:, v_axis],
        h=points[:, vertical_axis],
        label=label,
    )
