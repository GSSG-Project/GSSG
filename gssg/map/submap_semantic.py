"""Per-(cell, semantic_id) reductions over the stable cloud, the foundation for
correct submap eviction.

Three readers walk the WHOLE stable cloud per frame (object geometry size/center/AABB,
fusion voxel-overlap, clean-empty), so an evicted cell would be undercounted. To let
eviction engage, those readers answer from this index instead of the resident rows. It
keeps, per (cell, id), the exactly decomposable
reductions, so an object's whole-map aggregate recomposes from its cells (resident or
evicted) bit-identically to a gather over the full cloud:

    size   = sum count
    center = sum xyz_sum / sum count
    aabb   = (elementwise min of mins, elementwise max of maxs)

min/max are not invertible under row removal, so a resident cell is recomputed on demand
from the live rows (`update_cells`); an evicted cell is frozen (its rows can't change) and
its cached reduction stays valid. A fusion merge remaps ids via `relabel`: reduction keys
are remapped immediately everywhere, and for evicted cells the remap is also journaled so
the caller can replay the identical bucketize on the frozen tensor blob at page-in.

Opt-in via `submap_evict`: the mapper refreshes resident cells each frame, mirrors fusion
relabels here, and answers object geometry from `aggregate` (folded with the resident
active cloud).
"""

from __future__ import annotations

import numpy as np


def _reduce(sem, xyz):
    """{sem_id -> [count, xyz_sum(3), xyz_min(3), xyz_max(3)]} for one cell's rows."""
    red = {}
    for sid in np.unique(sem):
        m = sem == sid
        p = xyz[m]
        red[int(sid)] = [int(m.sum()), p.sum(axis=0), p.min(axis=0), p.max(axis=0)]
    return red


def _merge_into(dst, src):
    """Compose two per-id reduction entries (used when a relabel collides two ids)."""
    dst[0] += src[0]
    dst[1] = dst[1] + src[1]
    dst[2] = np.minimum(dst[2], src[2])
    dst[3] = np.maximum(dst[3], src[3])
    return dst


VOXEL_SIZE = 0.075  # fusion-overlap quantization edge (matches semantic_fusion voxel_size)
_VBITS = 21  # bits/axis for exact voxel-key packing (+-78km at 0.075 m)
_VOFF = 1 << 20


def _voxelize(xyz):
    """Unique quantized voxel keys (sorted int64) for a point set, using exact bit-packing
    so stored per-(cell,id) sets compose by union across cells."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    q = np.floor(xyz / VOXEL_SIZE).astype(np.int64) + _VOFF
    return np.unique((q[:, 0] << (2 * _VBITS)) | (q[:, 1] << _VBITS) | q[:, 2])


def _voxelize_by_id(sem, xyz):
    """{sem_id -> unique voxel-key array} for one cell's rows."""
    return {int(sid): _voxelize(xyz[sem == sid]) for sid in np.unique(sem)}


class SemanticReductionIndex:
    def __init__(self):
        self.cells: dict[int, dict[int, list]] = {}  # cell_id -> {sem_id -> reduction}
        self.voxels: dict[int, dict[int, np.ndarray]] = {}  # cell_id -> {sem_id -> voxel keys}
        self.evicted: set[int] = set()
        self.journal: dict[int, list[dict]] = {}  # cell_id -> [id_map, ...] to replay on blob

    # -------------------------------------------------------------- maintenance
    def update_cells(self, cell_ids, sem_ids, xyz):
        """Recompute reductions for every resident cell present in this batch from its full
        current membership. Call with the touched cells' complete rows after any add/remove;
        evicted (frozen) cells are skipped."""
        cell_ids = np.asarray(cell_ids).astype(np.int64)
        sem_ids = np.asarray(sem_ids).astype(np.int64)
        xyz = np.asarray(xyz, dtype=np.float64)
        for c in np.unique(cell_ids):
            c = int(c)
            if c in self.evicted:
                continue
            m = cell_ids == c
            self.cells[c] = _reduce(sem_ids[m], xyz[m])
            self.voxels[c] = _voxelize_by_id(sem_ids[m], xyz[m])

    def evict_cell(self, cell_id):
        """Freeze a cell: its cached reduction is retained and no longer recomputed."""
        self.evicted.add(int(cell_id))

    def restore_cell(self, cell_id):
        """Unfreeze a cell. Its reduction keys are already current (relabel remapped them
        live); the tensor-side journal has been replayed by the caller, so drop it."""
        cell_id = int(cell_id)
        self.evicted.discard(cell_id)
        self.journal.pop(cell_id, None)

    def prune_resident_cells(self, present):
        """Drop index entries for non-evicted cells that no longer hold any resident rows
        (e.g. a cell emptied when optimization moved its points across a boundary).
        update_cells only overwrites cells present in its batch, so without this an emptied
        cell keeps a stale reduction and double-counts. Evicted (frozen) cells kept."""
        present = {int(c) for c in np.asarray(present).reshape(-1)}
        for c in [c for c in self.cells if c not in present and c not in self.evicted]:
            self.cells.pop(c, None)
            self.voxels.pop(c, None)

    def pop_journal(self, cell_id):
        """The list of id-remaps applied while a cell was evicted, for the caller to replay
        (composed) on the frozen tensor blob before paging it back in. Empty if none."""
        return self.journal.get(int(cell_id), [])

    # ------------------------------------------------------------------ relabel
    def relabel(self, id_map):
        """Apply a fusion id remap {old: new}. Reduction keys are remapped immediately in
        every cell (aggregates stay correct); for evicted cells the remap is journaled so the
        caller can replay the same bucketize on the frozen blob at page-in."""
        id_map = {int(o): int(n) for o, n in id_map.items() if int(o) != int(n)}
        if not id_map:
            return
        for c, red in self.cells.items():
            self._remap_cell(red, id_map)
            vox = self.voxels.get(c)
            if vox is not None:
                self._remap_voxels(vox, id_map)
            if c in self.evicted:
                self.journal.setdefault(c, []).append(dict(id_map))

    @staticmethod
    def _remap_cell(red, id_map):
        for old, new in id_map.items():
            if old not in red:
                continue
            moved = red.pop(old)
            if new in red:
                _merge_into(red[new], moved)
            else:
                red[new] = moved

    @staticmethod
    def _remap_voxels(vox, id_map):
        for old, new in id_map.items():
            if old not in vox:
                continue
            moved = vox.pop(old)
            vox[new] = np.union1d(vox[new], moved) if new in vox else moved

    # ---------------------------------------------------------------- aggregate
    def aggregate(self, sem_id):
        """Whole-map (size, center, aabb) for an object, composed across all its cells —
        resident and evicted. None if the id has no points anywhere."""
        sem_id = int(sem_id)
        count = 0
        xyz_sum = np.zeros(3)
        mn = mx = None
        for red in self.cells.values():
            r = red.get(sem_id)
            if r is None:
                continue
            count += r[0]
            xyz_sum = xyz_sum + r[1]
            mn = r[2].copy() if mn is None else np.minimum(mn, r[2])
            mx = r[3].copy() if mx is None else np.maximum(mx, r[3])
        if count == 0:
            return None
        return {"size": count, "center": xyz_sum / count, "aabb": (mn, mx)}

    def cells_of(self, sem_id):
        """Set of cell ids currently holding an object — the inverted index (id -> cells)."""
        sem_id = int(sem_id)
        return {c for c, red in self.cells.items() if sem_id in red}

    def reduction(self, sem_id):
        """Raw composed reduction (count, xyz_sum[3], min[3], max[3]) across all cells —
        resident and evicted — or None. Lets a caller fold in extra rows (e.g. the always-
        resident active cloud) exactly before deriving center/size/aabb."""
        sem_id = int(sem_id)
        count = 0
        xyz_sum = np.zeros(3)
        mn = mx = None
        for red in self.cells.values():
            r = red.get(sem_id)
            if r is None:
                continue
            count += r[0]
            xyz_sum = xyz_sum + r[1]
            mn = r[2].copy() if mn is None else np.minimum(mn, r[2])
            mx = r[3].copy() if mx is None else np.maximum(mx, r[3])
        if count == 0:
            return None
        return (count, xyz_sum, mn, mx)

    def voxel_keys(self, sem_id):
        """Whole-object unique voxel keys (sorted int64) across all cells resident+evicted —
        for eviction-correct fusion overlap (an object spanning an evicted cell keeps its
        voxels here even though its rows left the GPU)."""
        sem_id = int(sem_id)
        parts = [v[sem_id] for v in self.voxels.values() if sem_id in v]
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(parts))

    def verify_equivalence(self, sem_ids_full, xyz_full, atol=1e-3):
        """Confirm every object's index aggregate matches a fresh whole-cloud gather
        (resident rows plus every evicted blob's rows). Returns (ok, [mismatched_ids])."""
        sem_ids_full = np.asarray(sem_ids_full).astype(np.int64)
        xyz_full = np.asarray(xyz_full, dtype=np.float64)
        bad = []
        for sid in np.unique(sem_ids_full):
            sid = int(sid)
            m = sem_ids_full == sid
            p = xyz_full[m]
            agg = self.aggregate(sid)
            if (
                agg is None
                or agg["size"] != int(m.sum())
                or not np.allclose(agg["center"], p.mean(axis=0), atol=atol)
                or not np.allclose(agg["aabb"][0], p.min(axis=0), atol=atol)
                or not np.allclose(agg["aabb"][1], p.max(axis=0), atol=atol)
            ):
                bad.append(sid)
        return (len(bad) == 0, bad)
