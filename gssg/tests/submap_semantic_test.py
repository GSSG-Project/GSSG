"""Tests that an object's whole-map (size, center, AABB) recomposed from per-(cell, id) reductions equals a gather over the whole cloud, including for objects spanning several cells and across fusion relabels that happen while a cell is evicted."""

import sys

import numpy as np

from gssg.map.submap_semantic import SemanticReductionIndex

CELL = 4.0


def cell_of(xyz):
    u = np.floor(xyz[:, 0] / CELL).astype(np.int64)
    v = np.floor(xyz[:, 1] / CELL).astype(np.int64)
    return (u << 21) | (v & ((1 << 21) - 1))


def gather(sem, xyz, sid):
    m = sem == sid
    if not m.any():
        return None
    p = xyz[m].astype(np.float64)
    return {"size": int(m.sum()), "center": p.mean(axis=0), "aabb": (p.min(axis=0), p.max(axis=0))}


def assert_agg_eq(a, b):
    assert a is not None and b is not None
    assert a["size"] == b["size"], (a["size"], b["size"])
    assert np.allclose(a["center"], b["center"], atol=1e-9)
    assert np.allclose(a["aabb"][0], b["aabb"][0]) and np.allclose(a["aabb"][1], b["aabb"][1])


def _scene(seed=0, n=4000, ids=8):
    rng = np.random.default_rng(seed)
    # Spread points over many cells so most ids span >1 cell (tests decomposition).
    xyz = rng.uniform(-12, 12, size=(n, 3))
    sem = rng.integers(1, ids + 1, size=n).astype(np.int64)
    return sem, xyz


def test_exact_decomposition_multicell():
    sem, xyz = _scene(0)
    idx = SemanticReductionIndex()
    idx.update_cells(cell_of(xyz), sem, xyz)
    spanned = 0
    for sid in np.unique(sem):
        assert_agg_eq(idx.aggregate(sid), gather(sem, xyz, sid))
        if len(idx.cells_of(sid)) > 1:
            spanned += 1
    assert spanned >= 1, "test scene should have multi-cell objects"
    print(f"PASS exact_decomposition_multicell  ({spanned} multi-cell objects)")


def test_relabel_matches_gather():
    sem, xyz = _scene(1)
    idx = SemanticReductionIndex()
    idx.update_cells(cell_of(xyz), sem, xyz)
    idx.relabel({5: 3})
    clone = np.where(sem == 5, 3, sem)
    assert_agg_eq(idx.aggregate(3), gather(clone, xyz, 3))
    assert idx.aggregate(5) is None
    print("PASS relabel_matches_gather")


def test_journaled_relabel_survives_evict_restore():
    sem, xyz = _scene(2)
    cells = cell_of(xyz)
    idx = SemanticReductionIndex()
    idx.update_cells(cells, sem, xyz)

    # Evict one cell that holds id 5, then merge 5->3 and 3->2 while it is away.
    victim = int(idx.cells_of(5).pop())
    idx.evict_cell(victim)
    idx.relabel({5: 3})
    idx.relabel({3: 2})

    # Aggregates are correct even while evicted (keys remapped live).
    clone = sem.copy()
    clone[clone == 5] = 3
    clone[clone == 3] = 2
    assert_agg_eq(idx.aggregate(2), gather(clone, xyz, 2))

    # The journal carries both epochs for the caller to replay on the frozen blob.
    journal = idx.pop_journal(victim)
    assert journal == [{5: 3}, {3: 2}], journal

    idx.restore_cell(victim)
    assert idx.pop_journal(victim) == []
    assert_agg_eq(idx.aggregate(2), gather(clone, xyz, 2))
    print("PASS journaled_relabel_survives_evict_restore")


def test_resident_removal_recompute():
    """Removing rows from a resident cell (recompute-on-dirty) tracks the gather exactly;
    min/max are not invertible, so this is the path that must rescan."""
    sem, xyz = _scene(3)
    cells = cell_of(xyz)
    idx = SemanticReductionIndex()
    idx.update_cells(cells, sem, xyz)

    keep = np.ones(len(sem), bool)
    keep[::7] = False  # drop ~1/7 of rows
    sem2, xyz2, cells2 = sem[keep], xyz[keep], cells[keep]
    # Recompute only the touched cells from their new full membership.
    touched = np.unique(cells[~keep])
    in_touched = np.isin(cells2, touched)
    idx.update_cells(cells2[in_touched], sem2[in_touched], xyz2[in_touched])

    for sid in np.unique(sem2):
        assert_agg_eq(idx.aggregate(sid), gather(sem2, xyz2, sid))
    print("PASS resident_removal_recompute")


def test_verify_equivalence_self_check():
    """verify_equivalence matches a gather and flags injected drift."""
    sem, xyz = _scene(4)
    idx = SemanticReductionIndex()
    idx.update_cells(cell_of(xyz), sem, xyz)

    ok, bad = idx.verify_equivalence(sem, xyz)
    assert ok and not bad, bad

    # Inject drift into one cell's reduction; verify_equivalence must flag exactly that id.
    any_cell = next(iter(idx.cells))
    any_id = next(iter(idx.cells[any_cell]))
    idx.cells[any_cell][any_id][0] += 100  # inflate the count
    ok2, bad2 = idx.verify_equivalence(sem, xyz)
    assert (not ok2) and any_id in bad2, (ok2, bad2)
    print("PASS verify_equivalence_self_check (matches gather; catches injected drift)")


if __name__ == "__main__":
    test_exact_decomposition_multicell()
    test_relabel_matches_gather()
    test_journaled_relabel_survives_evict_restore()
    test_resident_removal_recompute()
    test_verify_equivalence_self_check()
    print("\nAll submap_semantic tests passed.")
    sys.exit(0)
