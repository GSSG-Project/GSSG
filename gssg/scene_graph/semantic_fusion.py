import logging

import numpy as np
import torch
import torch.nn.functional as F

from gssg.map.submap_semantic import _voxelize
from gssg.utils.performance_utils import timing_decorator


@timing_decorator
def incremental_object_fusion(mapper, global_run=False):
    """Batched, GPU-accelerated fusion of new objects into existing map objects."""
    args = getattr(mapper, "args", None)
    spatial_method = "overlap"
    voxel_size = 0.075
    # Config-overridable fusion gates. semantic_threshold is intentionally loose
    # (random CLIP image-image cosine is ~0.3-0.5) — raise it (e.g. 0.8) to stop distinct
    # adjacent objects merging on weak embedding overlap and smearing their features.
    spatial_threshold = float(getattr(args, "fusion_spatial_threshold", 0.1))
    semantic_threshold = float(getattr(args, "fusion_semantic_threshold", 0.5))
    combined_threshold = float(
        getattr(args, "fusion_combined_threshold_global", 1.3)
        if global_run
        else getattr(args, "fusion_combined_threshold", 1.15)
    )

    sg = mapper.scene_graph
    device = mapper.active_gaussians._xyz.device

    # Identify candidates. Snapshot to a list — sg.all_objects is mutated by merges below.
    new_ids = list(sg.new_detection_ids) if not global_run else list(sg.all_objects.keys())
    if not new_ids:
        return

    map_ids = set(sg.all_objects.keys()) - set(new_ids)
    candidate_pairs = []  # (new_id, old_id)

    for nid in new_ids:
        nearby = sg.rtree_find_nearby_objects(nid, threshold=voxel_size)
        for item in nearby:
            oid = item.id if hasattr(item, "id") else item
            if oid != nid and (nid, oid) not in candidate_pairs and (global_run or oid in map_ids):
                candidate_pairs.append((nid, oid))

    if not candidate_pairs:
        return

    # Batch semantic similarity check.
    valid_candidates = []

    for nid, oid in candidate_pairs:
        try:
            feat_n, _ = sg.get_object_embedding_and_confidence(nid, device)
            feat_o, _ = sg.get_object_embedding_and_confidence(oid, device)
            sem_sim = F.cosine_similarity(feat_n.flatten(), feat_o.flatten(), dim=0).item()
            sem_sim = max(0.0, sem_sim)
            if sem_sim >= semantic_threshold:
                valid_candidates.append({"nid": nid, "oid": oid, "sem": sem_sim})
        except Exception:
            continue

    if not valid_candidates:
        return

    # Pre-group points by semantic id once, avoiding an O(N_gaussians) scan per candidate
    # pair. Gather only the needed objects' points from each cloud before concatenating, so
    # the whole map's xyz is never materialized.
    needed_ids = set(c["nid"] for c in valid_candidates) | set(c["oid"] for c in valid_candidates)
    pcds = [p for p in [mapper.active_gaussians, mapper.stable_gaussians] if p.get_points_num > 0]
    needed_t = torch.as_tensor(list(needed_ids), device=pcds[0]._semantic.device)
    xyzs, sems = [], []
    for p in pcds:
        sem = p._semantic.squeeze(-1)
        m = torch.isin(sem, needed_t)
        if m.any():
            xyzs.append(p._xyz[m])
            sems.append(sem[m])
    if xyzs:
        cat_xyz, cat_sem = torch.cat(xyzs), torch.cat(sems)
    else:
        cat_xyz, cat_sem = pcds[0]._xyz[:0], pcds[0]._semantic.squeeze(-1)[:0]
    pts_by_id = {sid: cat_xyz[cat_sem == sid] for sid in needed_ids}

    # Eviction-correct overlap: an object spanning an evicted cell is missing rows from the
    # resident clouds, so fold the index's whole-object voxel keys (resident + evicted) with
    # the resident points' voxels. The union is idempotent, so resident-stable double-coverage
    # is harmless. sem_index is None when eviction/semantics is off (resident-only path).
    si = getattr(mapper, "sem_index", None)
    vox_by_id = None
    if si is not None:
        vox_by_id = {}
        for sid in needed_ids:
            res = pts_by_id[sid]
            res_vox = (
                _voxelize(res.detach().cpu().numpy())
                if res.shape[0] > 0
                else np.empty(0, dtype=np.int64)
            )
            vox_by_id[sid] = np.union1d(res_vox, si.voxel_keys(sid))

    # Spatial verification.
    final_merges = {}  # new_id -> (score, old_id)

    for cand in valid_candidates:
        nid, oid = cand["nid"], cand["oid"]

        obj_n = sg.get_object_by_id(nid)
        obj_o = sg.get_object_by_id(oid)

        if obj_n is None or obj_o is None:
            continue

        pts_n = pts_by_id.get(nid)
        pts_o = pts_by_id.get(oid)

        if pts_n is None or pts_o is None:
            continue

        if vox_by_id is not None:
            spat_sim = overlap_from_voxel_sets(vox_by_id[nid], vox_by_id[oid])
        else:
            spat_sim = compute_spatial_similarity(pts_n, pts_o, voxel_size, spatial_method)

        if spat_sim < spatial_threshold:
            continue

        total_score = spat_sim + cand["sem"]

        if total_score > combined_threshold:
            if nid not in final_merges or total_score > final_merges[nid][0]:
                final_merges[nid] = (total_score, oid, spat_sim, cand["sem"])

    # Apply merges.
    if not final_merges:
        return

    removed_ids = set()
    sorted_merges = sorted(final_merges.items(), key=lambda x: x[1][0], reverse=True)
    id_map = {i: i for i in sg.all_objects.keys()}

    for nid, (score, oid, spat, sem) in sorted_merges:
        root_target = id_map.get(oid, oid)

        logging.info(
            f"[Fusion] Merge {nid} -> {root_target} | Score: {score:.2f} (Spat: {spat:.2f}, Sem: {sem:.2f})"
        )

        sg.merge_objects(source_id=nid, target_id=root_target)
        removed_ids.add(nid)
        id_map[nid] = root_target

    sg.new_detection_ids.clear()

    if removed_ids:
        update_global_tensors(mapper, id_map)

    return id_map


def compute_spatial_similarity(pts_a, pts_b, voxel_size, method):
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return 0.0

    if method == "nearest_neighbor":
        # Ratio of points in pts_a within voxel_size of any point in pts_b.
        # cdist is (N, M) — heavy memory if N, M > 20k.
        dists = torch.cdist(pts_a.float(), pts_b.float())
        min_dists, _ = torch.min(dists, dim=1)
        valid_count = (min_dists < voxel_size).sum()
        return (valid_count / pts_a.shape[0]).item()

    q_a = (pts_a / voxel_size).long().detach()
    q_b = (pts_b / voxel_size).long().detach()

    # Unique voxels (removes density bias).
    u_a = torch.unique(q_a, dim=0)
    u_b = torch.unique(q_b, dim=0)

    # Coordinate hashing for the intersection.
    p1, p2 = 73856093, 19349663
    hash_a = u_a[:, 0] * p1 + u_a[:, 1] * p2 + u_a[:, 2]
    hash_b = u_b[:, 0] * p1 + u_b[:, 1] * p2 + u_b[:, 2]

    intersect_mask = torch.isin(hash_a, hash_b)
    intersection = intersect_mask.sum().float()

    vol_a = float(u_a.shape[0])
    vol_b = float(u_b.shape[0])

    if method == "iou_min":
        min_vol = min(vol_a, vol_b)
        return (intersection / min_vol).item() if min_vol > 0 else 0.0

    elif method == "iou":
        union = vol_a + vol_b - intersection
        return (intersection / union).item() if union > 0 else 0.0

    elif method == "overlap":
        # Overlap relative to the new object (pts_a).
        return (intersection / vol_a).item() if vol_a > 0 else 0.0

    return 0.0


def overlap_from_voxel_sets(keys_a, keys_b):
    """Voxel overlap relative to the new object a: |a ∩ b| / |a|, on whole-object
    voxel-key arrays (resident + evicted) so an object spanning an evicted cell isn't
    undercounted."""
    if keys_a.shape[0] == 0:
        return 0.0
    inter = np.intersect1d(keys_a, keys_b, assume_unique=True).shape[0]
    return inter / float(keys_a.shape[0])


def _resolve_roots(changes):
    """Flatten transitive merge chains so {A: B, B: C} becomes {A: C, B: C}."""
    roots = {}
    for start in changes:
        path = []
        cur = start
        while cur in changes and cur not in roots:
            if cur in path:  # a source merges once, so cycles cannot form
                break
            path.append(cur)
            cur = changes[cur]
        root = roots.get(cur, cur)
        for node in path:
            roots[node] = root
    return {k: v for k, v in roots.items() if k != v}


def update_global_tensors(mapper, id_map):
    """Relabel the Gaussian _semantic tensors for every merge in one vectorized pass
    per cloud.

    Keying on the _semantic value (not row index) is what makes this safe: the point
    clouds are boolean-compacted several times per frame, so a persistent object->row
    directory would be invalidated constantly, but _semantic values are invariant under
    add/settle/delete, so a value-remap at merge time is correct regardless of how the
    rows have since shifted.
    """
    changes = _resolve_roots({k: v for k, v in id_map.items() if k != v})
    if not changes:
        return

    old_ids = sorted(changes.keys())
    new_ids = [changes[o] for o in old_ids]

    with torch.no_grad():
        for pcd in (mapper.active_gaussians, mapper.stable_gaussians):
            if pcd.get_points_num == 0:
                continue
            sem = pcd._semantic
            dev = sem.device
            sem_l = sem.view(-1).long()
            olds = torch.tensor(old_ids, device=dev, dtype=torch.long)
            news = torch.tensor(new_ids, device=dev, dtype=torch.long)
            # Per row, locate its slot among the sorted old-ids; the exact-match check
            # keeps ids absent from `changes` (background 0, unmerged objects) mapped to
            # themselves.
            idx = torch.bucketize(sem_l, olds).clamp(max=olds.numel() - 1)
            hit = olds[idx] == sem_l
            sem_l = torch.where(hit, news[idx], sem_l)
            pcd._semantic = sem_l.to(sem.dtype).view_as(sem)
            pcd.bump_version()

    # Mirror the merge onto the reduction index: remap keys in resident cells and journal
    # the remap for evicted (frozen) cells to replay at page-in.
    si = getattr(mapper, "sem_index", None)
    if si is not None:
        si.relabel(changes)
