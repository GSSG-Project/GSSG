"""Out-of-core eviction of the stable Gaussian cloud to keep GPU memory bounded by the
local working set rather than the whole map.

  * `stable_gaussians` stays ONE monolithic GaussianPointCloud holding only the
    GPU-resident rows. Cells are a pure function of position (`cell_of`), so cell
    membership is recomputed from `_xyz` whenever needed (value/position is the durable
    key, not row index).
  * Each frame, cells farther than `submap_radius` from every working-set camera
    (current frame + the local-optimization window + the recently-used keyframes) are
    evicted: their rows are `remove()`d from the resident cloud and stashed on the host
    (RAM), on disk, or on a hybrid of the two. Cells that come back within radius are
    paged in (`cat`).
  * Because `submap_radius >= max_depth`, any Gaussian a working-set camera can render
    lies in a resident cell, so the rendered set is identical to the all-resident run.

Out-of-core offline ops (so the whole map never has to fit on the GPU at once):
  * Export streams the stable PLY cell-by-cell (`save_full_stable_ply`): resident rows
    plus every evicted cell are written to one binary PLY with a peak of a single cell.
  * Full residency (`make_full_resident`) is used for the periodic/final joint global
    optimization, but only when the whole map fits a GPU budget; the mapper skips the
    final joint pass otherwise and exports in full via streaming.

Tier semantics:
  * "host"   — evicted cells live in CPU RAM.
  * "disk"   — evicted cells are torch.save'd to `page_dir`.
  * "hybrid" — cells stay in RAM until the RAM blobs exceed `submap_host_ram_frac` of
    total system RAM, then the least-recently-used cells spill to disk.

Eviction is opt-in via `submap_evict`; object geometry stays correct out-of-core because
the scene-graph readers answer from the SemanticReductionIndex instead of the resident
rows. Without it, `submapping: true` keeps everything resident (partition metadata is
still tracked, for inspection and Rerun).
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Cell coords packed into one int64 key; 21 bits/ground-plane axis (+-2^20 cells).
_BITS = 21
_OFF = 1 << (_BITS - 1)


def _total_ram_bytes() -> int:
    """Total physical RAM in bytes (Linux); conservative fallback elsewhere."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 16 * 1024**3


def _blob_bytes(params: dict) -> int:
    return sum(v.element_size() * v.nelement() for v in params.values() if torch.is_tensor(v))


class SubmapManager:
    def __init__(self, args):
        self.cell_size = float(getattr(args, "submap_cell_size", 4.0))
        self.vertical_axis = int(getattr(args, "vertical_axis", 2))
        # Residency radius (m) around every working-set camera. Must be >= max_depth so a
        # camera can't render a cell that has been evicted.
        max_depth = float(getattr(args, "max_depth", 6.0)) or 6.0
        r = float(getattr(args, "submap_radius", 0.0) or 0.0)
        # 0 = auto: max view distance + one cell.
        self.radius = r if r > 0 else max_depth + self.cell_size
        self.tier = getattr(args, "submap_evict_tier", "host")  # "host" | "disk" | "hybrid"
        self.page_dir = getattr(args, "submap_page_dir", "") or None
        if self.tier in ("disk", "hybrid") and not self.page_dir:
            self.page_dir = os.path.join(tempfile.gettempdir(), "gssg_submap_pages")
        self.host_ram_frac = float(getattr(args, "submap_host_ram_frac", 0.7) or 0.7)
        self._host_budget = int(self.host_ram_frac * _total_ram_bytes())
        # cell_id -> CPU param dict (in RAM) or {"path": ...} (on disk)
        self._evicted: dict[int, object] = {}
        self._evicted_meta: dict[int, dict] = {}  # cell_id -> {count, center, bytes}
        self._host_bytes = 0  # bytes of RAM-resident evicted blobs
        self._lru: list[int] = []  # RAM cell_ids, least-recently-used first
        self.n_evictions = 0
        self.n_pageins = 0
        self.n_spills = 0
        self.sem_index = None  # SemanticReductionIndex; set by the mapper when active

    # ------------------------------------------------------------------ cells
    def _plane_axes(self):
        # The two non-vertical axes (the ground plane), per vertical_axis convention.
        return {0: (1, 2), 1: (0, 2), 2: (0, 1)}.get(self.vertical_axis, (0, 1))

    def cell_of(self, xyz: torch.Tensor) -> torch.Tensor:
        """[N,3] world positions -> [N] int64 packed cell ids (axis-aware ground plane)."""
        a, b = self._plane_axes()
        u = torch.floor(xyz[:, a] / self.cell_size).long() + _OFF
        v = torch.floor(xyz[:, b] / self.cell_size).long() + _OFF
        return (u << _BITS) | v

    def _cell_center_plane(self, cell_id: int):
        v = (cell_id & ((1 << _BITS) - 1)) - _OFF
        u = (cell_id >> _BITS) - _OFF
        return (u + 0.5) * self.cell_size, (v + 0.5) * self.cell_size

    # -------------------------------------------------------------- residency
    def _camera_centers(self, mapping, frame):
        centers = []
        for cam in [frame, *list(mapping.processed_frames)]:
            if cam is not None:
                centers.append(cam.get_c2w[:3, 3])
        # Keyframes the periodic global pass may revisit must stay resident too.
        kf = getattr(mapping, "keyframe_list", [])
        gk = int(getattr(mapping, "global_keyframe_num", 3))
        for cam in kf[-gk:]:
            centers.append(cam.get_c2w[:3, 3])
        if not centers:
            return None
        return torch.stack([c.to(torch.float32) for c in centers], dim=0)  # [C,3]

    def _needed_cells(self, mapping, frame):
        """Cells within `radius` of any working-set camera, in the ground plane (vectorized
        corner-distance test over all cameras x candidate cells)."""
        centers = self._camera_centers(mapping, frame)
        if centers is None:
            return set()
        import math

        a, b = self._plane_axes()
        cen2d = centers[:, [a, b]].detach().cpu().numpy()  # [C, 2]
        cs, r = self.cell_size, self.radius
        span = int(math.ceil(r / cs)) + 1
        cu = np.floor(cen2d[:, 0] / cs).astype(np.int64)  # [C]
        cv = np.floor(cen2d[:, 1] / cs).astype(np.int64)
        d = np.arange(-span, span + 1, dtype=np.int64)
        du, dv = (g.ravel() for g in np.meshgrid(d, d, indexing="ij"))  # [S*S]
        u = cu[:, None] + du[None, :]  # [C, S*S]
        v = cv[:, None] + dv[None, :]
        nx = np.maximum(np.abs((u + 0.5) * cs - cen2d[:, 0:1]) - 0.5 * cs, 0.0)
        ny = np.maximum(np.abs((v + 0.5) * cs - cen2d[:, 1:2]) - 0.5 * cs, 0.0)
        keep = (nx * nx + ny * ny) <= r * r
        keys = ((u[keep] + _OFF) << _BITS) | (v[keep] + _OFF)
        return {int(k) for k in keys.tolist()}

    # ----------------------------------------------------------- evict / page
    def n_evicted(self) -> int:
        return len(self._evicted)

    def evicted_points(self) -> int:
        return sum(m["count"] for m in self._evicted_meta.values())

    def evicted_bytes(self) -> int:
        return sum(m.get("bytes", 0) for m in self._evicted_meta.values())

    def total_stable_count(self, resident_count: int) -> int:
        return resident_count + self.evicted_points()

    @torch.no_grad()
    def update_residency(self, mapping, frame):
        """Evict cells that left the working set; page back cells that re-entered it.
        Operates on the monolithic resident `mapping.stable_gaussians`."""
        self.set_resident_cells(mapping, self._needed_cells(mapping, frame))

    @torch.no_grad()
    def set_resident_cells(self, mapping, needed):
        """Make exactly the cells in `needed` resident: page in the needed-but-evicted ones,
        then bulk-evict the resident cells not in `needed`. Shared by per-frame residency
        (needed = working set) and the per-cell final optimize (needed = the cell refined now).

        Eviction is ONE bulk pass: remove all evicted rows at once, then split the removed
        rows by cell to stash per-cell blobs."""
        needed = {int(c) for c in needed}
        stable = mapping.stable_gaussians
        self._page_in_many(stable, [c for c in self._evicted if c in needed])

        _sx = stable.get_xyz
        if stable.get_points_num > 0 and _sx.ndim == 2 and _sx.shape[1] >= 3:
            cells = self.cell_of(_sx)
            present = torch.unique(cells)
            if needed:
                needed_t = torch.tensor(sorted(needed), dtype=cells.dtype, device=cells.device)
                evict_ids = present[~torch.isin(present, needed_t)]
            else:
                evict_ids = present
            if evict_ids.numel() > 0:
                params = stable.remove(torch.isin(cells, evict_ids))  # single bulk removal
                ec = self.cell_of(params["xyz"])  # cell of each removed row (one pass)
                for cid in evict_ids.tolist():
                    cmask = ec == cid
                    cell_params = {
                        k: (v[cmask] if torch.is_tensor(v) else v) for k, v in params.items()
                    }
                    self._evict_store(int(cid), cell_params)

    def resident_cell_ids(self, mapping):
        """Cell ids currently resident on the GPU (unique cells of the stable cloud)."""
        stable = mapping.stable_gaussians
        _sx = stable.get_xyz
        if stable.get_points_num == 0 or _sx.ndim != 2 or _sx.shape[1] < 3:
            return []
        return torch.unique(self.cell_of(_sx)).tolist()

    def all_cell_ids(self, mapping):
        """Every cell in the map — resident on the GPU plus evicted (host/disk)."""
        return sorted(set(self.resident_cell_ids(mapping)) | {int(c) for c in self._evicted})

    @torch.no_grad()
    def _evict(self, stable, cell_id: int, mask: torch.Tensor):
        self._evict_store(cell_id, stable.remove(mask))  # detaches those rows out of the cloud

    @torch.no_grad()
    def _evict_store(self, cell_id: int, params: dict):
        """Freeze + stash already-removed rows for one cell (sem-index + meta + host/disk blob)."""
        count = int(params["xyz"].shape[0])
        if count == 0:
            return
        center = params["xyz"].mean(dim=0).detach().cpu()
        cpu_params = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in params.items()}
        self._evicted_meta[cell_id] = {
            "count": count,
            "center": center,
            "bytes": _blob_bytes(cpu_params),
        }
        if self.sem_index is not None:
            # Freeze this cell's per-id reduction from the exact rows being evicted, so the
            # scene graph keeps recomposing the whole object after the cell leaves the GPU.
            self.sem_index.update_cells(
                np.full(count, cell_id, dtype=np.int64),
                cpu_params["semantic"].view(-1).numpy(),
                cpu_params["xyz"].numpy(),
            )
            self.sem_index.evict_cell(cell_id)
        if self.tier == "disk":
            self._store_disk(cell_id, cpu_params)
        else:  # "host" or "hybrid"
            self._store_host(cell_id, cpu_params)
            if self.tier == "hybrid":
                self._spill_until_under_budget()
        self.n_evictions += 1

    def _store_host(self, cell_id: int, cpu_params: dict):
        self._evicted[cell_id] = cpu_params
        self._host_bytes += _blob_bytes(cpu_params)
        self._touch(cell_id)

    def _store_disk(self, cell_id: int, cpu_params: dict):
        os.makedirs(self.page_dir, exist_ok=True)
        path = os.path.join(self.page_dir, f"cell_{cell_id}.pt")
        torch.save(cpu_params, path)
        self._evicted[cell_id] = {"path": path}

    def _touch(self, cell_id: int):
        """Mark a RAM cell as most-recently-used (move to the end of the LRU list)."""
        try:
            self._lru.remove(cell_id)
        except ValueError:
            pass
        self._lru.append(cell_id)

    def _spill_until_under_budget(self):
        """Spill least-recently-used RAM cells to disk until under the host budget."""
        if not self.page_dir:
            return
        while self._host_bytes > self._host_budget and self._lru:
            cell_id = self._lru.pop(0)  # least-recently-used
            blob = self._evicted.get(cell_id)
            if not isinstance(blob, dict) or "path" not in blob:  # still in RAM
                self._host_bytes -= _blob_bytes(blob)
                self._store_disk(cell_id, blob)
                self.n_spills += 1

    @torch.no_grad()
    def _load_blob(self, cell_id: int) -> dict:
        """Pop + materialize one evicted cell's CPU param dict: disk load if spilled, host
        accounting/LRU cleanup if resident, and journal-replay so the rows carry current
        semantic ids. Does NOT touch the GPU cloud (the H2D copy + cat happen in
        _page_in_many so a batch of cells is concatenated in one shot)."""
        blob = self._evicted.pop(cell_id)
        self._evicted_meta.pop(cell_id, None)
        if isinstance(blob, dict) and "path" in blob:
            path = blob["path"]
            blob = torch.load(path, map_location="cpu")
            try:
                os.remove(path)
            except OSError:
                pass
        else:
            self._host_bytes -= _blob_bytes(blob)
            try:
                self._lru.remove(cell_id)
            except ValueError:
                pass
        if self.sem_index is not None:
            # Replay the relabels that fused this object while the cell was frozen, so the
            # paged-in rows carry current semantic ids.
            journal = self.sem_index.pop_journal(cell_id)
            if journal:
                semv = blob["semantic"].view(-1).clone()
                for idmap in journal:  # in application order
                    for old, new in idmap.items():
                        semv[semv == int(old)] = int(new)
                blob["semantic"] = semv.view_as(blob["semantic"])
            self.sem_index.restore_cell(cell_id)
        return blob

    @torch.no_grad()
    def _page_in_many(self, stable, cell_ids):
        """Page a batch of evicted cells back onto the GPU with a SINGLE concatenation: merge
        the blobs on the host first, then one H2D copy + one cat -> O(resident).
        Order-independent (the map is a set), so identical to sequential page-in."""
        cell_ids = [c for c in cell_ids if c in self._evicted]
        if not cell_ids:
            return
        blobs = [self._load_blob(c) for c in cell_ids]
        if len(blobs) == 1:
            merged = blobs[0]
        else:
            merged = {
                k: (torch.cat([b[k] for b in blobs], dim=0) if torch.is_tensor(v) else v)
                for k, v in blobs[0].items()
            }
        cuda_params = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in merged.items()}
        stable.cat(cuda_params)
        self.n_pageins += len(cell_ids)

    @torch.no_grad()
    def _page_in(self, stable, cell_id: int):
        self._page_in_many(stable, [cell_id])

    @torch.no_grad()
    def make_full_resident(self, mapping):
        """Page every evicted cell back to GPU (the joint global optimize needs the whole
        map). The mapper gates this on a GPU budget."""
        if not self._evicted:
            return
        n = len(self._evicted)
        self._page_in_many(mapping.stable_gaussians, list(self._evicted))
        logging.info("[SUBMAP] full residency restored (%d cells paged in)", n)

    # ------------------------------------------------------- out-of-core ops
    def iter_evicted_cpu(self):
        """Yield each evicted cell's CPU param dict, one at a time (peak = one cell).
        Used by streaming export; does not change residency."""
        for cell_id in list(self._evicted):
            blob = self._evicted[cell_id]
            if isinstance(blob, dict) and "path" in blob:
                yield torch.load(blob["path"], map_location="cpu")
            else:
                yield blob

    @torch.no_grad()
    def save_full_stable_ply(
        self, stable, path: str, include_confidence: bool = True, include_anchor: bool = False
    ) -> int:
        """Write the WHOLE stable map (resident rows + every evicted cell) to one binary
        PLY, streaming cell-by-cell so peak memory is a single cell. Returns vertices written."""
        from gssg.map.gaussian_pointcloud import gaussian_ply_row_array, write_ply_streaming

        attr_names = stable.construct_list_of_attributes(include_confidence, include_anchor)
        total = int(stable.get_points_num) + self.evicted_points()

        def batches():
            if stable.get_points_num > 0:
                conf = stable._confidence if include_confidence else None
                anch = stable._anchor_frame if include_anchor else None
                yield gaussian_ply_row_array(
                    stable._xyz,
                    stable._features_dc,
                    stable._features_rest,
                    stable._opacity,
                    stable._scaling,
                    stable._rotation,
                    conf,
                    anch,
                )
            for blob in self.iter_evicted_cpu():
                conf = blob["confidence"] if include_confidence else None
                anch = blob["anchor_frame"] if include_anchor else None
                yield gaussian_ply_row_array(
                    blob["xyz"],
                    blob["features_dc"],
                    blob["features_rest"],
                    blob["opacity"],
                    blob["scaling"],
                    blob["rotation"],
                    conf,
                    anch,
                )

        n = write_ply_streaming(path, attr_names, batches(), total)
        logging.info(
            "[SUBMAP] streamed full stable PLY: %d verts (%d resident + %d evicted) -> %s",
            n,
            int(stable.get_points_num),
            self.evicted_points(),
            path,
        )
        return n

    # -------------------------------------------------- per-cell PLY export (viewer LOD)
    def _viewer_cell_size(self) -> float:
        """Cell size (m) used to group splats into one PLY per viewer cell.

        The on-GPU `cell_size` (default 4 m) yields too many tiny HTTP requests for the
        viewer, so it is coarsened to ~8-16 m super-cells. The coarse size is an integer
        multiple of `cell_size`, so a super-cell is an exact union of GPU cells and every
        splat lands in exactly one super-cell."""
        target = float(getattr(self, "viewer_cell_size", 0.0) or 0.0) or 12.0
        mult = max(1, round(target / self.cell_size))
        return mult * self.cell_size

    def _viewer_cell_of(self, xyz, cs: float):
        """[N,3] -> [N] int64 packed viewer-cell ids at coarse size `cs` (ground plane),
        matching the cell_of packing. Works on torch or numpy."""
        a, b = self._plane_axes()
        if torch.is_tensor(xyz):
            u = torch.floor(xyz[:, a] / cs).long() + _OFF
            v = torch.floor(xyz[:, b] / cs).long() + _OFF
            return (u << _BITS) | v
        u = np.floor(xyz[:, a] / cs).astype(np.int64) + _OFF
        v = np.floor(xyz[:, b] / cs).astype(np.int64) + _OFF
        return (u << _BITS) | v

    @torch.no_grad()
    def save_stable_ply_per_cell(
        self, stable_gaussians, out_dir: str, include_confidence: bool = True
    ) -> dict:
        """Partition the WHOLE stable map (resident rows by viewer-cell, PLUS every evicted
        cell) into per-cell groups and write one binary PLY per cell to
        `<out_dir>/cells/cell_{id}.ply`, plus `<out_dir>/cells.json` (a manifest of
        {cell_id, ply, center, aabb, count}). Peak memory is one viewer cell. Handles both
        the submapping case and the fully-resident case. Returns the parsed cells.json dict."""
        import json

        from gssg.map.gaussian_pointcloud import gaussian_ply_row_array

        cells_dir = os.path.join(out_dir, "cells")
        os.makedirs(cells_dir, exist_ok=True)
        attr_names = stable_gaussians.construct_list_of_attributes(include_confidence, False)
        cs = self._viewer_cell_size()

        # group_id -> {"count", "min"(np[3]), "max"(np[3]), "sum"(np[3])}
        groups: dict[int, dict] = {}

        def _accumulate(gid: int, xyz_np: np.ndarray):
            g = groups.get(gid)
            mn = xyz_np.min(axis=0)
            mx = xyz_np.max(axis=0)
            sm = xyz_np.sum(axis=0)
            if g is None:
                groups[gid] = {"count": xyz_np.shape[0], "min": mn, "max": mx, "sum": sm}
            else:
                g["count"] += xyz_np.shape[0]
                g["min"] = np.minimum(g["min"], mn)
                g["max"] = np.maximum(g["max"], mx)
                g["sum"] = g["sum"] + sm

        # Append-mode streaming so a viewer cell spanning resident rows and several evicted
        # cells still ends up in ONE file, peak-bounded by the largest contributing chunk.
        files: dict[int, object] = {}  # gid -> open file handle (header written, body appended)
        written: dict[int, int] = {}

        def _row_dtype():
            return np.dtype([(a, "<f4") for a in attr_names])

        def _append_rows(gid: int, rows: np.ndarray):
            if rows is None or rows.shape[0] == 0:
                return
            if rows.shape[1] != len(attr_names):
                raise ValueError(
                    f"per-cell PLY: batch has {rows.shape[1]} cols, expected {len(attr_names)}"
                )
            f = files.get(gid)
            if f is None:
                # Reserve the header with a placeholder count; rewritten after all appends.
                f = open(os.path.join(cells_dir, f"cell_{gid}.ply"), "wb+")
                placeholder = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    "element vertex 0000000000\n"
                    + "".join(f"property float {a}\n" for a in attr_names)
                    + "end_header\n"
                )
                f.write(placeholder.encode("ascii"))
                files[gid] = f
                written[gid] = 0
            rec = np.empty(rows.shape[0], dtype=_row_dtype())
            for j, a in enumerate(attr_names):
                rec[a] = rows[:, j]
            f.write(rec.tobytes())
            written[gid] += int(rows.shape[0])

        def _emit_chunk(xyz, f_dc, f_rest, opacity, scaling, rotation, conf):
            """Split one source chunk by viewer cell and append each sub-group's rows."""
            if xyz.shape[0] == 0:
                return
            rows = gaussian_ply_row_array(xyz, f_dc, f_rest, opacity, scaling, rotation, conf, None)
            xyz_np = rows[:, 0:3]  # gaussian_ply_row_array writes x,y,z first
            gids = self._viewer_cell_of(xyz_np, cs)
            for gid in np.unique(gids):
                m = gids == gid
                gid = int(gid)
                _append_rows(gid, rows[m])
                _accumulate(gid, xyz_np[m])

        # resident rows (sliced by GPU cell so peak stays bounded even fully resident)
        if stable_gaussians.get_points_num > 0:
            sx = stable_gaussians.get_xyz
            gpu_cells = self.cell_of(sx)
            conf_all = stable_gaussians._confidence if include_confidence else None
            for cid in torch.unique(gpu_cells).tolist():
                m = gpu_cells == cid
                _emit_chunk(
                    stable_gaussians._xyz[m],
                    stable_gaussians._features_dc[m],
                    stable_gaussians._features_rest[m],
                    stable_gaussians._opacity[m],
                    stable_gaussians._scaling[m],
                    stable_gaussians._rotation[m],
                    conf_all[m] if conf_all is not None else None,
                )

        # evicted cells (host/disk), one CPU blob at a time
        for blob in self.iter_evicted_cpu():
            conf = blob["confidence"] if include_confidence else None
            _emit_chunk(
                blob["xyz"],
                blob["features_dc"],
                blob["features_rest"],
                blob["opacity"],
                blob["scaling"],
                blob["rotation"],
                conf,
            )

        # Finalize: patch each header's vertex count and close.
        for gid, f in files.items():
            f.seek(0)
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {written[gid]:010d}\n"
                + "".join(f"property float {a}\n" for a in attr_names)
                + "end_header\n"
            )
            f.write(header.encode("ascii"))
            f.close()

        cells = []
        for gid, g in groups.items():
            center = (g["sum"] / max(1, g["count"])).tolist()
            mn, mx = g["min"].tolist(), g["max"].tolist()
            cells.append(
                {
                    "cell_id": int(gid),
                    "ply": f"cells/cell_{gid}.ply",
                    "center": [float(c) for c in center],
                    "aabb": [float(x) for x in (mn + mx)],
                    "count": int(g["count"]),
                }
            )
        cells.sort(key=lambda c: -c["count"])
        manifest = {
            "viewer_cell_size": float(cs),
            "vertical_axis": int(self.vertical_axis),
            "total_count": int(sum(c["count"] for c in cells)),
            "cell_count": len(cells),
            "cells": cells,
        }
        with open(os.path.join(out_dir, "cells.json"), "w") as f:
            json.dump(manifest, f)
        logging.info(
            "[SUBMAP] per-cell export: %d cells, %d verts (viewer cell %.1f m) -> %s",
            len(cells),
            manifest["total_count"],
            cs,
            os.path.join(out_dir, "cells.json"),
        )
        return manifest

    def status_line(self, resident_count: int) -> str:
        """One-line residency summary for periodic run-loop logging."""
        ev_pts = self.evicted_points()
        total = resident_count + ev_pts
        parts = [f"resident {resident_count / 1e3:.1f}k / {total / 1e3:.1f}k pts"]
        if self._evicted or self.n_evictions:
            host_mb = self._host_bytes / 1024**2
            disk_mb = max(self.evicted_bytes() - self._host_bytes, 0) / 1024**2
            seg = (
                f"evicted {len(self._evicted)} cells, {ev_pts / 1e3:.1f}k pts (RAM {host_mb:.0f} MB"
            )
            if disk_mb > 0:
                seg += f", disk {disk_mb:.0f} MB"
            parts.append(seg + ")")
            parts.append(f"pageins {self.n_pageins}, spills {self.n_spills}")
        else:
            parts.append("0 evicted (working set still within radius)")
        return "[SUBMAP] " + " | ".join(parts)

    def stats(self):
        return {
            "evicted_cells": len(self._evicted),
            "evicted_points": self.evicted_points(),
            "n_evictions": self.n_evictions,
            "n_pageins": self.n_pageins,
            "n_spills": self.n_spills,
            "host_mb": round(self._host_bytes / 1024**2, 1),
        }
