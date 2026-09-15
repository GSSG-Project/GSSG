import copy
import json
import logging
import os
import random
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from cuda_utils._C_depth import accumulate_gaussian_error
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gssg.dataset_reader.cameras import Camera
from gssg.map import gaussian_pointcloud
from gssg.map.gaussian_pointcloud import GaussianPointCloud, configure_bg_scale_mult
from gssg.map.grid_filter import radius_inside_mask
from gssg.map.render import Renderer
from gssg.map.submap import SubmapManager
from gssg.scene_graph.room_segmentation import segment_rooms_from_arrays
from gssg.scene_graph.semantic_fusion import incremental_object_fusion
from gssg.utils.eval import eval_frame
from gssg.utils.export_frame import export_frame_and_params
from gssg.utils.fps import FPSMeter
from gssg.utils.general_utils import devB, devF, devI, inverse_sigmoid
from gssg.utils.object_room_viz import export_object_room_map
from gssg.utils.openlex3d_export import export_for_openlex3d
from gssg.utils.rerun_utils import rerun_log_gaussians
from gssg.utils.runtime_stats import RuntimeStats
from gssg.utils.sampling_helper import (
    configure_objectness_weights,
    loss_update_v1,
    loss_update_v2,
    sample_pixels_v1,
    sample_pixels_v2,
    smooth_objectness,
)
from gssg.utils.utils import (
    NO_CUDA_IPC,
    bbox_filter,
    colorerror2tilemask,
    frame_tensors_to,
    merge_ply,
    move_to_cpu,
    move_to_cpu_map,
    move_to_gpu,
    move_to_gpu_map,
    restore_frame_map_to_gpu,
    rot_compare,
    slerp,
    trans_compare,
    transmission2tilemask,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def adaptive_iter_count(base_iter, added_points, uniform_sample_num, trans_ratio, min_ratio):
    """Scale the per-frame optimization iteration budget by frame novelty."""
    novelty = max(
        added_points / max(1.0, 0.3 * uniform_sample_num),
        trans_ratio / 0.10,
    )
    novelty = min(1.0, novelty)
    scale = min_ratio + (1.0 - min_ratio) * novelty
    return max(4, int(round(base_iter * scale)))


_NVML_HANDLE = None


def _nvml_used_gb():
    """Whole-GPU memory in use, i.e. what nvidia-smi reports. 0.0 if unavailable."""
    global _NVML_HANDLE
    try:
        import pynvml

        if _NVML_HANDLE is None:
            pynvml.nvmlInit()
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE).used / 2**30
    except Exception:
        return 0.0


class Mapping:
    def __init__(self, args, scene_graph=None) -> None:
        self.args = args
        self.seed_gaussians = GaussianPointCloud(args, scene_graph, name="seed_gaussians")
        self.active_gaussians = GaussianPointCloud(args, scene_graph, name="active_gaussians")
        self.stable_gaussians = GaussianPointCloud(args, scene_graph, name="stable_gaussians")
        self.scene_graph = scene_graph

        self.sampling_version = args.sampling_version
        self.loss_function_version = args.sampling_version
        self.sampling_function = (
            sample_pixels_v1 if self.sampling_version == 1 else sample_pixels_v2
        )
        # Set here (not in run.py) so the MP variant's mapping process gets it too.
        gamma = getattr(args, "sample_gamma", 3.0)
        self.sample_gamma = gamma
        configure_objectness_weights(gamma, getattr(args, "sample_beta", 0.3))
        # Object-centric mapping is v2-only. v1 (or gamma=0) is the semantics-blind
        # baseline: uniform sampling, plain L1 loss, no background-splat enlargement.
        use_objectness = self.sampling_version == 2 and gamma > 0
        configure_bg_scale_mult(getattr(args, "bg_scale_mult", 1.5) if use_objectness else 1.0)

        self.renderer = Renderer(args)
        self.optimizer = None
        self.time = 0
        self.iter = 0
        self.gaussian_update_iter = args.gaussian_update_iter
        self.gaussian_update_frame = args.gaussian_update_frame
        self.frustum_cull_stable = getattr(args, "frustum_cull_stable", True)
        self.rerun_log_step = max(1, getattr(args, "rerun_log_step", 10))
        self.final_global_iter = args.final_global_iter

        self.memory_length = args.memory_length
        self.optimize_frames_ids = []
        self.processed_frames = deque(maxlen=self.memory_length)
        self.processed_map = deque(maxlen=self.memory_length)
        self.keyframe_ids = []
        self.keyframe_list = []
        self.keymap_list = []
        self.global_keyframe_num = args.global_keyframe_num
        self.keyframe_trans_thres = args.keyframe_trans_thres
        self.keyframe_theta_thres = args.keyframe_theta_thres
        # Mapper-side keyframe thresholds for local-vs-global routing in check_keyframe
        # (0 => fall back to the perception thresholds above). Looser here => more
        # local_optimize => active graduates to stable => active pool stays bounded.
        self._kf_theta_thres_global = (
            float(getattr(args, "keyframe_theta_thres_global", 0.0)) or self.keyframe_theta_thres
        )
        self._kf_trans_thres_global = (
            float(getattr(args, "keyframe_trans_thres_global", 0.0)) or self.keyframe_trans_thres
        )
        self.history_merge_max_weight = args.history_merge_max_weight

        self.uniform_sample_num = args.uniform_sample_num
        self.add_depth_thres = args.add_depth_thres
        self.add_normal_thres = args.add_normal_thres
        self.add_color_thres = args.add_color_thres
        self.add_transmission_thres = args.add_transmission_thres

        self.transmission_sample_ratio = args.transmission_sample_ratio
        self.error_sample_ratio = args.error_sample_ratio
        self.stable_confidence_thres = args.stable_confidence_thres
        self.unstable_time_window = args.unstable_time_window
        self.max_radius = args.max_radius
        # "grid" = voxel-hash fixed-radius seed filter; "knn" = pytorch3d kNN.
        self.seed_filter_mode = getattr(args, "seed_filter_mode", "grid")
        # Out-of-core submapping. See gssg/map/submap.py.
        self.submapping = getattr(args, "submapping", False)
        self.submaps = SubmapManager(args) if self.submapping else None
        # Out-of-core eviction of the stable cloud (opt-in). Object geometry is answered from
        # the reduction index below, so eviction stays correct with semantics on.
        self._submap_evict = self.submapping and getattr(args, "submap_evict", False)
        self._submap_final_pass = getattr(args, "submap_final_pass", "auto")
        self._submap_gpu_budget_bytes = int(
            float(getattr(args, "submap_gpu_budget_gb", 0.0) or 0.0) * 1024**3
        )
        self._submap_verify_every = int(getattr(args, "submap_verify_every", 0) or 0)
        # Commit-on-leave: bound the active pool for scalable traverses (needs submapping).
        self.commit_on_leave = bool(getattr(args, "commit_on_leave", False))
        self._commit_min_age = int(getattr(args, "commit_min_age", 10) or 0)
        self._committed_on_leave = 0
        # Hard active-pool ceiling. 0 = off.
        self._submap_active_max = int(getattr(args, "submap_active_max", 0) or 0)
        self._forced_grad = 0
        # Per-frame watchdog: warn when one mapping() exceeds this many ms (0 = off).
        self._mapper_watchdog_ms = float(getattr(args, "mapper_watchdog_ms", 0.0) or 0.0)
        # Hard per-frame wall-clock cap on the local_optimize loop (ms). 0 = off. When set,
        # the optimize loop stops early once the budget is spent (after >=1 iter).
        self._optimize_budget_ms = float(getattr(args, "optimize_budget_ms", 0.0) or 0.0)
        if self._submap_active_max and self.submaps is not None:
            print(
                f"[COMMIT] active-pool hard cap ON: submap_active_max={self._submap_active_max}",
                flush=True,
            )
        if self.commit_on_leave and self.submaps is None:
            logging.warning("[COMMIT] commit_on_leave needs submapping:true — disabling.")
            self.commit_on_leave = False
        elif self.commit_on_leave:
            print(
                f"[COMMIT] commit-on-leave ON (min_age={self._commit_min_age}): active cells "
                "leaving the working set freeze to stable -> active pool stays O(local).",
                flush=True,
            )
        # Per-(cell,id) reduction index so object geometry recomposes across evicted cells.
        self.sem_index = None
        if self._submap_evict and self.submaps is not None:
            from gssg.map.submap_semantic import SemanticReductionIndex

            self.sem_index = SemanticReductionIndex()
            self.submaps.sem_index = self.sem_index
            logging.info(
                "[SUBMAP] eviction ON (object geometry via reduction "
                "index; object_fusion forced synchronous)"
            )
        self._submap_log_every = max(1, int(getattr(args, "fps_log_every", 20)))
        self._submap_log_tick = 0
        if self.submapping:
            eviction = "ON" if self._submap_evict else "OFF (all-resident)"
            # print(), not logging: the mapper subprocess pins the root logger at WARNING,
            # so an info-level banner would be invisible exactly when submapping is active.
            print(
                f"[SUBMAP] enabled (cell={self.submaps.cell_size:.1f}m, "
                f"radius={self.submaps.radius:.1f}m, eviction={eviction})",
                flush=True,
            )
        # Novelty-driven local_optimize budget (see adaptive_iter_count).
        self.adaptive_optimize = getattr(args, "adaptive_optimize", False)
        self.adaptive_min_iter_ratio = getattr(args, "adaptive_min_iter_ratio", 0.25)
        self._last_added_points = 0
        self._last_trans_ratio = 1.0

        # All map shape is [H, W, C], please note the raw image shape is [C, H, W]
        self.min_depth, self.max_depth = args.min_depth, args.max_depth
        self.depth_filter = args.depth_filter
        self.frame_map = {
            "depth_map": torch.empty(0),
            "color_map": torch.empty(0),
            "normal_map_c": torch.empty(0),
            "normal_map_w": torch.empty(0),
            "vertex_map_c": torch.empty(0),
            "vertex_map_w": torch.empty(0),
            "confidence_map": torch.empty(0),
        }
        self.model_map = {
            "render_color": torch.empty(0),
            "render_depth": torch.empty(0),
            "render_normal": torch.empty(0),
            "render_color_index": torch.empty(0),
            "render_depth_index": torch.empty(0),
            "render_transmission": torch.empty(0),
            "confidence_map": torch.empty(0),
        }

        self.save_path = args.save_path
        self.save_step = args.save_step
        self.verbose = args.verbose
        self.processing_mode = args.processing_mode
        self.dataset_type = args.dataset_type
        assert self.processing_mode == "single" or self.processing_mode == "multi"
        self.use_tensorboard = args.use_tensorboard
        self.tb_writer = None

        self.feature_lr_coef = args.feature_lr_coef
        self.scaling_lr_coef = args.scaling_lr_coef
        self.rotation_lr_coef = args.rotation_lr_coef

        _lvl = getattr(args, "log_level", "INFO").upper()
        logging.getLogger().setLevel(getattr(logging, _lvl, logging.INFO))

        # torch.compile() on the renderer — opt-in; falls back silently on failure
        if getattr(args, "compile_renderer", False):
            try:
                self.renderer.render = torch.compile(
                    self.renderer.render,
                    mode="reduce-overhead",
                    dynamic=True,
                )
                logging.info("[OPT] renderer compiled with torch.compile(reduce-overhead)")
            except Exception as e:
                logging.warning(f"[OPT] torch.compile failed, using eager renderer: {e}")

        # Background object_fusion — opt-in; runs in a worker thread, awaited next frame
        self._background_fusion = getattr(args, "background_fusion", False)
        self._main_thread_id = threading.get_ident()
        if self._background_fusion:
            self._fusion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fusion")
            self._fusion_future = None
            logging.info("[OPT] object_fusion running in background thread")
        else:
            self._fusion_executor = None
            self._fusion_future = None

        # Background checkpoint writer: the periodic save snapshots the active +
        # resident-stable clouds to CPU and serializes their PLYs on a worker thread, so
        # the mapper loop never blocks on disk I/O (see _background_checkpoint).
        self._checkpoint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")
        self._checkpoint_future = None

        self.runtime_stats = RuntimeStats()

    def mapping(self, frame, frame_map, frame_id, optimization_params):
        # Set the rerun timeline cursor for this subprocess so the per-frame logs land at
        # the current tracker frame index (otherwise they all stack at time=0).
        if getattr(self.args, "visualize", False):
            import rerun as rr

            rr.set_time("time", sequence=int(frame_id))

        logging.info(
            f"[GAUSSIANS] Total Number of STBL: {self.stable_gaussians.get_points_num} "
            f"PCD: {self.active_gaussians.get_points_num} TEMP: {self.seed_gaussians.get_points_num}"
        )

        # Await any background fusion submitted on the previous frame before this
        # frame touches the scene graph or gaussian _semantic tensors.
        if self._background_fusion and self._fusion_future is not None:
            self._fusion_future.result()
            self._fusion_future = None

        self.frame_map = frame_map

        # Bound the active pool: freeze active cells that left the working set to stable
        # (by location, not confidence) so the optimized set stays O(local). Runs before
        # eviction so committed cells page out the same frame and the sem_index stays
        # consistent (eviction updates it).
        if self.commit_on_leave:
            self._commit_active_on_leave(frame)
        if self._submap_active_max:
            self._enforce_active_cap(frame)

        # Evict stable cells that left the working set / page back those that re-entered,
        # before this frame's renders.
        if self._submap_evict:
            self.submaps.update_residency(self, frame)

        with self.runtime_stats.timer("assoc_graph"):
            self.scene_graph.process_semantic_results(
                frame_map, frame_id, frame.get_c2w.cpu().numpy()
            )

        with self.runtime_stats.timer("seed_add"):
            self.gaussians_add(frame)

        self.processed_frames.append(frame)
        self.processed_map.append(frame_map)

        if (self.time + 1) % self.gaussian_update_frame == 0 or self.time == 0:
            self.optimize_frames_ids.append(frame_id)
            is_keyframe = self.check_keyframe(frame, frame_id)
            move_to_gpu(frame)
            with self.runtime_stats.timer("optimize", cuda_sync=True):
                if not is_keyframe or self.get_stable_num <= 0:
                    self.local_optimize(frame, optimization_params)
                else:
                    self.global_optimization(
                        optimization_params, select_keyframe_num=self.global_keyframe_num
                    )
            self.gaussians_delete_outliers(unstable=False)

        self.gaussians_upgrade()
        self.gaussians_delete_mismatches()
        self.gaussians_delete_outliers()
        # Keep the reduction index race-free by running fusion on the main thread.
        if self._background_fusion and self.sem_index is None:

            def _timed_fusion():
                with self.runtime_stats.timer("fusion"):
                    self.object_fusion()

            self._fusion_future = self._fusion_executor.submit(_timed_fusion)
        else:
            with self.runtime_stats.timer("fusion"):
                self.object_fusion()
        if (
            self.sem_index is not None
            and self._submap_verify_every
            and ((frame_id + 1) % self._submap_verify_every == 0)
        ):
            self._refresh_sem_index()
            self.verify_sem_index()

        # The full (growing) cloud is the heaviest Rerun payload; logging it every
        # frame outpaces the viewer and backpressures the SLAM thread to a stall.
        # Throttle to every rerun_log_step frames (set 1 to restore per-frame).
        if self.args.visualize and self.time % self.rerun_log_step == 0:
            rerun_log_gaussians(self)
        move_to_cpu(frame)

        # Periodic status: submap residency (when evicting) + active-pool size + commits,
        # on the same cadence as the [MAPPER] FPS line. print() survives the subprocess
        # WARNING level.
        if (self._submap_evict or self.commit_on_leave) and self.submaps is not None:
            self._submap_log_tick += 1
            if self._submap_log_tick % self._submap_log_every == 0:
                line = self.submaps.status_line(int(self.stable_gaussians.get_points_num))
                if self.commit_on_leave:
                    line += (
                        f" | active {int(self.active_gaussians.get_points_num) / 1e3:.1f}k"
                        f" | committed {self._committed_on_leave / 1e3:.1f}k"
                    )
                    if self._submap_active_max:
                        line += f" | forced_grad {self._forced_grad / 1e3:.1f}k"
                print(line, flush=True)

    def _can_fit_full_resident(self):
        """True if paging every evicted cell back to GPU fits the budget (or nothing is
        evicted). Budget = submap_gpu_budget_gb, or a fraction of free VRAM when auto."""
        if not (self._submap_evict and self.submaps is not None) or not self.submaps._evicted:
            return True
        try:
            free, _ = torch.cuda.mem_get_info()
        except Exception:
            return True
        budget = self._submap_gpu_budget_bytes or int(0.7 * free)
        return self.submaps.evicted_bytes() < budget

    def _final_pass_should_run(self):
        """Whether the monolithic joint final global optimize can run. With eviction it
        needs the whole map resident, so it is gated on the GPU budget (submap_final_pass:
        auto = run iff it fits; always = force; skip = never when evicting)."""
        if self._submap_final_pass == "always":
            return True
        if (
            self._submap_final_pass == "skip"
            and self._submap_evict
            and self.submaps is not None
            and self.submaps._evicted
        ):
            return False
        return self._can_fit_full_resident()

    def ensure_full_residency(self):
        """Page every evicted submap cell back to GPU when it fits the GPU budget; no-op
        unless evicting. When the map exceeds the budget the cells stay out-of-core."""
        if not (self._submap_evict and self.submaps is not None) or not self.submaps._evicted:
            return
        if self._can_fit_full_resident():
            self.submaps.make_full_resident(self)
        else:
            logging.warning(
                "[SUBMAP] full residency NOT restored: map exceeds the GPU budget; "
                "%d cells stay out-of-core (export streams, final joint pass skipped).",
                len(self.submaps._evicted),
            )

    def _evicting(self) -> bool:
        """True when out-of-core eviction is active and at least one cell is off-GPU."""
        return self._submap_evict and self.submaps is not None and self.submaps.n_evicted() > 0

    def await_checkpoint(self):
        """Join any in-flight background checkpoint write; call before the authoritative
        final save so the whole-map PLY isn't interleaved with a periodic one."""
        if self._checkpoint_future is not None:
            self._checkpoint_future.result()
            self._checkpoint_future = None

    @torch.no_grad()
    def _background_checkpoint(self):
        """Periodic non-blocking progress checkpoint: snapshot the active + resident-stable
        clouds to CPU and write their geometry PLYs on a worker thread, so the mapper loop
        blocks on neither disk I/O nor paging the evicted map back. The out-of-core stable
        map is not rewritten every step (its evicted cells are durable on disk and streamed
        once into the authoritative PLY at finalize). If the previous checkpoint is still
        writing, skip this one rather than queue and stall."""
        if self._checkpoint_future is not None and not self._checkpoint_future.done():
            logging.info("[CKPT] previous checkpoint still writing; skipping frame %d", self.time)
            return
        out_dir = os.path.join(self.save_path, "save_model", f"frame_{self.time:04d}")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"iter_{self.iter:04d}")
        active_snap = self.active_gaussians.clone_to_cpu() if self.get_unstable_num > 0 else None
        stable_snap = self.stable_gaussians.clone_to_cpu() if self.get_stable_num > 0 else None
        if stable_snap is not None:
            self._last_stable_ply = os.path.relpath(base + "_stable.ply", self.save_path)

        def _write():
            if active_snap is not None:
                active_snap.save_model_ply(base + ".ply", include_confidence=True)
            if stable_snap is not None:
                stable_snap.save_model_ply(
                    base + "_stable.ply", include_confidence=True, include_anchor=True
                )

        self._checkpoint_future = self._checkpoint_executor.submit(_write)

    def gaussians_add(self, frame):
        self.temp_points_init(frame)
        self.temp_points_filter()
        self.temp_points_attach(frame)
        self.temp_to_optimize()

    def await_background_fusion(self):
        """Block until any in-flight background object_fusion worker finishes.

        Must be called before the main thread touches the scene graph directly, because
        the rtree spatial index (libspatialindex) is not thread-safe: a concurrent delete
        on the main thread + intersection on the worker thread segfaults.
        """
        if self._background_fusion and self._fusion_future is not None:
            self._fusion_future.result()
            self._fusion_future = None

    def object_fusion(self, global_run=False):
        # If called from the main thread (e.g., final global pass from run.py),
        # ensure any pending background fusion has completed first to avoid
        # concurrent modification of the scene graph.
        if (
            self._background_fusion
            and self._fusion_future is not None
            and threading.get_ident() == self._main_thread_id
        ):
            self._fusion_future.result()
            self._fusion_future = None

        fused_id_map = incremental_object_fusion(self, global_run)
        if fused_id_map:
            updated_ids = list(fused_id_map.keys()) + list(fused_id_map.values())
            self.update_object_geometry(updated_ids)

    def update_object_geometry(self, object_ids):
        if 0 in object_ids:
            object_ids.remove(0)
        if self.sem_index is not None:
            self._refresh_sem_index()
        self.scene_graph.update_object_geometry(
            self.active_gaussians, self.stable_gaussians, object_ids, sem_index=self.sem_index
        )

    @torch.no_grad()
    def _refresh_sem_index(self):
        """Recompute the reduction index for all resident stable cells (evicted cells stay
        frozen); bounded by the working set."""
        st = self.stable_gaussians
        xyz_t = st.get_xyz
        if xyz_t.ndim != 2 or xyz_t.shape[0] == 0 or xyz_t.shape[1] < 3:
            if xyz_t.shape[0] > 0:
                logging.warning(
                    "[SUBMAP] skipping sem-index refresh: degenerate stable xyz %s",
                    tuple(xyz_t.shape),
                )
            return
        cells = self.submaps.cell_of(xyz_t).cpu().numpy()
        sem = st._semantic.view(-1).cpu().numpy()
        xyz = xyz_t.detach().cpu().numpy()
        self.sem_index.update_cells(cells, sem, xyz)
        self.sem_index.prune_resident_cells(cells)

    @torch.no_grad()
    def verify_sem_index(self):
        """Check that the index's per-object aggregate equals a gather over the whole
        stable cloud (resident plus every evicted blob); logs mismatched ids."""
        if self.sem_index is None:
            return True
        import numpy as _np

        st = self.stable_gaussians
        sem_parts, xyz_parts = [], []
        if st.get_points_num > 0:
            sem_parts.append(st._semantic.view(-1).cpu().numpy())
            xyz_parts.append(st._xyz.detach().cpu().numpy())
        # Evicted blobs keep their pre-merge ids until page-in, while the index relabels
        # keys live — so replay each cell's journal on the gathered ids before comparing
        # (this is exactly what page-in will do to the blob).
        for cell_id, blob in list(self.submaps._evicted.items()):
            if isinstance(blob, dict) and "path" in blob:
                blob = torch.load(blob["path"], map_location="cpu")
            sem = blob["semantic"].view(-1).numpy().copy()
            for idmap in self.sem_index.journal.get(int(cell_id), []):
                for old, new in idmap.items():
                    sem[sem == int(old)] = int(new)
            sem_parts.append(sem)
            xyz_parts.append(blob["xyz"].numpy())
        if not sem_parts:
            return True
        ok, bad = self.sem_index.verify_equivalence(
            _np.concatenate(sem_parts), _np.concatenate(xyz_parts)
        )
        # print() (not logging) so the signal survives the run's INFO suppression.
        msg = (
            f"[SUBMAP] semantic-index check {'OK' if ok else 'FAILED'} | "
            f"evicted_cells={self.submaps.n_evicted()} evicted_pts={self.submaps.evicted_points()}"
        )
        if not ok:
            msg += f" mismatched_ids={bad[:10]}"
        print(msg, flush=True)
        return ok

    def _merge_global_params(self, unstable_params, stable_params):
        keys = (
            "xyz",
            "opacity",
            "scales",
            "rotations",
            "shs",
            "radius",
            "normal",
            "confidence",
            "semantics",
        )
        return {k: devF(torch.cat([unstable_params[k], stable_params[k]])) for k in keys}

    def _stable_frustum_mask(self, frame, xyz):
        # Off-view stable gaussians contribute nothing to this frame's pixels and receive
        # no gradient (stable is detached in local_optimize), so they are dropped before
        # rasterizing, making the render scale with the visible count, not the whole map.
        if xyz.shape[0] == 0:
            return torch.ones(0, dtype=torch.bool, device=xyz.device)
        w2c = frame.get_w2c().to(xyz.device)
        intrinsic = frame.get_intrinsic.to(xyz.device)
        xyz_c = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        uvw = xyz_c @ intrinsic.T
        uv = uvw[:, :2] / uvw[:, 2:].clamp(min=1e-6)
        # The rasterizer's frustum cull is symmetric about the principal point (cx, cy),
        # not the image centre, so the bound is centred there too — taken from the
        # intrinsic so it matches the projection for offset-principal-point cameras
        # (ROS/ZED/HM3D). The half-width margin retains splats overlapping the border.
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        hw, hh = 0.65 * frame.image_width, 0.65 * frame.image_height
        return (
            (xyz_c[:, 2] > frame.znear)
            & (uv[:, 0] >= cx - hw)
            & (uv[:, 0] < cx + hw)
            & (uv[:, 1] >= cy - hh)
            & (uv[:, 1] < cy + hh)
        )

    def _cull_stable(self, frame, stable_params):
        # Restrict a stable-param dict to the frame's frustum. Returns (culled params,
        # global row indices kept); vis_idx maps a position in the culled set back to its
        # row in stable_gaussians, needed wherever the rasterizer's per-pixel indices are
        # dereferenced. Returns the input unchanged with vis_idx=None when culling is off.
        xyz = stable_params["xyz"]
        if not self.frustum_cull_stable or xyz.shape[0] == 0:
            return stable_params, None
        mask = self._stable_frustum_mask(frame, xyz)
        vis_idx = mask.nonzero(as_tuple=True)[0]
        return {k: v[mask] for k, v in stable_params.items()}, vis_idx

    def local_optimize(self, frame, update_args):
        param_groups = self.active_gaussians.parametrize(update_args)
        history_stat = {
            "opacity": self.active_gaussians._opacity.detach().clone(),
            "confidence": self.active_gaussians.get_confidence.detach().clone(),
            "xyz": self.active_gaussians._xyz.detach().clone(),
            "features_dc": self.active_gaussians._features_dc.detach().clone(),
            "features_rest": self.active_gaussians._features_rest.detach().clone(),
            "scaling": self.active_gaussians._scaling.detach().clone(),
            "rotation": self.active_gaussians.get_rotation.detach().clone(),
            "rotation_raw": self.active_gaussians._rotation.detach().clone(),
        }
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        gaussian_update_iter = self.gaussian_update_iter
        if self.adaptive_optimize and self.time > 0:
            gaussian_update_iter = adaptive_iter_count(
                self.gaussian_update_iter,
                self._last_added_points,
                self.uniform_sample_num,
                self._last_trans_ratio,
                self.adaptive_min_iter_ratio,
            )
            if self.verbose and gaussian_update_iter < self.gaussian_update_iter:
                logging.info(
                    f"[ADAPT] low-novelty frame: {gaussian_update_iter}/"
                    f"{self.gaussian_update_iter} iters"
                )
        render_masks = []
        tile_masks = []
        for frame in self.processed_frames:
            render_mask, tile_mask, render_ratio = self.evaluate_render_range(frame)
            render_masks.append(render_mask)
            tile_masks.append(tile_mask)
            if self.verbose:
                tile_raito = 1
                if tile_mask is not None:
                    tile_raito = tile_mask.sum() / torch.numel(tile_mask)
                logging.info(f"tile mask ratio: {tile_raito:f}")
        logging.info(
            f"Unstable gaussians: {self.get_unstable_num}, Stable gaussians: {self.get_stable_num}"
        )

        # Stable gaussians are constant during this loop (mutated only after
        # local_optimize), so derive them once instead of rebuilding the full map each iter.
        stable_cached = {k: v.detach() for k, v in self.stable_params.items()}
        # Cull stable gaussians to the optimized frame's frustum. The camera is fixed per
        # frame index, so the visible subset is gathered once per index and reused.
        cull = self.frustum_cull_stable and stable_cached["xyz"].shape[0] > 0
        vis_cache = {}

        # Optional hard wall-clock cap: stop optimizing once the per-frame budget is spent
        # (after >=1 iteration) so a single heavy frame can't blow the budget.
        budget_ms = self._optimize_budget_ms
        t_opt0 = time.perf_counter()
        with tqdm(total=gaussian_update_iter, desc="Map Update", disable=not self.verbose) as pbar:
            for iter in range(gaussian_update_iter):
                self.iter = iter
                random_index = random.randint(0, len(self.processed_frames) - 1)
                if iter > gaussian_update_iter / 2:
                    random_index = -1
                opt_frame = self.processed_frames[random_index]
                opt_frame_map = self.processed_map[random_index]
                opt_render_mask = render_masks[random_index]
                opt_tile_mask = tile_masks[random_index]
                if cull:
                    stable_use = vis_cache.get(random_index)
                    if stable_use is None:
                        m = self._stable_frustum_mask(opt_frame, stable_cached["xyz"])
                        stable_use = {k: v[m] for k, v in stable_cached.items()}
                        vis_cache[random_index] = stable_use
                else:
                    stable_use = stable_cached
                render_ouput = self.renderer.render(
                    opt_frame,
                    self._merge_global_params(self.unstable_params, stable_use),
                    tile_mask=opt_tile_mask,
                )
                image_input = {
                    "color_map": devF(opt_frame_map["color_map"]),
                    "depth_map": devF(opt_frame_map["depth_map"]),
                    "normal_map": devF(opt_frame_map["normal_map_w"]),
                    "objectness_map": (
                        devF(opt_frame_map["objectness_map"])
                        if opt_frame_map["objectness_map"] is not None
                        else None
                    ),
                }
                loss, reported_losses = self.loss_update(
                    render_ouput,
                    image_input,
                    history_stat,
                    update_args,
                    render_mask=opt_render_mask,
                    unstable=True,
                )
                pbar.set_postfix({"loss": f"{loss:1.5f}"})
                pbar.update(1)
                if budget_ms and (time.perf_counter() - t_opt0) * 1000.0 >= budget_ms:
                    break

        self.active_gaussians.detach()
        self.iter = 0

        self.history_merge(history_stat, self.history_merge_max_weight)

    def history_merge(self, history_stat, max_weight=0.5):
        if max_weight <= 0:
            return
        history_weight = (
            max_weight * history_stat["confidence"] / (self.active_gaussians.get_confidence + 1e-6)
        )
        if self.verbose:
            logging.info("===== History Merge ======")
            logging.info(f"History Weight: {history_weight.mean():.2f}")
        xyz_merge = (
            history_stat["xyz"] * history_weight
            + (1 - history_weight) * self.active_gaussians.get_xyz
        )

        features_dc_merge = (
            history_stat["features_dc"] * history_weight[0]
            + (1 - history_weight[0]) * self.active_gaussians._features_dc
        )
        features_rest_merge = (
            history_stat["features_rest"] * history_weight[0]
            + (1 - history_weight[0]) * self.active_gaussians._features_rest
        )
        scaling_merge = (
            history_stat["scaling"] * history_weight[0]
            + (1 - history_weight[0]) * self.active_gaussians._scaling
        )

        rotation_merge = slerp(
            history_stat["rotation"], self.active_gaussians.get_rotation, 1 - history_weight
        )

        self.active_gaussians._xyz = xyz_merge
        self.active_gaussians._features_dc = features_dc_merge
        self.active_gaussians._features_rest = features_rest_merge
        self.active_gaussians._scaling = scaling_merge
        self.active_gaussians._rotation = rotation_merge
        self.active_gaussians.bump_version()

    @torch.no_grad()
    def _commit_active_on_leave(self, frame):
        """Force-graduate active gaussians whose ground-plane cell has left the working set
        to stable, regardless of confidence (the camera has moved on), bounding the active
        (Adam-optimized) pool to the local working set instead of the whole trajectory.
        No-op unless commit_on_leave is on and submapping exists."""
        if not self.commit_on_leave or self.submaps is None:
            return
        active = self.active_gaussians
        if active.get_points_num == 0:
            return
        needed = self.submaps._needed_cells(self, frame)  # set of working-set cell ids
        if not needed:
            return
        cells = self.submaps.cell_of(active.get_xyz)  # [N] int64
        needed_t = torch.tensor(sorted(needed), dtype=cells.dtype, device=cells.device)
        left = ~torch.isin(cells, needed_t)
        # Don't freeze splats that haven't had their optimization dwell yet.
        age_ok = (self.time - active._add_tick.squeeze(-1)) > self._commit_min_age
        mask = left & age_ok
        k = int(mask.sum())
        if k > 0:
            self.gaussians_upgrade(mask=mask)
            self._committed_on_leave += k

    @torch.no_grad()
    def _enforce_active_cap(self, frame):
        """Hard ceiling on the active (Adam) pool: force-graduate the most-converged surplus
        active splats to stable, so peak VRAM stays bounded in large environments even when
        location-based commit-on-leave lags."""
        cap = self._submap_active_max
        active = self.active_gaussians
        n = int(active.get_points_num)
        if cap <= 0 or n <= cap:
            return
        over = n - cap
        age = self.time - active._add_tick.squeeze(-1)
        conf = active.get_confidence.squeeze(-1)
        # Prefer graduating splats past their optimization dwell (age) and most converged
        # (confidence); fall back to raw confidence if too few are age-eligible (cap is hard).
        eligible = age > self._commit_min_age
        score = torch.where(eligible, conf, torch.full_like(conf, -1e9))
        if int(eligible.sum()) < over:
            score = conf
        idx = torch.topk(score, over, largest=True).indices
        mask = torch.zeros(n, dtype=torch.bool, device=conf.device)
        mask[idx] = True
        self.gaussians_upgrade(mask=mask)
        self._forced_grad += over

    def gaussians_upgrade(self, mask=None):
        if mask is None:
            confidence_mask = (
                self.active_gaussians.get_confidence > self.stable_confidence_thres
            ).squeeze()
            stable_mask = confidence_mask
        else:
            stable_mask = mask.squeeze()
        if self.verbose:
            logging.info("===== Points Fix =====")
            logging.info(f"Fix Gaussian Num: {stable_mask.sum():d}")
        if stable_mask.sum() > 0:
            stable_params = self.active_gaussians.remove(stable_mask)
            stable_params["confidence"] = torch.clip(
                stable_params["confidence"], max=self.stable_confidence_thres
            )
            self.stable_gaussians.cat(stable_params)

    def gaussians_downgrade(self, mask):
        if mask.sum() > 0:
            unstable_params = self.stable_gaussians.remove(mask)
            unstable_params["confidence"] = devF(torch.zeros_like(unstable_params["confidence"]))
            unstable_params["add_tick"] = self.time * devF(
                torch.ones_like(unstable_params["add_tick"])
            )
            self.active_gaussians.cat(unstable_params)

    def gaussians_delete_outliers(self, unstable=True):
        # Remove too small/big gaussians, long time unstable gaussians, insolated_gaussians
        if unstable:
            pointcloud = self.active_gaussians
        else:
            pointcloud = self.stable_gaussians
        if pointcloud.get_points_num == 0:
            return
        big_gaussian_mask = (pointcloud.get_radius > (pointcloud.get_radius.mean() * 10)).squeeze()
        unstable_time_mask = (
            (self.time - pointcloud.get_add_tick) > self.unstable_time_window
        ).squeeze()
        if unstable:
            delete_mask = big_gaussian_mask | unstable_time_mask
        else:
            # Don't size-prune stable: mean radius drops as the scan grows, which would
            # make earlier-area Gaussians (seeded near max_radius) look like outliers. Bad
            # geometry is still caught by gaussians_delete_mismatches() via error counters.
            delete_mask = torch.zeros_like(big_gaussian_mask)
        if self.verbose:
            logging.info(
                f"Big: {big_gaussian_mask.sum()}, Unstable: {unstable_time_mask.sum()}, Deleted: {delete_mask.sum()}"
            )
        removed_ids = torch.unique(pointcloud.get_semantic[delete_mask]).long().cpu().tolist()
        self.update_object_geometry(removed_ids)
        pointcloud.delete(delete_mask)

    def gaussians_delete_mismatches(self):
        if self.get_stable_num <= 0:
            return
        # check error by backprojection
        check_frame = self.processed_frames[-1]
        check_map = self.processed_map[-1]
        # Cull off-view stable gaussians. The per-pixel error counters are then
        # accumulated over the visible subset, and vis_idx maps those back to the full
        # stable rows. Off-view gaussians cover no pixel, so they get no error either way.
        if self.frustum_cull_stable:
            stable_c, vis_idx = self._cull_stable(check_frame, self.stable_params)
            params = self._merge_global_params(self.unstable_params, stable_c)
        else:
            vis_idx, params = None, self.global_params
        with torch.no_grad():
            render_output = self.renderer.render(check_frame, params)
        # [unstable, stable]
        unstable_points_num = self.get_unstable_num
        stable_points_num = vis_idx.numel() if vis_idx is not None else self.get_stable_num

        color = render_output["render"].permute(1, 2, 0)
        depth = render_output["depth"].permute(1, 2, 0)
        render_output["normal"].permute(1, 2, 0)
        depth_index = render_output["depth_index_map"].permute(1, 2, 0)
        color_index = render_output["color_index_map"].permute(1, 2, 0)

        depth_error = torch.abs(check_map["depth_map"] - depth)
        depth_error[(check_map["depth_map"] - depth) < 0] = 0
        image_error = torch.abs(check_map["color_map"] - color)
        color_error = torch.sum(image_error, dim=-1, keepdim=True)

        normal_error = devF(torch.zeros_like(depth_error))
        invalid_mask = (check_map["depth_map"] == 0) | (depth_index == -1)
        invalid_mask = invalid_mask.squeeze()

        depth_error[invalid_mask] = 0
        color_error[check_map["depth_map"] == 0] = 0
        normal_error[invalid_mask] = 0
        H, W = self.frame_map["color_map"].shape[:2]
        P = unstable_points_num + stable_points_num
        (
            gaussian_color_error,
            gaussian_depth_error,
            gaussian_normal_error,
            outlier_count,
        ) = accumulate_gaussian_error(
            H,
            W,
            P,
            color_error,
            depth_error,
            normal_error,
            color_index,
            depth_index,
            self.add_color_thres,
            self.add_depth_thres,
            self.add_normal_thres,
            True,
        )

        color_filter_thres = 2 * self.add_color_thres
        depth_filter_thres = 2 * self.add_depth_thres

        depth_delete_mask = (gaussian_depth_error > depth_filter_thres).squeeze()
        color_release_mask = (gaussian_color_error > color_filter_thres).squeeze()
        if self.verbose:
            logging.info("===== Outlier Remove =====")
            logging.info(
                f"Color Outlier Num: {color_release_mask.sum()}, "
                f"Depth Outlier Num: {depth_delete_mask.sum()}"
            )

        depth_delete_mask_stable = depth_delete_mask[unstable_points_num:, ...]
        color_release_mask_stable = color_release_mask[unstable_points_num:, ...]

        if vis_idx is not None:
            # masks index the visible subset; map back to full stable rows.
            self.stable_gaussians._depth_error_counter[vis_idx[depth_delete_mask_stable]] += 1
            self.stable_gaussians._color_error_counter[vis_idx[color_release_mask_stable]] += 1
        else:
            self.stable_gaussians._depth_error_counter[depth_delete_mask_stable] += 1
            self.stable_gaussians._color_error_counter[color_release_mask_stable] += 1

        delete_thresh = 10
        depth_delete_mask = (self.stable_gaussians._depth_error_counter >= delete_thresh).squeeze()
        color_release_mask = (self.stable_gaussians._color_error_counter >= delete_thresh).squeeze()

        removed_ids = (
            torch.unique(self.stable_gaussians.get_semantic[depth_delete_mask])
            .long()
            .cpu()
            .tolist()
        )
        self.stable_gaussians.delete(depth_delete_mask)
        self.update_object_geometry(removed_ids)
        self.gaussians_downgrade(color_release_mask[~depth_delete_mask])

    def check_keyframe(self, frame, frame_id):
        if self.time == 0:
            self.keyframe_list.append(frame.move_to_cpu_clone())
            self.keyframe_ids.append(frame_id)

            image_input = {
                "color_map": self.frame_map["color_map"].detach().cpu(),
                "depth_map": self.frame_map["depth_map"].detach().cpu(),
                "normal_map": self.frame_map["normal_map_w"].detach().cpu(),
                "objectness_map": (
                    self.frame_map["objectness_map"].detach().cpu()
                    if self.frame_map["objectness_map"] is not None
                    else None
                ),
            }
            self.keymap_list.append(image_input)
            return False

        prev_rot = self.keyframe_list[-1].R.T
        prev_trans = self.keyframe_list[-1].T
        curr_rot = frame.R.T
        curr_trans = frame.T

        # rot_compare returns (radians, degrees); compare in radians so
        # _kf_theta_thres_global (radians) routes correctly.
        theta_diff, theta_deg = rot_compare(prev_rot, curr_rot)
        _, l2_diff = trans_compare(prev_trans, curr_trans)

        if self.verbose:
            logging.info(f"rot diff: {theta_deg:.1f} deg, move diff: {l2_diff:.3f} m")

        # Uses the mapper-side thresholds so most frames take the local_optimize path
        # (active graduates) instead of global.
        if theta_diff > self._kf_theta_thres_global or l2_diff > self._kf_trans_thres_global:
            image_input = {
                "color_map": self.frame_map["color_map"].detach().cpu(),
                "depth_map": self.frame_map["depth_map"].detach().cpu(),
                "normal_map": self.frame_map["normal_map_w"].detach().cpu(),
                "objectness_map": (
                    self.frame_map["objectness_map"].detach().cpu()
                    if self.frame_map["objectness_map"] is not None
                    else None
                ),
            }
            self.keyframe_list.append(frame.move_to_cpu_clone())
            self.keymap_list.append(image_input)
            self.keyframe_ids.append(frame_id)
            return True
        return False

    def loss_update(
        self, render_output, image_input, init_stat, update_args, render_mask=None, unstable=True
    ):
        if self.loss_function_version == 1:
            return loss_update_v1(
                self, render_output, image_input, init_stat, update_args, render_mask, unstable
            )
        elif self.loss_function_version == 2:
            return loss_update_v2(
                self, render_output, image_input, init_stat, update_args, render_mask, unstable
            )

    @torch.no_grad()
    def evaluate_render_range(
        self, frame, global_opt=False, sample_ratio=-1, unstable=True, stable_mask=None
    ):
        if unstable:
            render_output = self.renderer.render(frame, self.unstable_params)
        else:
            stable_params = self.stable_params
            if stable_mask is not None:
                stable_params = {k: v[stable_mask] for k, v in stable_params.items()}
            render_output = self.renderer.render(frame, stable_params)
        unstable_T_map = render_output["T_map"]

        if global_opt:
            if sample_ratio > 0:
                render_image = render_output["render"].permute(1, 2, 0)
                gt_image = frame.original_image.permute(1, 2, 0).cuda()
                image_diff = (render_image - gt_image).abs()
                color_error = torch.sum(image_diff, dim=-1, keepdim=False)
                filter_mask = render_image.sum(dim=-1) == 0
                color_error[filter_mask] = 0
                tile_mask = colorerror2tilemask(color_error, 16, sample_ratio)
                render_mask = (
                    F.interpolate(
                        tile_mask.float().unsqueeze(0).unsqueeze(0),
                        scale_factor=16,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .bool()
                )[: color_error.shape[0], : color_error.shape[1]]
            # after training, real global optimization
            else:
                render_mask = (unstable_T_map != 1).squeeze(0)
                tile_mask = None
        else:
            render_mask = (unstable_T_map != 1).squeeze(0)
            tile_mask = transmission2tilemask(render_mask, 16, 0.5)

        render_ratio = render_mask.sum() / self.get_pixel_num
        return render_mask, tile_mask, render_ratio

    def global_optimization(
        self, update_args, select_keyframe_num=-1, is_end=False, keyframe_indices=None
    ):
        """Update the stable gaussians against a set of keyframes."""
        logging.info("===== Global Optimize =====")
        # Targeted pass (per-cell final optimize): optimize the resident stable cloud
        # against an explicit keyframe set. Windowed (select_keyframe_num != -1) so it
        # freezes xyz, keeps the cloud resident, and skips the full-residency setup below.
        if keyframe_indices is not None:
            select_keyframe_num = len(keyframe_indices)
            if select_keyframe_num == 0:
                return
        # The final pass sweeps all keyframes (their visible union ~ the whole map), so it
        # needs every cell resident. The periodic pass uses only recent keyframes, whose
        # cells are kept resident by the residency radius.
        if select_keyframe_num == -1:
            if not self._final_pass_should_run():
                if self._submap_final_pass == "per_cell":
                    self._per_cell_final_optimization(update_args)
                    return
                logging.warning(
                    "[SUBMAP] final joint global optimization SKIPPED: the full map "
                    "exceeds the GPU budget (%d cells out-of-core). The map stays "
                    "coherent from the per-frame + windowed periodic passes and is "
                    "exported in full via streaming; set submap_final_pass: per_cell for a "
                    "VRAM-bounded per-cell refinement.",
                    len(self.submaps._evicted),
                )
                return
            self.ensure_full_residency()
        if select_keyframe_num == -1:
            self.gaussians_upgrade(mask=(self.active_gaussians.get_confidence > -1))
        logging.info(
            f"Keyframes: {self.get_keyframe_num:d}, Stable Gaussians {self.get_stable_num:d}"
        )
        if self.get_stable_num == 0:
            return
        param_groups = self.stable_gaussians.parametrize(update_args)
        # Window the global pass to the stable Gaussians each keyframe actually sees. xyz
        # is frozen here (position_lr forced to 0 below), so a per-keyframe frustum mask
        # stays valid across all iterations. Only the render input is culled; the optimizer
        # keeps the full leaf tensors, so the attach-loss / init_stat alignment and Adam
        # state are untouched, and the rendered image is identical (off-view stable splats
        # cover no pixel). Applied to the final pass too: peak VRAM is per-render (one
        # keyframe at a time), keeping the whole-map final optimize within the GPU budget.
        cull_stable = self.frustum_cull_stable
        stable_xyz = self.stable_gaussians._xyz.detach() if cull_stable else None
        if select_keyframe_num != -1:
            param_groups[0]["lr"] = 0
            for i in range(1, len(param_groups)):
                param_groups[i]["lr"] *= 0.1
        else:
            param_groups[0]["lr"] = 0.0000
            param_groups[1]["lr"] *= self.feature_lr_coef
            param_groups[2]["lr"] *= self.feature_lr_coef
            param_groups[4]["lr"] *= self.scaling_lr_coef
            param_groups[5]["lr"] *= self.rotation_lr_coef
        is_final = False
        init_stat = {
            "opacity": self.stable_gaussians._opacity.detach().clone(),
            "scaling": self.stable_gaussians._scaling.detach().clone(),
            "xyz": self.stable_gaussians._xyz.detach().clone(),
            "rotation_raw": self.stable_gaussians._rotation.detach().clone(),
        }
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        total_iter = int(self.gaussian_update_iter)
        sample_ratio = 0.4
        if select_keyframe_num == -1:
            total_iter = int(self.final_global_iter)
            is_final = True
            select_keyframe_num = self.get_keyframe_num
            update_args.depth_weight = 0
            sample_ratio = -1

        random_kframes = False

        select_keyframe_num = min(select_keyframe_num, self.get_keyframe_num)
        if keyframe_indices is not None:
            # explicit keyframe_list indices (per-cell pass); used as-is, no recency remap
            select_kframe_indexs = list(keyframe_indices)
        elif random_kframes:
            if select_keyframe_num >= self.get_keyframe_num:
                select_kframe_indexs = list(range(0, self.get_keyframe_num))
            else:
                select_kframe_indexs = np.random.choice(
                    np.arange(1, min(select_keyframe_num * 2, self.get_keyframe_num)),
                    select_keyframe_num - 1,
                    replace=False,
                ).tolist() + [0]
            select_kframe_indexs = [i * -1 - 1 for i in select_kframe_indexs]
        else:
            select_kframe_indexs = [i * -1 - 1 for i in range(select_keyframe_num)]

        select_frame = []
        select_map = []
        select_render_mask = []
        select_tile_mask = []
        select_stable_mask = []
        select_id = []
        for index in select_kframe_indexs:
            move_to_gpu(self.keyframe_list[index])
            move_to_gpu_map(self.keymap_list[index])
            select_frame.append(self.keyframe_list[index])
            select_map.append(self.keymap_list[index])
            kf_stable_mask = None
            if cull_stable:
                kf_stable_mask = self._stable_frustum_mask(self.keyframe_list[index], stable_xyz)
                if not kf_stable_mask.any():
                    kf_stable_mask = None  # nothing visible: render full (empty either way)
            select_stable_mask.append(kf_stable_mask)
            render_mask, tile_mask, _ = self.evaluate_render_range(
                self.keyframe_list[index],
                global_opt=True,
                unstable=False,
                sample_ratio=sample_ratio,
                stable_mask=kf_stable_mask,
            )
            if select_keyframe_num == -1:
                move_to_cpu(self.keyframe_list[index])
                move_to_cpu_map(self.keymap_list[index])
            select_render_mask.append(render_mask)
            select_tile_mask.append(tile_mask)
            select_id.append(self.keyframe_ids[index])
            if self.verbose:
                if tile_mask is not None:
                    tile_mask.sum() / torch.numel(tile_mask)

        with tqdm(total=total_iter, desc="Global Optimization", disable=not self.verbose) as pbar:
            for iter in range(total_iter):
                self.iter = iter
                random_index = random.randint(0, select_keyframe_num - 1)
                # Second half of a windowed pass pins the oldest keyframe in the window.
                # Decided before any per-keyframe data is selected so the camera, target
                # maps and all masks stay paired to the same keyframe.
                if not random_kframes and iter > total_iter / 2 and not is_final:
                    random_index = -1
                frame_input = select_frame[random_index]
                image_input = select_map[random_index]
                frame_stable_mask = select_stable_mask[random_index] if cull_stable else None
                if select_keyframe_num == -1:
                    move_to_gpu(frame_input)
                    move_to_gpu_map(image_input)
                current_tile_mask = select_tile_mask[random_index]
                # If a tile mask exists but contains no active tiles, treat it as if there's no mask.
                # This prevents a crash when a keyframe sees no stable points.
                if current_tile_mask is not None and current_tile_mask.sum() == 0:
                    current_tile_mask = None
                stable_params = self.stable_params
                if frame_stable_mask is not None:
                    stable_params = {k: v[frame_stable_mask] for k, v in stable_params.items()}
                render_ouput = self.renderer.render(
                    frame_input,
                    stable_params,
                    tile_mask=current_tile_mask,
                )
                loss, reported_losses = self.loss_update(
                    render_ouput,
                    image_input,
                    init_stat,
                    update_args,
                    render_mask=select_render_mask[random_index],
                    unstable=False,
                )
                if select_keyframe_num == -1:
                    move_to_cpu(frame_input)
                    move_to_cpu_map(image_input)
                pbar.set_postfix({"loss": f"{loss:1.5f}"})
                pbar.update(1)

        for index in select_kframe_indexs:
            move_to_cpu(self.keyframe_list[index])
            move_to_cpu_map(self.keymap_list[index])
        self.stable_gaussians.detach()

    @torch.no_grad()
    def _keyframes_observing(self, xyz, sample=2048):
        """Indices into keyframe_list whose frustum contains any of `xyz` (a cell's splats);
        xyz is subsampled for speed."""
        n_kf = self.get_keyframe_num
        if n_kf == 0 or xyz.shape[0] == 0:
            return []
        if xyz.shape[0] > sample:
            xyz = xyz[torch.randperm(xyz.shape[0], device=xyz.device)[:sample]]
        return [
            i for i in range(n_kf) if self._stable_frustum_mask(self.keyframe_list[i], xyz).any()
        ]

    def _per_cell_final_optimization(self, update_args):
        """Final appearance refinement when the whole map exceeds the GPU budget: optimize
        the stable map one cell at a time, so peak VRAM stays bounded by a single cell.
        Positions are frozen (the windowed pass forces xyz lr 0) so cells never drift apart,
        and the photometric loss is masked to covered pixels (T_map != 1) so a resident cell
        is never pushed to explain evicted-region pixels. Each cell is refined against
        exactly the keyframes that observe it. Opt-in via submap_final_pass: per_cell."""
        sm = self.submaps
        if sm is None:
            return
        cells = sm.all_cell_ids(self)
        if not cells:
            return
        logging.info(
            "[SUBMAP] per-cell final optimize: %d cells, VRAM-bounded, positions frozen",
            len(cells),
        )
        refined = 0
        for cid in cells:
            sm.set_resident_cells(self, [cid])
            if self.get_stable_num == 0:
                continue
            kf = self._keyframes_observing(self.stable_gaussians.get_xyz)
            if not kf:
                continue
            self.global_optimization(update_args, keyframe_indices=kf)
            refined += 1
        logging.info(
            "[SUBMAP] per-cell final optimize done: %d/%d cells refined", refined, len(cells)
        )

    # Sample some pixels as the init gaussians
    def temp_points_init(self, frame: Camera):

        # Blur the objectness map once here and reuse it across every sampling call this
        # frame. Only v2 with gamma>0 uses objectness.
        objectness_map = self.frame_map["objectness_map"]
        smoothed_objectness = None
        if objectness_map is not None and self.sampling_version == 2 and self.sample_gamma > 0:
            smoothed_objectness = smooth_objectness(
                objectness_map, self.frame_map["vertex_map_w"].device
            )

        # First frame: sample uniformly from valid depth pixels
        if self.time == 0:
            depth_range_mask = self.frame_map["depth_map"] > 0

            xyz, normal, color, semantic = self.sampling_function(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                self.uniform_sample_num,
                depth_range_mask,
                semantic_map=self.frame_map["semantic_map"],
                objectness_map=objectness_map,
                is_error_sampling=False,
                name="init",
                smoothed_objectness=smoothed_objectness,
            )

            self.seed_gaussians.add_empty_points(xyz, normal, color, self.time, semantic=semantic)
            return

        # For later frames
        self.get_render_output(frame)

        # ===== Transmission-based sampling =====
        transmission_mask = (
            self.model_map["render_transmission"] > self.add_transmission_thres
        ) & (self.frame_map["depth_map"] > 0)

        transmission_ratio = transmission_mask.sum() / self.get_pixel_num
        self._last_trans_ratio = float(transmission_ratio)
        transmission_sample_num = devI(
            self.transmission_sample_ratio * transmission_ratio * self.uniform_sample_num
        )

        if self.verbose:
            logging.info(
                f"transmission empty num = {transmission_mask.sum():d}, sample num = {transmission_sample_num:d}"
            )

        xyz_trans, normal_trans, color_trans, semantic_trans = self.sampling_function(
            self.frame_map["vertex_map_w"],
            self.frame_map["normal_map_w"],
            self.frame_map["color_map"],
            transmission_sample_num,
            transmission_mask,
            semantic_map=self.frame_map["semantic_map"],
            objectness_map=objectness_map,
            is_error_sampling=False,
            name="trans",
            smoothed_objectness=smoothed_objectness,
        )
        self.seed_gaussians.add_empty_points(
            xyz_trans, normal_trans, color_trans, self.time, semantic=semantic_trans
        )

        # ===== Error-based sampling (depth + color) =====
        depth_error = torch.abs(self.frame_map["depth_map"] - self.model_map["render_depth"])

        color_error = torch.abs(self.frame_map["color_map"] - self.model_map["render_color"]).mean(
            dim=-1, keepdim=True
        )

        depth_error_mask = (
            (depth_error > self.add_depth_thres)
            & (self.frame_map["depth_map"] > 0)
            & (self.model_map["render_depth_index"] > -1)
        )

        color_error_mask = (
            (color_error > self.add_color_thres)
            & (self.frame_map["depth_map"] > 0)
            & (self.model_map["render_transmission"] < self.add_transmission_thres)
        )

        # Combine masks and exclude transmission samples
        error_mask = (depth_error_mask | color_error_mask) & (~transmission_mask)
        sample_num = devI(error_mask.sum() * self.error_sample_ratio)

        if self.verbose:
            logging.info(
                f"wrong depth num = {depth_error_mask.sum():d}, wrong color num = {color_error_mask.sum():d}, sample num = {sample_num:d}"
            )

        xyz_err, normal_err, color_err, semantic_err = self.sampling_function(
            self.frame_map["vertex_map_w"],
            self.frame_map["normal_map_w"],
            self.frame_map["color_map"],
            sample_num,
            error_mask,
            semantic_map=self.frame_map["semantic_map"],
            objectness_map=objectness_map,
            is_error_sampling=True,
            name="error",
            smoothed_objectness=smoothed_objectness,
        )
        self.seed_gaussians.add_empty_points(
            xyz_err, normal_err, color_err, self.time, semantic=semantic_err
        )

    # Remove temp points that fall within the existing unstable Gaussian.
    def temp_points_filter(self, topk=3):
        if self.get_unstable_num >= topk:
            temp_xyz = self.seed_gaussians.get_xyz
            if self.verbose:
                logging.info(f"init {self.seed_gaussians.get_points_num} temp points")
            unstable_params = self.unstable_params
            exist_xyz = unstable_params["xyz"]
            exist_raidus = unstable_params["radius"]

            inbbox_mask = bbox_filter(temp_xyz, exist_xyz)
            exist_xyz = exist_xyz[inbbox_mask]
            exist_raidus = exist_raidus[inbbox_mask]

            if exist_xyz.shape[0] == 0 or temp_xyz.shape[0] == 0:
                if self.verbose:
                    logging.info("No overlapping points found for KNN. Skipping filter.")
                return

            if self.seed_filter_mode == "grid":
                # Voxel-hash fixed-radius test, O(N+M) vs the kNN's O(N*M). Tests every
                # in-range gaussian, not just the 3 nearest. Background seeds are excluded
                # over the full footprint of the enlarged wall splats (BG_SCALE_MULT);
                # object seeds keep the tight cap so object edges next to walls stay dense.
                temp_sem = self.seed_gaussians._semantic
                if temp_sem is not None and temp_sem.shape[0] == temp_xyz.shape[0]:
                    is_bg = temp_sem.view(-1) <= 0
                else:
                    is_bg = torch.zeros(temp_xyz.shape[0], dtype=torch.bool, device=temp_xyz.device)
                inside_mask = torch.zeros_like(is_bg)
                cap_obj = self.max_radius
                cap_bg = gaussian_pointcloud.BG_SCALE_MULT * self.max_radius
                if (~is_bg).any():
                    inside_mask[~is_bg] = radius_inside_mask(
                        temp_xyz[~is_bg], exist_xyz,
                        torch.clamp(exist_raidus * 0.6, max=cap_obj), max_radius=cap_obj,
                    )
                if is_bg.any():
                    inside_mask[is_bg] = radius_inside_mask(
                        temp_xyz[is_bg], exist_xyz,
                        torch.clamp(exist_raidus * 0.6, max=cap_bg), max_radius=cap_bg,
                    )
            else:
                # Lazy: only the legacy seed_filter_mode "knn" needs pytorch3d, which has
                # no aarch64 wheel and is an expensive source build on Jetson.
                from pytorch3d.ops import knn_points

                nn_dist, nn_indices, _ = knn_points(
                    temp_xyz[None, ...], exist_xyz[None, ...], norm=2, K=topk, return_nn=True
                )
                nn_dist = torch.sqrt(nn_dist).squeeze(0)
                nn_indices = nn_indices.squeeze(0)

                corr_radius = exist_raidus[nn_indices] * 0.6
                inside_mask = (nn_dist < corr_radius).any(dim=-1)
            if self.verbose:
                logging.info(f"delete {inside_mask.sum().item()} temp points")
            self.seed_gaussians.delete(inside_mask)

    # Attach seed gaussians that fall within stable gaussians; attached ones get low opacity.
    def temp_points_attach(self, frame: Camera, unstable_opacity_low=0.1):
        if self.get_stable_num == 0:
            return

        # project unstable gaussians and compute uv
        unstable_xyz = self.seed_gaussians.get_xyz
        origin_indices = torch.arange(unstable_xyz.shape[0]).cuda().long()
        unstable_opacity = self.seed_gaussians.get_opacity
        unstable_opacity_filter = (unstable_opacity > unstable_opacity_low).squeeze(-1)
        unstable_xyz = unstable_xyz[unstable_opacity_filter]

        unstable_uv = frame.get_uv(unstable_xyz)
        indices = torch.arange(unstable_xyz.shape[0]).cuda().long()
        unstable_mask = (
            (unstable_uv[:, 0] >= 0)
            & (unstable_uv[:, 0] < frame.image_width)
            & (unstable_uv[:, 1] >= 0)
            & (unstable_uv[:, 1] < frame.image_height)
        )
        project_uv = unstable_uv[unstable_mask]

        # get the corresponding stable gaussians (culled to this frame's frustum;
        # stable_vis_idx maps the rendered index back to the full stable rows below)
        stable_c, stable_vis_idx = self._cull_stable(frame, self.stable_params)
        if stable_vis_idx is not None and stable_vis_idx.numel() == 0:
            return
        try:
            with torch.no_grad():
                stable_render_output = self.renderer.render(frame, stable_c)
        except Exception as e:
            print(f"STABLE PARAM NUMS: {self.stable_params.get('xyz').numel()}")
            export_frame_and_params(frame, self.stable_params)
            print("-----------------------------------------")
            print("-----------------------------------------")
            print("--------- FILES SAVED FOR DEBUG ---------")
            print("-----------------------------------------")
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

        stable_index = stable_render_output["color_index_map"].permute(1, 2, 0)
        intersect_mask = stable_index[project_uv[:, 1], project_uv[:, 0]] >= 0
        indices = indices[unstable_mask][intersect_mask[:, 0]]

        # Check if there are any intersections with rendered stable points.
        if indices.shape[0] == 0:
            return

        # compute point to plane distance
        intersect_stable_index = (
            (stable_index[unstable_uv[indices, 1], unstable_uv[indices, 0]]).squeeze(-1).long()
        )
        if stable_vis_idx is not None:
            intersect_stable_index = stable_vis_idx[intersect_stable_index]

        stable_normal_check = self.stable_gaussians.get_normal[intersect_stable_index]
        stable_xyz_check = self.stable_gaussians.get_xyz[intersect_stable_index]
        unstable_xyz_check = self.seed_gaussians.get_xyz[indices]
        point_to_plane_distance = (
            (stable_xyz_check - unstable_xyz_check) * stable_normal_check
        ).sum(dim=-1)
        intersect_check = point_to_plane_distance.abs() < 0.5 * self.add_depth_thres
        indices = indices[intersect_check]
        indices = origin_indices[unstable_opacity_filter][indices]

        # set opacity
        self.seed_gaussians._opacity[indices] = inverse_sigmoid(
            unstable_opacity_low * torch.ones_like(self.seed_gaussians._opacity[indices])
        )
        if self.verbose:
            logging.info(f"attach {indices.shape[0]} unstable gaussians")

    # Initialize seed points as unstable (active) gaussians.
    def temp_to_optimize(self):
        global_params = self.global_params
        self.seed_gaussians.update_geometry(global_params["xyz"], global_params["radius"])
        if self.verbose:
            logging.info("===== Points Add =====")
            logging.info(f"New gaussian num: {self.seed_gaussians.get_points_num:d}")
        remove_mask = devB(torch.ones(self.seed_gaussians.get_points_num))
        temp_params = self.seed_gaussians.remove(remove_mask)
        self._last_added_points = temp_params["xyz"].shape[0]
        self.active_gaussians.cat(temp_params)
        added_ids = torch.unique(temp_params["semantic"]).long().cpu().tolist()
        self.update_object_geometry(added_ids)

    def create_workspace(self):
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        logging.info(self.save_path)
        os.makedirs(self.save_path, exist_ok=True)
        render_save_path = os.path.join(self.save_path, "eval_render")
        os.makedirs(render_save_path, exist_ok=True)
        model_save_path = os.path.join(self.save_path, "save_model")
        os.makedirs(model_save_path, exist_ok=True)

        if self.processing_mode == "single" and self.use_tensorboard:
            self.tb_writer = SummaryWriter(self.save_path)
        else:
            self.tb_writer = None

    def _save_stable_per_cell(self, out_dir):
        """Write the per-cell viewer export (cells/cell_*.ply + cells.json) of the full
        stable map into `out_dir`. Uses the live SubmapManager when submapping is on,
        otherwise spins up a transient one to partition the fully-resident cloud the same
        way. Best-effort: a failure here never aborts the run (the monolithic PLY is the
        fallback)."""
        if self.stable_gaussians.get_points_num == 0 and (
            self.submaps is None or self.submaps.n_evicted() == 0
        ):
            return
        try:
            sm = self.submaps if self.submaps is not None else SubmapManager(self.args)
            manifest = sm.save_stable_ply_per_cell(
                self.stable_gaussians, out_dir, include_confidence=True
            )
            cells_json = os.path.join(out_dir, "cells.json")
            self._last_cells_manifest = os.path.relpath(cells_json, self.save_path)
            logging.info(
                "[VIEWER] per-cell export: %d cells, %d verts -> %s",
                manifest["cell_count"],
                manifest["total_count"],
                self._last_cells_manifest,
            )
        except Exception as e:
            self._last_cells_manifest = None
            logging.warning("[VIEWER] per-cell export failed (%s); monolithic PLY remains", e)

    def save_model(self, path=None, save_data=True, save_sibr=True, save_merge=True):
        # When evicting, the stable PLY is streamed cell-by-cell below (out-of-core, peak
        # = one cell); otherwise page the whole map resident first.
        evicting = self._evicting()
        if not evicting:
            self.ensure_full_residency()  # PLYs must contain the whole map
        if path is None:
            frame_name = f"frame_{self.time:04d}"
            model_save_path = os.path.join(self.save_path, "save_model", frame_name)
            os.makedirs(model_save_path, exist_ok=True)
            path = os.path.join(model_save_path, f"iter_{self.iter:04d}")
        if save_data:
            self.active_gaussians.save_model_ply(
                path + ".ply",
                include_confidence=True,
            )
            self.stable_gaussians.save_model_ply(
                path + "_stable_sem.ply",
                include_confidence=True,
                semantics_only=True,
                scene_graph=self.scene_graph,
            )
            if evicting:
                self.submaps.save_full_stable_ply(
                    self.stable_gaussians,
                    path + "_stable.ply",
                    include_confidence=True,
                    include_anchor=True,
                )
            else:
                self.stable_gaussians.save_model_ply(
                    path + "_stable.ply",
                    include_confidence=True,
                    semantics_only=False,
                    visualize_semantics=False,
                    include_anchor=True,
                )
            # Canonical stable PLY for the run manifest (relative to save_path).
            self._last_stable_ply = os.path.relpath(path + "_stable.ply", self.save_path)
            # Per-cell export for the web viewer (fast load + distance LOD); the monolithic
            # PLY above is left untouched as the viewer's fallback.
            self._save_stable_per_cell(os.path.dirname(path + "_stable.ply"))
        if save_sibr:
            self.active_gaussians.save_model_ply(
                path + "_sibr.ply",
                include_confidence=False,
            )
            if evicting:
                self.submaps.save_full_stable_ply(
                    self.stable_gaussians, path + "_stable_sibr.ply", include_confidence=False
                )
            else:
                self.stable_gaussians.save_model_ply(
                    path + "_stable_sibr.ply",
                    include_confidence=False,
                )
        if self.get_unstable_num > 0 and self.get_stable_num > 0 and not evicting:
            if save_data and save_merge:
                merge_ply(
                    path + ".ply",
                    path + "_stable.ply",
                    path + "_merge.ply",
                    include_confidence=True,
                )
            if save_sibr and save_merge:
                merge_ply(
                    path + "_sibr.ply",
                    path + "_stable_sibr.ply",
                    path + "_merge_sibr.ply",
                    include_confidence=False,
                )
        elif evicting and save_merge:
            # merge_ply loads both PLYs into RAM; for an out-of-core map that defeats the
            # purpose, so the full stable map is left as its own streamed PLY.
            logging.info("[SUBMAP] merge PLY skipped (stable map streamed out-of-core)")

    def train_report(self, iteration, losses):
        if self.tb_writer is not None:
            for loss in losses:
                self.tb_writer.add_scalar(f"train/{loss}", losses[loss], iteration)

    @torch.no_grad()
    def get_render_output(self, frame):
        # Cull off-view stable gaussians: the rendered image/depth/transmission are
        # unchanged for visible pixels, and the only index consumer reads it as a validity
        # mask (render_depth_index > -1), so no index remap is needed.
        if self.frustum_cull_stable and self.get_stable_num > 0:
            stable_c, _ = self._cull_stable(frame, self.stable_params)
            params = self._merge_global_params(self.unstable_params, stable_c)
        else:
            params = self.global_params
        render_output = self.renderer.render(frame, params)
        self.model_map["render_color"] = render_output["render"].permute(1, 2, 0)
        self.model_map["render_depth"] = render_output["depth"].permute(1, 2, 0)
        self.model_map["render_normal"] = render_output["normal"].permute(1, 2, 0)
        self.model_map["render_color_index"] = render_output["color_index_map"].permute(1, 2, 0)
        self.model_map["render_depth_index"] = render_output["depth_index_map"].permute(1, 2, 0)
        self.model_map["render_transmission"] = render_output["T_map"].permute(1, 2, 0)

    @torch.no_grad()
    def segment_rooms(self):
        # Normals are recomputed from scales/rotations (get_normal).
        xyz, normals, opacities, scales, rotations = [], [], [], [], []
        for gaussians, count in (
            (self.active_gaussians, self.get_unstable_num),
            (self.stable_gaussians, self.get_stable_num),
        ):
            if count > 0:
                xyz.append(gaussians.get_xyz)
                normals.append(gaussians.get_normal)
                opacities.append(gaussians.get_opacity.reshape(-1))
                scales.append(gaussians.get_scaling)
                rotations.append(gaussians.get_rotation)
        if not xyz:
            return
        occupancy_output_dir = os.path.join(self.save_path, "occupancy")
        result = segment_rooms_from_arrays(
            torch.cat(xyz),
            torch.cat(normals),
            torch.cat(opacities),
            scales=torch.cat(scales),
            rotations=torch.cat(rotations),
            method=getattr(self.args, "room_seg_method", "ours"),
            vertical_axis=self.args.vertical_axis,
            output_dir=occupancy_output_dir,
        )
        self.scene_graph.update_object_rooms(result.rooms, storeys=result.storeys)
        export_object_room_map(
            self.scene_graph, save_path=os.path.join(occupancy_output_dir, "object_rooms.png")
        )

    def export_all(self, full=True):
        sg_json = f"scene_graph/objects_{self.time + 1}.json"
        self.scene_graph.export_objects(os.path.join(self.save_path, sg_json))
        # Real basenames (index.index / index.pkl), not empty-prefix dotfiles that
        # rsync/cp/zip silently drop; the manifest records the prefix.
        vdb_prefix = "scene_graph/vector_db/index"
        self.scene_graph.db.save(os.path.join(self.save_path, vdb_prefix))
        # OpenLex3D export walks the whole stable cloud; skip it mid-run while evicting
        # (it would only see the resident set) — finalize runs it on the full map.
        # Also skipped when there are no objects (a run with zero detections).
        if full:
            if len(self.scene_graph.all_objects) > 0:
                export_for_openlex3d(
                    self.stable_gaussians,
                    self.scene_graph,
                    output_dir=os.path.join(self.save_path, "openlex3d"),
                )
            else:
                logging.info("[EXPORT] no objects in scene graph — skipping OpenLex3D export")
            if torch.cuda.is_available():
                print(
                    f"[MAIN] peak CUDA memory: "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB"
                )
        self.write_run_manifest(scene_graph_json=sg_json, vector_db_prefix=vdb_prefix)

    def write_run_manifest(self, scene_graph_json=None, vector_db_prefix=None):
        """Write <save_path>/manifest.json: one source of truth for what the run produced
        and where, so loaders (e.g. the dashboard) resolve explicit paths instead of globs.
        Written last and atomically, so its presence marks a complete, consistent run."""
        try:
            dim = int(self.scene_graph.db.dim)
        except Exception:
            dim = None
        manifest = {
            "scene": os.path.basename(os.path.normpath(self.save_path)),
            "up_axis": int(getattr(self.args, "vertical_axis", 2)),
            "time": int(self.time),
            "iteration": int(getattr(self, "iter", -1)),
            "clip_model": getattr(self.args, "clip_model", None),
            "vector_db_dim": dim,
            "stable_ply": getattr(self, "_last_stable_ply", None),
            "cells_manifest": getattr(self, "_last_cells_manifest", None),
            "scene_graph_json": scene_graph_json,
            "vector_db_prefix": vector_db_prefix,
            "openlex3d_dir": "openlex3d",
            "config": "config.yaml",
        }
        # Intrinsics of the mapped stream (ROS runs get them from camera_info, so the
        # saved config alone can't reconstruct them). Consumed by gssg-render.
        if self.keyframe_list:
            from gssg.utils.graphics_utils import fov2focal

            cam = self.keyframe_list[-1]
            w, h = int(cam.image_width), int(cam.image_height)
            manifest["camera"] = {
                "width": w,
                "height": h,
                "fx": float(fov2focal(cam.FoVx, w)),
                "fy": float(fov2focal(cam.FoVy, h)),
                "cx": float(cam.cx) if float(cam.cx) > 0 else w / 2.0,
                "cy": float(cam.cy) if float(cam.cy) > 0 else h / 2.0,
            }
        tmp = os.path.join(self.save_path, "manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, os.path.join(self.save_path, "manifest.json"))

    _PARAM_KEYS = (
        "xyz",
        "opacity",
        "scales",
        "rotations",
        "shs",
        "radius",
        "normal",
        "confidence",
        "semantics",
    )

    @staticmethod
    def _build_params(cloud):
        have = cloud.get_points_num > 0
        return {
            "xyz": devF(cloud.get_xyz if have else torch.empty(0)),
            "opacity": devF(cloud.get_opacity if have else torch.empty(0)),
            "scales": devF(cloud.get_scaling if have else torch.empty(0)),
            "rotations": devF(cloud.get_rotation if have else torch.empty(0)),
            "shs": devF(cloud.get_features if have else torch.empty(0)),
            "radius": devF(cloud.get_radius if have else torch.empty(0)),
            "normal": devF(cloud.get_normal if have else torch.empty(0)),
            "confidence": devF(cloud.get_confidence if have else torch.empty(0)),
            "semantics": devF(cloud.get_semantic if have else torch.empty(0)),
        }

    def _cached_params(self, cloud, cache_attr):
        # The derived params (exp scaling, build_rotation+gather normal, SH cat) are
        # rebuilt over the whole cloud at every property access, so cache them keyed on the
        # cloud's mutation counter. While the cloud is parametrized the optimizer mutates
        # the leaf tensors in place without bumping the version, so serve fresh (and don't
        # cache) whenever grad is required.
        if cloud._xyz.requires_grad:
            return self._build_params(cloud)
        cached = getattr(self, cache_attr, None)
        if cached is not None and cached[0] == cloud._version:
            return cached[1]
        params = self._build_params(cloud)
        setattr(self, cache_attr, (cloud._version, params))
        return params

    @property
    def stable_params(self):
        return self._cached_params(self.stable_gaussians, "_stable_params_cache")

    @property
    def unstable_params(self):
        return self._cached_params(self.active_gaussians, "_unstable_params_cache")

    @property
    def global_params_detach(self):
        unstable_params = self.unstable_params
        stable_params = self.stable_params
        return {
            k: devF(torch.cat([unstable_params[k].detach(), stable_params[k].detach()]))
            for k in self._PARAM_KEYS
        }

    @property
    def global_params(self):
        unstable_params = self.unstable_params
        stable_params = self.stable_params
        return {
            k: devF(torch.cat([unstable_params[k], stable_params[k]])) for k in self._PARAM_KEYS
        }

    @property
    def get_pixel_num(self):
        return self.frame_map["depth_map"].shape[0] * self.frame_map["depth_map"].shape[1]

    @property
    def get_total_iter(self):
        return self.iter + self.time * self.gaussian_update_iter

    @property
    def get_stable_num(self):
        return self.stable_gaussians.get_points_num

    @property
    def get_unstable_num(self):
        return self.active_gaussians.get_points_num

    @property
    def get_total_num(self):
        return self.get_stable_num + self.get_unstable_num

    @property
    def get_keyframe_num(self):
        return len(self.keyframe_list)


class MappingProcess(Mapping):
    def __init__(self, map_params, optimization_params, slam, scene_graph=None):
        self.processing_mode = "multi"
        self.map_params = map_params
        self.scene_graph_ref = scene_graph
        self.optimization_params = optimization_params
        self.save_path = map_params.save_path
        logging.info("finish init")
        self.create_workspace()

        self.slam = slam
        self._tracker2mapper_call = slam._tracker2mapper_call
        self._tracker2mapper_frame_queue = slam._tracker2mapper_frame_queue
        self._mapper2system_call = slam._mapper2system_call
        self._mapper2system_map_queue = slam._mapper2system_map_queue
        self._mapper2system_requires = slam._mapper2system_requires
        self._mapper2tracker_call = slam._mapper2tracker_call
        self._mapper2tracker_map_queue = slam._mapper2tracker_map_queue

        self._requests = [False, False]
        self._stop = False
        self.input = {}
        self.output = {}
        self.processed_tick = []
        self.time = 0
        self._end = slam._end
        self.max_frame_id = -1
        self.finish = mp.Event()

    def set_input(self):
        self.frame_map["depth_map"] = self.input["depth_map"]
        self.frame_map["color_map"] = self.input["color_map"]
        self.frame_map["normal_map_c"] = self.input["normal_map_c"]
        self.frame_map["normal_map_w"] = self.input["normal_map_w"]
        self.frame_map["vertex_map_c"] = self.input["vertex_map_c"]
        self.frame_map["vertex_map_w"] = self.input["vertex_map_w"]
        self.frame_map["semantic_map"] = self.input["semantic_map"]
        self.frame_map["semantic_results"] = self.input["semantic_results"]
        self.frame_map["objectness_map"] = self.input["objectness_map"]
        self.frame_map["frame_embedding"] = self.input["frame_embedding"]
        self.time = self.input["time"]
        self.last_send_time = -1
        # Tegra: the tracker CPU-staged these maps to cross the queue; move the
        # GPU-consumed ones back to CUDA (frame_embedding/semantic_results stay CPU by
        # design). No-op on dGPU.
        restore_frame_map_to_gpu(self.frame_map)

    def send_output(self):
        self.output = {
            "active_gaussians": self.active_gaussians,
            "stable_gaussians": self.stable_gaussians,
            "time": self.time,
            "iter": self.iter,
        }
        try:
            print(f"Sending final output at time: {self.time}")
        except Exception as e:
            print(e)
        # Ensure no grad graph survives into the IPC snapshot: deepcopy of a non-leaf
        # tensor (e.g. a cached _normal set while parametrized) raises.
        self.active_gaussians.detach()
        self.stable_gaussians.detach()
        # The clouds hold a back-ref to the full scene graph (FAISS index, R-tree, object
        # CLIP embeddings — some non-leaf tensors). The system process only writes PLYs
        # from the point data and has its own scene graph, so drop the back-ref before
        # deepcopy: it would otherwise drag the whole graph into the queue and crash.
        sg_a, sg_s = self.active_gaussians.scene_graph, self.stable_gaussians.scene_graph
        self.active_gaussians.scene_graph = None
        self.stable_gaussians.scene_graph = None
        try:
            out = copy.deepcopy(self.output)
        finally:
            self.active_gaussians.scene_graph = sg_a
            self.stable_gaussians.scene_graph = sg_s
        # Tegra: CPU-stage the gaussians for the save queue (no CUDA IPC). No-op on dGPU.
        if NO_CUDA_IPC:
            out["active_gaussians"].to_("cpu")
            out["stable_gaussians"].to_("cpu")
        self._mapper2system_map_queue.put(out)
        self._mapper2system_requires[1] = True
        with self._mapper2system_call:
            self._mapper2system_call.notify()

    def pack_map_to_tracker(self, curr_frame):
        # Heartbeat only: the tracker consumes nothing but frame_id (it drives the
        # strict/loose sync).
        map_info = {"frame_id": self.processed_tick[-1]}
        logging.info(f"mapper send map {self.processed_tick[-1]} to tracker")
        with self._mapper2tracker_call:
            self._mapper2tracker_map_queue.put(map_info)
            self._mapper2tracker_call.notify()

    def run(self):
        logging.info("Mapper Process: Initializing Gaussians & Renderers...")
        super().__init__(self.map_params, scene_graph=self.scene_graph_ref)

        fps_meter = FPSMeter(name="MAPPER", log_every=getattr(self.map_params, "fps_log_every", 20))

        while True:
            with self._tracker2mapper_call:
                while self._tracker2mapper_frame_queue.empty():
                    logging.info("waiting tracker to wakeup")
                    self._tracker2mapper_call.wait(timeout=3.5)
                self.input = self._tracker2mapper_frame_queue.get()

            self.max_frame_id = max(self.max_frame_id, self.input["time"])

            if "time" in self.input and self.input["time"] == -1:
                del self.input
                break

            self.set_input()
            self.processed_tick.append(self.time)
            # Tegra: the frame crossed the queue on CPU; restore all its tensors to GPU
            # (not just image/depth). dGPU: move_to_gpu is enough (IPC kept it GPU).
            if NO_CUDA_IPC:
                frame_tensors_to(self.input["frame"], "cuda")
            else:
                move_to_gpu(self.input["frame"])

            _wd0 = time.perf_counter()
            with fps_meter:
                self.mapping(
                    self.input["frame"],
                    self.frame_map,
                    self.input["time"],
                    self.optimization_params,
                )
            if self._mapper_watchdog_ms:
                _wd_ms = (time.perf_counter() - _wd0) * 1000.0
                if _wd_ms > self._mapper_watchdog_ms:
                    print(
                        f"[WATCHDOG] frame {self.time}: mapping() {_wd_ms:.0f} ms "
                        f"(> {self._mapper_watchdog_ms:.0f} ms budget)",
                        flush=True,
                    )

            if getattr(self.args, "log_scaling_series", False):
                # per-frame sample for the scaling figure; mirrors gssg/run.py's hook so
                # single- and multi-process runs produce the same runtime_series.json
                resident = int(self.get_stable_num + self.get_unstable_num)
                total = resident
                if self.submaps is not None:
                    total = int(self.submaps.total_stable_count(resident))
                # memory_allocated counts only live tensors: it sawtooths as tensors
                # are freed and understates the process by 20-30x, because the caching
                # allocator keeps the pool (reserved) and the CUDA context is on top.
                # reserved ~ what this process shows in nvidia-smi; nvml_used_gb is the
                # whole GPU (mapper + perception + ZED), i.e. what actually OOMs.
                _scaling = self.runtime_stats.extra.setdefault("scaling", [])
                if len(_scaling) % 10 == 0:
                    # release cached-free blocks to the driver so nvml_used_gb tracks
                    # live usage instead of the caching allocator's high-water mark
                    torch.cuda.empty_cache()
                _scaling.append({
                    "frame": int(self.time),
                    "gaussians": resident,
                    "gaussians_total": total,
                    "vram_gb": torch.cuda.memory_allocated() / 2**30,
                    "vram_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "vram_reserved_gb": torch.cuda.memory_reserved() / 2**30,
                    "vram_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                    "nvml_used_gb": _nvml_used_gb(),
                })
                if int(self.time) % 25 == 0:
                    self.runtime_stats.write_series(self.save_path)

            if (self.time + 1) % self.save_step == 0:
                logging.info(f"Saving checkpoint at frame {self.time}")
                # Off the critical path: no full-residency paging (the evicted map stays
                # out-of-core and durable as blobs), bounded PLYs written on a worker
                # thread, and — when evicting — the whole-map room-seg / OpenLex3D export
                # deferred to finalize (mid-run they only see the resident set anyway).
                eval_frame(
                    self,
                    self.input["frame"],
                    os.path.join(self.save_path, "eval_render"),
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                    save_picture=True,
                    run_pcd=False,
                )
                self._background_checkpoint()
                if not self._evicting():
                    self.segment_rooms()
                self.export_all(full=not self._evicting())

            self.pack_map_to_tracker(self.input["frame"])

        self.global_optimization(self.optimization_params)
        logging.info("Saving final model...")
        self.await_checkpoint()  # let the last background PLY land before the authoritative save
        self.save_model(save_data=True)
        # Final room segmentation on the complete map. With eviction, the per-save_step
        # segment_rooms only sees the resident set (-> missing/zero rooms), and save_step
        # may never fire on a short run. Page the whole map resident and segment once here
        # so the exported scene graph's rooms come from the full floor plan.
        self.ensure_full_residency()
        self.segment_rooms()
        self.export_all()

        self.time = -1
        self.send_output()

        logging.info(f"processed frames: {self.optimize_frames_ids}")
        logging.info(f"keyframes: {self.keyframe_ids}")

        self._end[1] = 1
        with self._mapper2system_call:
            self._mapper2system_call.notify()

        logging.info("mapper cleaning up...")
        time.sleep(1.0)

        logging.info("mapper wating finish")
        self.finish.wait()
        logging.info("map finish")

    def stop(self):
        self.finish.set()
