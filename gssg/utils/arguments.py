#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t is bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t is bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

    def extract_dict(self, config):
        group = GroupParams()
        for k, v in config.items():
            if k in vars(self) or ("_" + k) in vars(self):
                setattr(group, k, v)
        return group


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.position_lr = 0.0016
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        self.color_weight = 0.8
        self.depth_weight = 1
        self.ssim_weight = 0.2
        self.history_weight = 0.1
        self.normal_weight = 0.1
        super().__init__(parser, "Optimization Parameters")


class DatasetParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._resolution = -1
        self.dataset_type = "replica"
        self.gpu_device = "cuda"
        self.eval = False
        self.init_mode = "random"
        self.frame_max = -1
        self.frame_start = 0
        self.frame_step = 0
        self.eval_llff = 8
        self.sh_degree = 3
        self.preload = False
        self.resolution_scales = [1.0]
        self.use_keyframe = False
        self.keyframe_theta_thres = 0.1
        self.keyframe_trans_thres = 0.05
        # Mapper-side global-optimization selection thresholds, decoupled from the perception
        # keyframe gate above. Larger => fewer global (stable-only) passes and more
        # local_optimize. 0 = fall back to keyframe_*_thres.
        self.keyframe_theta_thres_global = 0.0
        self.keyframe_trans_thres_global = 0.0
        # OpenGL->OpenCV camera Y/Z flip; False for optical-frame (ZED) captures.
        self.apply_view_transform = True
        # FFT blur filtering for offline frame selection (read from YAML, not cam_params.json).
        self.filter_blur = False
        self.blur_filter_radius = 5
        # --- ROS 1 live stream ---
        self.ros1_color_topic = "/camera/color/image_raw"
        self.ros1_depth_topic = "/camera/depth/image_rect_raw"
        self.ros1_world_frame = "world"
        self.ros1_camera_frame = "camera_link"
        # --- ROS 2 / Isaac cuVSLAM / ZED2i live stream ---
        # Only used when dataset_type == "ros2". Defaults match a stock
        # zed_wrapper + isaac_ros_visual_slam setup.
        self.ros2_color_topic = "/zed/zed_node/rgb/color/rect/image"
        self.ros2_depth_topic = "/zed/zed_node/depth/depth_registered"
        self.ros2_camera_info_topic = "/zed/zed_node/rgb/color/rect/camera_info"
        # ZED PT publishes its pose directly on /zed/zed_node/odom (Odometry,
        # in `odom` frame, with 6x6 covariance). We subscribe with
        # ApproximateTimeSynchronizer alongside color+depth so each frame's
        # pose is the exact pose ZED computed for that frame's timestamp — no
        # TF interpolation, no extrapolation, lowest possible pose latency.
        self.ros2_pose_topic = "/zed/zed_node/odom"
        # World frame = the odom-msg's header.frame_id; "odom" for live mapping. For a
        # globally-consistent, drift-free pose use ZED Area Memory localization (see docs).
        self.ros2_world_frame = "odom"
        self.ros2_camera_frame = "zed_left_camera_frame_optical"
        # Localization-confidence gate: skip frames whose ZED odom covariance trace exceeds
        # these (poison the map otherwise). Position (m^2): cov[0]+cov[7]+cov[14]; ZED reports
        # ~1e-5 tracking well, >1e-2 = LOST. Rotational (rad^2): cov[21]+cov[28]+cov[35], spin
        # uncertainty on fast turns. 0.0 = accept all.
        self.ros2_pose_cov_max = 0.01
        self.ros2_pose_rot_cov_max = 0.0
        # ZED tracking-status gate: drop frames while /pose/status reports broken VIO
        # (odometry NOT_OK) or area-map relocalization (spatial SEARCHING). Inert when
        # the topic is silent or zed_msgs is absent (non-ZED sources).
        self.ros2_gate_pose_status = True
        self.ros2_pose_status_topic = "/zed/zed_node/pose/status"
        self.ros2_pose_status_max_age = 1.0  # s; older status = unknown = pass
        # Seconds without a new frame (after the first one) before the session is
        # considered ended. Raise for replays that pause mid-stream (chain replay,
        # heavy area-file relocalization stalls).
        self.ros2_idle_timeout = 8.0
        # External stop signal: publish an Empty message to this topic to ask
        # GSSG to wind down gracefully (e.g. from another terminal).
        #   ros2 topic pub --once /gssg/stop std_msgs/msg/Empty '{}'
        self.ros2_stop_topic = "/gssg/stop"
        # Online motion-blur gate (live ROS 2 streams): skip frames whose FFT
        # sharpness score is below this. 0.0 = off. Calibrate from the periodic
        # "[ROS2Stream] blur: score ..." log line (same metric as filter_blur).
        self.online_blur_thres = 0.0
        super().__init__(parser, "Dataset Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class MapParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.init_opacity = 0.999
        self.max_sh_degree = 4
        self.active_sh_degree = -1
        self.uniform_sample_num = 5000
        # Objectness sampling weights w = M_conf^gamma + beta (gamma=0 -> uniform)
        self.sample_gamma = 3.0
        self.sample_beta = 0.3
        # Background-splat radius multiplier; sampling_version 1 or gamma=0 forces 1.0 (off)
        self.bg_scale_mult = 1.5
        self.gaussian_update_iter = 300
        # Hard per-frame wall-clock cap (ms) on the local_optimize loop; 0 = off (bound only
        # by gaussian_update_iter). Set it on live configs for a guaranteed real-time budget.
        self.optimize_budget_ms = 0.0
        self.gaussian_update_frame = 1

        self.spatial_lr_scale = 1
        self.save_path = "output/slam_test"
        self.min_depth = 0
        self.max_depth = 0
        self.renderer_opaque_threshold = 0.7
        self.renderer_normal_threshold = 80
        self.renderer_depth_threshold = 1.0

        self.memory_length = 10
        self.frustum_cull_stable = True
        # Submapping (out-of-core eviction of the stable cloud). Off by default = monolithic
        # behaviour. When on, stable Gaussians in cells beyond submap_radius of the working-set
        # cameras are evicted (host/disk) and paged back on approach, bounding VRAM by the local
        # working set. Eviction itself is opt-in via submap_evict.
        self.submapping = False
        self.submap_cell_size = 4.0  # m, ground-plane cell edge
        self.submap_radius = 0.0  # m, residency radius; 0 = max_depth + cell_size
        self.submap_evict_tier = "host"  # "host" (pinned CPU) | "disk" (torch.save) | "hybrid"
        self.submap_page_dir = ""  # disk/hybrid page directory ("" -> a temp dir)
        # hybrid tier: spill least-recently-used host cells to disk once the RAM blobs
        # exceed this fraction of total system RAM.
        self.submap_host_ram_frac = 0.7
        # Out-of-core offline passes. Budget (GB) for paging the whole map back to GPU;
        # 0 = auto (80% of free VRAM). If the map exceeds it, export streams the PLY
        # cell-by-cell and the final joint global optimize is skipped (see submap_final_pass).
        self.submap_gpu_budget_gb = 0.0
        # "auto" (monolithic if it fits else skip) | "always" | "skip" | "per_cell"
        # (when over budget, refine one cell at a time instead of skipping — VRAM-bounded,
        # positions frozen so no cross-cell seams).
        self.submap_final_pass = "auto"
        # Eviction switch. When on (with submapping) stable cells beyond the residency radius
        # page out-of-core: object geometry/count are answered from SemanticReductionIndex so
        # evicted cells don't undercount objects, and object_fusion runs synchronously to keep
        # the index race-free. Default off (submapping alone stays all-resident).
        self.submap_evict = False
        # Cadence to verify the index against a full gather every N frames (0 = off).
        self.submap_verify_every = 0
        # Commit-on-leave (scalable traverses): when on (needs submapping:true), active
        # gaussians whose ground-plane cell has left the working set are force-graduated to
        # stable — by location, not confidence — so the active (Adam-optimized) pool stays
        # O(local working set) instead of O(trajectory).
        self.commit_on_leave = False
        self.commit_min_age = 10  # frames a splat must be active before it can be committed
        # Hard scalability backstop (0 = off): ceiling on the active (Adam-optimized) pool.
        # Surplus active splats are force-graduated to stable (then radius-evicted), bounding
        # peak VRAM in big environments even when location-based commit-on-leave lags.
        self.submap_active_max = 0
        # Cap frames processed (0 = whole dataset).
        self.max_frames = 0
        # Seed-point dedup filter: "grid" = voxel-hash fixed-radius,
        # "knn" = pytorch3d brute-force kNN.
        self.seed_filter_mode = "grid"
        # Novelty-driven local_optimize iteration scaling (mapper.adaptive_iter_count).
        self.adaptive_optimize = False
        self.adaptive_min_iter_ratio = 0.4
        self.rerun_log_step = 10
        self.xyz_factor = [1, 1, 1]
        self.use_tensorboard = True
        self.add_depth_thres = 0.05
        self.add_normal_thres = 0.1
        self.add_color_thres = 0.1
        self.add_transmission_thres = 0.1
        self.transmission_sample_ratio = 0.5
        self.error_sample_ratio = 0.3
        self.save_step = 1
        self.stable_confidence_thres = 200
        self.unstable_time_window = 50
        self.min_radius = 0.01
        self.max_radius = 0.10
        self.scale_factor = 0.5
        self.color_sigma = 1.0
        self.depth_filter = False
        self.verbose = False
        self.compile_renderer = False
        self.background_fusion = False
        self.log_level = "INFO"
        self.fps_log_every = 20

        self.global_keyframe_num = 3
        self.sync_tracker2mapper_method = "strict"
        self.sync_tracker2mapper_frames = 5
        self.clip_model = "ViT-H-14"
        super().__init__(parser, "Map Parameters", sentinel)


class SceneGraphParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.clip_model = "ViT-H-14"
        # Local checkpoint for MobileCLIP (only used when clip_model starts with 'MobileCLIP').
        self.clip_checkpoint_path = ""  # empty = $GSSG_CHECKPOINTS/mobileclip2_s0.pt
        # Manual override for image-embedding dim (auto-detected from clip_model if blank/0).
        self.clip_dim = 0
        # Blur-outside-polygon trick before CLIP; default off.
        self.clip_mask_blur = False
        # Room segmentation is self-calibrating; its only inputs are vertical_axis and
        # the method switch: ours | ours_v2 | hydra | hovsg.
        self.vertical_axis = 2  # 0=X, 1=Y (HM3D), 2=Z (Replica/ROS)
        self.room_seg_method = "ours"
        # Room-embedding quality gates: a frame contributes to a room's CLIP embedding
        # only if the camera stood inside the room and at least this far (m) from any
        # wall (doorway frames straddle two rooms and contaminate both) ...
        self.room_embed_wall_margin = 0.3
        # ... and a room needs at least this many contributing frames to get an
        # embedding at all (barely-visited rooms otherwise get a noisy one).
        self.room_embed_min_frames = 5

        super().__init__(parser, "Scene Graph Parameters", sentinel)
