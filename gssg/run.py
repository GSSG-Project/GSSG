import matplotlib

matplotlib.use("Agg")

import copy
import logging
import os
import signal
import threading
import time
from argparse import ArgumentParser

import torch
import torchvision

from gssg.dataset_reader import Dataset
from gssg.map.mapper import _nvml_used_gb, Mapping
from gssg.map.perception import Perception
from gssg.scene_graph import SceneGraph
from gssg.utils.arguments import DatasetParams, OptimizationParams
from gssg.utils.camera_utils import loadCam
from gssg.utils.config_utils import _QUALITY_FILES, read_config
from gssg.utils.eval import eval_frame
from gssg.utils.export_frame import export_frame_and_params
from gssg.utils.fps import FPSMeter
from gssg.utils.general_utils import safe_state

# ROS2Stream is imported lazily below when dataset_type == "ros2" to avoid
# pulling rclpy into processes that don't have it on their path.
from gssg.utils.rerun_utils import rerun_init, rerun_log_frame, rerun_log_gaussians
from gssg.utils.ros_utils import ROSStream  # ROS 1 (rospy)
from gssg.utils.save_render import render_detections
from gssg.utils.utils import move_to_cpu, move_to_gpu, prepare_cfg

torch.set_printoptions(4, sci_mode=False)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, default="configs/datasets/replica.yaml")
    parser.add_argument(
        "--quality",
        type=str,
        default=None,
        choices=list(_QUALITY_FILES),
        help="Quality preset; overrides the dataset default (base<-quality<-dataset).",
    )
    parser.add_argument(
        "--full_render_frame", action="store_true", help="Render all frames at the end"
    )
    args_cli = parser.parse_args()

    args = read_config(args_cli.config, quality=args_cli.quality)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args.device_list)

    optimization_params = OptimizationParams(parser).extract(args)
    dataset_params = DatasetParams(parser, sentinel=True).extract(args)
    safe_state(getattr(args, "quiet", False))

    if args.visualize:
        rerun_init(args.save_path)

    dataset = Dataset(dataset_params)
    logging.info("Dataset loaded")

    scene_graph = SceneGraph(args)
    gaussian_map = Mapping(args, scene_graph)
    gaussian_map.create_workspace()
    perception = Perception(args)
    prepare_cfg(args)

    fps_meter = FPSMeter(name="MAIN", log_every=getattr(args, "fps_log_every", 20))
    runtime = gaussian_map.runtime_stats

    use_ros = dataset.dataset_type in ("ros", "ros2")
    ros_stream = None

    if dataset.dataset_type == "ros":
        logging.info("Initializing ROS 1 Stream for Single Process...")
        ros_stream = ROSStream(args)
    elif dataset.dataset_type == "ros2":
        logging.info("Initializing ROS 2 (ZED2i + cuVSLAM) Stream for Single Process...")
        from gssg.utils.ros2_utils import ROS2Stream

        ros_stream = ROS2Stream(args)
    else:
        train_cameras = dataset.scene_info.train_cameras

    frame_id = 0
    ros_timeout = float(getattr(args, "ros2_idle_timeout", 8.0))
    start_wait_log_step = 0

    # Graceful stop, triggered by any of: SIGINT (Ctrl+C); a publish to /gssg/stop
    # (ros2 only); or ROS topics going silent for ros_timeout seconds (ros / ros2).
    _stop_requested = threading.Event()

    def _sigint_handler(signum, frame):
        if _stop_requested.is_set():
            # Second Ctrl+C — abandon without saving.
            logging.warning("[STOP] FORCED EXIT on second Ctrl+C — scene NOT saved.")
            os._exit(130)
        logging.warning(
            "[STOP] Graceful stop requested. Will finish current frame, then run "
            "global fusion + final optimization + save (PLY, scene_graph, OpenLex3D). "
            "This can take 10-30 seconds. Press Ctrl+C AGAIN ONLY if you want to "
            "abandon without saving."
        )
        _stop_requested.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            # Stop checks run before blocking work so the loop exits within
            # one frame of receiving the signal/topic.
            if _stop_requested.is_set():
                logging.info("[STOP] stop flag set — exiting scan loop.")
                break
            if (
                use_ros
                and ros_stream is not None
                and getattr(ros_stream, "should_stop", lambda: False)()
            ):
                logging.info("[STOP] /gssg/stop received — exiting scan loop.")
                break

            frame_info = None

            if use_ros:
                frame_info = ros_stream.get_next_frame(frame_id)

                if frame_info is None:
                    if not ros_stream.has_received_first_frame:
                        if start_wait_log_step % 250 == 0:
                            logging.info("Waiting for ROS topic to start...")
                        start_wait_log_step += 1
                        time.sleep(0.1)
                        continue

                    time_since_last = time.time() - ros_stream.last_message_time
                    if time_since_last > ros_timeout:
                        logging.info(f"ROS Topic stopped for {ros_timeout}s. Ending session.")
                        break
                    time.sleep(0.01)
                    continue
            else:
                if frame_id >= len(train_cameras) or (
                    getattr(args, "max_frames", 0) and frame_id >= args.max_frames
                ):
                    logging.info("End of dataset reached.")
                    break
                frame_info = train_cameras[frame_id]

            with fps_meter:
                res_scale = 1.0 if use_ros else dataset_params.resolution_scales[0]
                curr_frame = loadCam(dataset_params, frame_id, frame_info, res_scale)

                if args.visualize:
                    rerun_log_frame(frame_id, frame_info.image)

                if frame_id % 10 == 0:
                    logging.info(f"\n\n========== CURRENT FRAME: ___{frame_id}___ ==========\n")

                move_to_gpu(curr_frame)

                with runtime.timer("preprocess", cuda_sync=True):
                    frame_map = perception.map_preprocess(curr_frame, frame_id)
                with runtime.timer("tracking"):
                    perception.tracking(curr_frame, frame_map)

                with runtime.timer("mapping", cuda_sync=True):
                    gaussian_map.mapping(curr_frame, frame_map, frame_id, optimization_params)
                with runtime.timer("render", cuda_sync=True):
                    gaussian_map.get_render_output(curr_frame)

                if getattr(args, "log_scaling_series", False):
                    # per-frame sample for the Fig. 4 scaling curves
                    resident = int(gaussian_map.get_stable_num + gaussian_map.get_unstable_num)
                    # with submapping the resident count drops on eviction, so also
                    # record the true map size (resident + evicted) for the x axis
                    total = resident
                    if getattr(gaussian_map, "submaps", None) is not None:
                        total = int(gaussian_map.submaps.total_stable_count(resident))
                    runtime.extra.setdefault("scaling", []).append({
                        "frame": frame_id,
                        "gaussians": resident,
                        "gaussians_total": total,
                        "vram_gb": torch.cuda.memory_allocated() / 2**30,
                        "vram_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                        "vram_reserved_gb": torch.cuda.memory_reserved() / 2**30,
                        "vram_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                        "nvml_used_gb": _nvml_used_gb(),
                    })
                    # flush periodically: an OOM/crash near the end must not
                    # take the whole series with it
                    if frame_id % 100 == 0:
                        runtime.write_series(args.save_path)

            if (gaussian_map.time + 1) % gaussian_map.save_step == 0:
                logging.info(f"Saving checkpoint at frame {frame_id}")
                gaussian_map.ensure_full_residency()  # eval/save read the whole map
                eval_frame(
                    gaussian_map,
                    curr_frame,
                    os.path.join(gaussian_map.save_path, "eval_render"),
                    min_depth=gaussian_map.min_depth,
                    max_depth=gaussian_map.max_depth,
                    save_picture=True,
                    run_pcd=False,
                )
                gaussian_map.save_model(save_data=True)
                gaussian_map.segment_rooms()

            gaussian_map.time += 1
            frame_id += 1
            move_to_cpu(curr_frame)
            # empty_cache forces a device sync and re-cudaMalloc churn, so throttle
            # it to a periodic safety valve rather than running it every frame.
            if frame_id % 50 == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        # Backstop: only reached if the interrupt arrives before the handler is installed.
        logging.info(
            "\n\n[USER INTERRUPT] Stopping capture. Proceeding to optimization and save...\n"
        )

    # Tell the live stream to wind down (closes the rclpy spin thread cleanly).
    if use_ros and ros_stream is not None:
        try:
            ros_stream.release()
        except Exception:
            pass

    if perception.semantic_encoder is not None and fps_meter.frames:
        encoder = perception.semantic_encoder
        runtime.extra["encoder_ms_per_frame"] = encoder.get_and_reset_perf_stats(fps_meter.frames)

    # The segmentation and CLIP models are only used inside the frame loop (object
    # naming reads precomputed text vectors from the DB, not the live encoder), so
    # release them before the post-loop passes: the final joint optimization needs
    # the whole map resident and runs within a few hundred MB of the GPU ceiling.
    # A background fusion worker may still be mid-flight here; join it first —
    # destroying BestSAM's captured CUDA graph while another thread launches CUDA
    # work segfaults in cuGraphExecDestroy (observed on Jetson Thor / L4T r39).
    gaussian_map.await_background_fusion()
    del perception
    torch.cuda.empty_cache()

    # Page every evicted cell back before the post-loop passes (fusion, final global
    # optimization, full render, eval, export) — they need the whole map resident.
    if getattr(gaussian_map, "submaps", None) is not None:
        logging.info("[SUBMAP] run stats: %s", gaussian_map.submaps.stats())
    gaussian_map.ensure_full_residency()

    logging.info("Cleaning objects without points...")
    scene_graph.clean_objects_without_points()

    logging.info("Running Object Fusion...")
    for _i in range(3):
        gaussian_map.object_fusion(global_run=True)

    if args.visualize:
        rerun_log_gaussians(gaussian_map)

    if not use_ros:
        render_detections(
            gaussian_map,
            dataset.scene_info.train_cameras,
            dataset_params,
            gaussian_map.save_path,
            args.frame_max,
            render_semantics=True,
            scene_graph=scene_graph,
        )

    logging.info("\n========== Main loop finish ==========\n")
    logging.info(
        f"Stable num: {gaussian_map.get_stable_num}, unstable num: {gaussian_map.get_unstable_num}"
    )
    logging.info(f"Processed frame: {gaussian_map.optimize_frames_ids}")
    save_dir = os.path.join(gaussian_map.save_path, "gaussian_map")

    # Check if curr_frame exists (in case loop crashed immediately)
    if "curr_frame" in locals():
        export_frame_and_params(curr_frame, gaussian_map.unstable_params, output_dir=save_dir)

    gaussian_map.global_optimization(optimization_params, is_end=True)

    runtime.extra["final_eval"] = eval_frame(
        gaussian_map,
        gaussian_map.keyframe_list[-1],
        os.path.join(gaussian_map.save_path, "eval_render"),
        min_depth=gaussian_map.min_depth,
        max_depth=gaussian_map.max_depth,
        save_picture=True,
        run_pcd=False,
    )

    gaussian_map.save_model(save_data=True)
    gaussian_map.time += 1

    gaussian_map.export_all()

    runtime.extra["frames"] = fps_meter.frames
    if fps_meter.total_ms > 0:
        runtime.extra["fps_mean"] = fps_meter.frames / (fps_meter.total_ms / 1000.0)
    runtime.extra["final_gaussians"] = int(
        gaussian_map.get_stable_num + gaussian_map.get_unstable_num
    )
    logging.info(f"Runtime stats written to {runtime.write(args.save_path)}")

    if args.full_render_frame:
        logging.info("Rendering all frames for final output (full_render_frame=True)...")
        from tqdm import tqdm

        save_render_dir = os.path.join(gaussian_map.save_path, "full_render_frame")
        os.makedirs(save_render_dir, exist_ok=True)

        # Reload dataset with no filters to get all frames
        logging.info("Reloading dataset for full render (ignoring stride/max_frames)...")
        full_dataset_params = copy.deepcopy(dataset_params)
        full_dataset_params.frame_max = -1
        full_dataset_params.frame_start = 0
        full_dataset_params.frame_step = 1
        full_dataset_params.use_keyframe = False

        full_dataset = Dataset(full_dataset_params)
        cameras_to_render = full_dataset.scene_info.train_cameras

        # Choose parameters to render with (Global = Active + Stable)
        render_params = gaussian_map.global_params

        with torch.no_grad():
            for frame_id, frame_info in enumerate(tqdm(cameras_to_render, desc="Rendering Frames")):
                res_scale = (
                    1.0
                    if dataset.dataset_type in ("ros", "ros2")
                    else dataset_params.resolution_scales[0]
                )
                curr_frame = loadCam(dataset_params, frame_id, frame_info, res_scale)
                move_to_gpu(curr_frame)

                render_output = gaussian_map.renderer.render(curr_frame, render_params)

                render_image = render_output["render"]
                torchvision.utils.save_image(
                    render_image, os.path.join(save_render_dir, f"frame_{frame_id:05d}.png")
                )

                move_to_cpu(curr_frame)
                torch.cuda.empty_cache()

        logging.info(f"All frames rendered to {save_render_dir}")
    else:
        logging.info("Skipping full frame rendering (use --full_render_frame to enable).")


if __name__ == "__main__":
    main()
