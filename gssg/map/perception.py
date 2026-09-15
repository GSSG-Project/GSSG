import logging
import os
import queue
import time as timex
from collections import defaultdict

import torch
import torch.multiprocessing as mp

from gssg.dataset_reader.dataset_readers import rot_compare, trans_compare
from gssg.map.preprocessor import preprocess
from gssg.scene_graph.semantic_encoder import SemanticEncoder
from gssg.utils.camera_utils import loadCam
from gssg.utils.fps import FPSMeter
from gssg.utils.rerun_utils import rerun_log_frame
from gssg.utils.ros_utils import ROSStream
from gssg.utils.utils import (
    get_rot,
    move_to_cpu,
    move_to_gpu,
    stage_map_input_for_queue,
    transform_map,
)


class Perception:
    def __init__(self, args):
        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.depth_filter = args.depth_filter
        self.verbose = args.verbose

        self.status = defaultdict(bool)
        self.pose_gt = []
        self.pose_es = []
        self.finish = mp.Condition()

        self.invalid_confidence_thresh = args.invalid_confidence_thresh
        self.semantic_encoder = SemanticEncoder(args)
        self.visualize = args.visualize

    def map_preprocess(self, frame, frame_id):
        return preprocess(self, frame, frame_id)

    def update_curr_status(
        self,
        frame,
        frame_id,
        depth_t1,
        depth_t1_filter,
        vertex_t1,
        normal_t1,
        color_t1,
        semantic_map,
    ):
        self.curr_frame = {
            "K": frame.get_intrinsic,
            "normal_map": normal_t1,
            "depth_map": depth_t1,
            "depth_map_filter": depth_t1_filter,
            "vertex_map": vertex_t1,
            "frame_id": frame_id,
            "pose_gt": frame.get_c2w.cpu().numpy(),
            "color_map": color_t1,
            "timestamp": frame.timestamp,
            "semantic_map": semantic_map,
        }

    def tracking(self, frame, frame_map):
        self.pose_gt.append(self.curr_frame["pose_gt"])
        pose_t1_w = self.pose_gt[-1]
        self.pose_es.append(pose_t1_w)
        frame.updatePose(pose_t1_w)
        frame_map["vertex_map_w"] = transform_map(frame_map["vertex_map_c"], frame.get_c2w)
        frame_map["normal_map_w"] = transform_map(frame_map["normal_map_c"], get_rot(frame.get_c2w))


class PerceptionProcess(Perception):
    def __init__(self, slam, args):
        self.args = args

        self.sync_tracker2mapper_method = slam.sync_tracker2mapper_method
        self.sync_tracker2mapper_frames = slam.sync_tracker2mapper_frames
        self._tracker2mapper_call = slam._tracker2mapper_call
        self._tracker2mapper_frame_queue = slam._tracker2mapper_frame_queue
        self.mapper_running = True
        self._mapper2tracker_call = slam._mapper2tracker_call
        self._mapper2tracker_map_queue = slam._mapper2tracker_map_queue

        self.dataset_type = slam.dataset.dataset_type
        self.ros_stream = None  # set below for ros/ros2; None for dataset runs
        if self.dataset_type in ("ros", "ros2"):
            self.dataset_cameras = []
            self.last_t = None
            self.last_r = None
        else:
            self.dataset_cameras = slam.dataset.scene_info.train_cameras

        self.map_process = slam.map_process
        self._end = slam._end
        self._stop = slam._stop  # graceful-stop flag set by the parent's SIGINT handler
        self.frame_id = 0
        self.last_mapper_frame_id = 0
        self.save_path = args.save_path
        # Frame/trajectory diagnostics.
        self._kf_skipped = 0  # frames received but rejected by the keyframe gate
        self._block_ms = 0.0  # ms spent blocked in wait_for_mapper (backpressure)
        self._q_dropped = 0  # keyframes dropped at a full mapper queue (mapper behind)
        self._traj_file = None
        self._traj_prev_xyz = None
        self._traj_prev_stamp = None
        self._traj_prev_wall = None
        self.finish = mp.Event()

    def map_preprocess_mp(self, frame, frame_id):
        self.map_input = super().map_preprocess(frame, frame_id)

    def send_frame_to_mapper(self, drop_if_full=True):
        # Staging CPU-copies the payload so it can cross the queue (no-op on dGPU; needed
        # where there is no CUDA IPC). Stage outside the lock.
        payload = stage_map_input_for_queue(self.map_input)
        # Enqueue OUTSIDE the condition lock: a blocking put on a full bounded queue while
        # holding _tracker2mapper_call would deadlock (the mapper needs the same lock to
        # get() and free space). The lock only guards the notify.
        if drop_if_full:
            # Free sync: if the mapper is behind (queue full), drop this keyframe instead
            # of stalling intake (graceful, pose-spaced coverage loss).
            try:
                self._tracker2mapper_frame_queue.put_nowait(payload)
            except queue.Full:
                self._q_dropped += 1
                return
        else:
            # Control message (e.g. the time=-1 shutdown sentinel): must be delivered.
            self._tracker2mapper_frame_queue.put(payload)
        with self._tracker2mapper_call:
            self.map_process._requests[0] = True
            self._tracker2mapper_call.notify()

    def finish_(self):
        # Graceful stop (Ctrl+C in the parent) ends scanning for every dataset type, so
        # the loop falls through to the sentinel -> final optimization + save path.
        if getattr(self, "_stop", None) is not None and self._stop[0] == 1:
            return True
        if self.dataset_type in ("ros", "ros2"):
            return False
        return self.frame_id >= len(self.dataset_cameras)

    def _ensure_ros_stream(self):
        if self.ros_stream is not None:
            return
        if self.dataset_type == "ros":
            print("Initializing ROS 1 Stream in Perception...")
            self.ros_stream = ROSStream(self.args)
        elif self.dataset_type == "ros2":
            print("Initializing ROS 2 (ZED2i + cuVSLAM) Stream in Perception...")
            from gssg.utils.ros2_utils import ROS2Stream

            self.ros_stream = ROS2Stream(self.args)

    def get_next_frame(self):
        if self.dataset_type in ("ros", "ros2"):
            self._ensure_ros_stream()
            frame_info = self.ros_stream.get_next_frame(self.frame_id)

            if frame_info is None:
                return None, None

            # --- Keyframe Selection Logic ---
            is_keyframe = None
            if self.last_r is not None and self.last_t is not None:
                _, theta_diff = rot_compare(self.last_r, frame_info.R)
                _, l2_diff = trans_compare(self.last_t, frame_info.T)
                is_keyframe = (
                    theta_diff > self.args.keyframe_theta_thres
                    or l2_diff > self.args.keyframe_trans_thres
                )

            if self.frame_id == 0 or is_keyframe:
                self.last_r = frame_info.R
                self.last_t = frame_info.T
                frame = loadCam(self.args, self.frame_id, frame_info, 1.0)  # Scale 1.0 for ROS
                self.frame_id += 1
                return frame, frame_info

            self._kf_skipped += 1
            return None, None

        # --- Dataset Logic ---
        if self.frame_id >= len(self.dataset_cameras):
            return None, None

        frame_info = self.dataset_cameras[self.frame_id]
        frame = loadCam(self.args, self.frame_id, frame_info, self.args.resolution_scales[0])
        self.frame_id += 1
        return frame, frame_info

    def run(self):
        print("Perception Process: Initializing Models (CLIP & Renderer)...")
        finish_event_handle = self.finish

        Perception.__init__(self, self.args)

        self.finish = finish_event_handle
        self.time = 0

        _lvl = getattr(self.args, "log_level", "INFO").upper()
        logging.getLogger().setLevel(getattr(logging, _lvl, logging.INFO))

        # Per-mapped-frame trajectory log. dpos + dt_stamp distinguish an odom teleport
        # (large dpos, small dt_stamp) from dropped/skipped camera frames (both large).
        # Best-effort: never break the mapping loop.
        try:
            os.makedirs(self.save_path, exist_ok=True)
            self._traj_file = open(os.path.join(self.save_path, "trajectory.csv"), "w", buffering=1)
            self._traj_file.write(
                "frame_id,stamp,wall_t,tx,ty,tz,qw,qx,qy,qz,dpos,dt_stamp,dt_wall\n"
            )
        except Exception as e:
            print(f"[TRAJ] could not open trajectory.csv: {e}", flush=True)
            self._traj_file = None

        # FPSMeter is silent; a single consolidated line below combines FPS + encoder
        # stage stats + perception stage stats.
        fps_meter = FPSMeter(
            name="PERCEPTION",
            log_every=getattr(self.args, "fps_log_every", 20),
            silent=True,
        )

        _stg_log_every = getattr(self.args, "fps_log_every", 20)
        _stg = {"rerun": 0.0, "ipc": 0.0}
        _stg_n = 0

        # --- ROS idle timeout: finalize the session after the live stream stops ---
        ros_timeout = float(getattr(self.args, "ros2_idle_timeout", 8.0))
        start_wait_log_step = 0

        while not self.finish_():
            frame, frame_info = self.get_next_frame()

            if frame is None:
                if self.dataset_type in ("ros", "ros2"):
                    if self.ros_stream is not None:
                        if not self.ros_stream.has_received_first_frame:
                            if start_wait_log_step % 200 == 0:
                                print("Perception: Waiting for ROS topic to start...")
                            start_wait_log_step += 1
                            timex.sleep(0.1)
                            continue
                        time_since_last = timex.time() - self.ros_stream.last_message_time
                        # Survive the ZED startup tracking-reset gap (pose/gravity re-init
                        # pauses the stream ~10-15 s after the first frame): require a few
                        # processed frames before finalizing on idle. 60 s hard cap exits
                        # if no stream forms.
                        if (
                            time_since_last > ros_timeout and self.time > 5
                        ) or time_since_last > 60.0:
                            print(
                                f"Perception: ROS topic idle {time_since_last:.1f}s "
                                f"(processed {self.time} frames). Ending session."
                            )
                            break
                    timex.sleep(0.01)
                    continue
                else:
                    break

            start_wait_log_step = 0

            frame_id = frame.uid
            self._log_traj(frame, frame_id)

            with fps_meter:
                move_to_gpu(frame)

                # map_preprocess includes the heavy semantic encoding (SAM3 + CLIP).
                self.map_preprocess_mp(frame, frame_id)

                _t = timex.perf_counter()
                if self.args.visualize:
                    rerun_log_frame(frame_id, frame_info.image)
                _rerun_ms = (timex.perf_counter() - _t) * 1000.0

                self.tracking(frame, self.map_input)
                self.map_input["frame"] = frame

                _t = timex.perf_counter()
                move_to_cpu(frame)
                self.send_frame_to_mapper()
                _ipc_ms = (timex.perf_counter() - _t) * 1000.0

            _stg["rerun"] += _rerun_ms
            _stg["ipc"] += _ipc_ms
            _stg_n += 1
            if _stg_n % _stg_log_every == 0:
                n = _stg_log_every
                if self.semantic_encoder is not None:
                    enc_stats = self.semantic_encoder.get_and_reset_perf_stats(n)
                    enc_part = (
                        f"SAM {enc_stats['sam_ms']:5.1f}  CLIP {enc_stats['clip_ms']:5.1f}  "
                        f"prep {enc_stats['clip_prep_ms']:5.1f}  "
                        f"post {enc_stats['mask_post_ms']:5.1f} | "
                    )
                else:
                    enc_part = ""
                # release cached-free blocks to the driver so whole-GPU nvml readings
                # reflect live usage; runs once per report interval (~every 20 keyframes)
                torch.cuda.empty_cache()
                print(
                    f"[PERCEPTION] {fps_meter.recent_fps:5.2f} FPS ({fps_meter.recent_ms:6.1f} ms)"
                    f" | {enc_part}"
                    f"rerun {_stg['rerun'] / n:4.1f}  ipc {_stg['ipc'] / n:4.1f} ms",
                    flush=True,
                )
                _stg = {"rerun": 0.0, "ipc": 0.0}

                ds = (
                    self.ros_stream.get_and_reset_drop_stats()
                    if (
                        self.ros_stream is not None
                        and hasattr(self.ros_stream, "get_and_reset_drop_stats")
                    )
                    else None
                )
                if ds is not None:
                    print(
                        f"[FRAMES] mapped {n} | recv {ds['received']} | "
                        f"dropped_busy {ds['dropped_busy']} | cov_skip {ds['cov_skipped']} | "
                        f"kf_skip {self._kf_skipped} | q_drop {self._q_dropped} | "
                        f"block {self._block_ms:6.0f} ms",
                        flush=True,
                    )
                    self._kf_skipped = 0
                    self._block_ms = 0.0
                    self._q_dropped = 0

            if not self.finish_() and self.mapper_running:
                self.unpack_map_to_tracker_noblock()

                if self.sync_tracker2mapper_method == "strict":
                    if (frame_id + 1) % self.sync_tracker2mapper_frames == 0:
                        self.wait_for_mapper()
                elif self.sync_tracker2mapper_method == "loose":
                    if (frame_id - self.last_mapper_frame_id) > self.sync_tracker2mapper_frames:
                        self.wait_for_mapper()

            self.time += 1

        if self._traj_file is not None:
            try:
                self._traj_file.close()
            except Exception:
                pass
            self._traj_file = None

        print("Perception: Loop finished. Sending shutdown signal to Mapper...")

        # Wind the live stream down cleanly (closes the rclpy spin thread / rospy subs).
        if self.dataset_type in ("ros", "ros2") and self.ros_stream is not None:
            try:
                self.ros_stream.release()
            except Exception as e:
                print(f"Perception: ros_stream release error: {e}")

        self.map_input = {}
        self.map_input["time"] = -1
        self.send_frame_to_mapper(drop_if_full=False)
        self._end[0] = 1

        # The mapper's global fusion + final joint optimize runs next and is the VRAM
        # peak of the whole run; this process would otherwise sit on ~10 GB of idle
        # encoder models until the finish event (mirrors run.py's del-before-post-loop).
        if self.semantic_encoder is not None:
            self.semantic_encoder.release()
            self.semantic_encoder = None

        print("Perception: Waiting for external finish event...")
        self.finish.wait()
        print("Perception: Finished.")

    def wait_for_mapper(self):
        _t0 = timex.perf_counter()
        try:
            with self._mapper2tracker_call:
                while (self.frame_id - self.last_mapper_frame_id) > self.sync_tracker2mapper_frames:
                    # Bail out on graceful stop / end-of-stream so a Ctrl+C while parked
                    # here doesn't deadlock (the mapper may be idle and never notify).
                    if self.finish_():
                        return
                    has_updates = self._consume_map_queue()
                    if not has_updates:
                        # Timeout so the loop + finish_() re-check even with no mapper update.
                        self._mapper2tracker_call.wait(timeout=0.5)
        finally:
            self._block_ms += (timex.perf_counter() - _t0) * 1000.0

    @staticmethod
    def _stamp_float(ts):
        """Coerce a frame timestamp (float seconds or a ROS Time msg) to float seconds."""
        if ts is None:
            return 0.0
        if hasattr(ts, "sec"):
            return float(ts.sec) + float(getattr(ts, "nanosec", 0)) * 1e-9
        try:
            return float(ts)
        except (TypeError, ValueError):
            return 0.0

    def _log_traj(self, frame, frame_id):
        """Append one row per mapped frame: pose + distance/time gap from the previous
        mapped frame. Best-effort: disables itself on any error."""
        if self._traj_file is None:
            return
        try:
            from gssg.utils.general_utils import matrix_to_quaternion_wxyz

            c2w = frame.get_c2w.detach()
            t = c2w[:3, 3].reshape(-1).cpu().numpy()
            q = matrix_to_quaternion_wxyz(c2w[:3, :3]).reshape(-1).cpu().numpy()  # (w,x,y,z)
            stamp = self._stamp_float(getattr(frame, "timestamp", 0.0))
            wall = timex.time()
            if self._traj_prev_xyz is not None:
                dpos = float(((t - self._traj_prev_xyz) ** 2).sum() ** 0.5)
                dt_stamp = stamp - self._traj_prev_stamp
                dt_wall = wall - self._traj_prev_wall
            else:
                dpos = dt_stamp = dt_wall = 0.0
            self._traj_file.write(
                f"{frame_id},{stamp:.6f},{wall:.6f},{t[0]:.5f},{t[1]:.5f},{t[2]:.5f},"
                f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f},"
                f"{dpos:.4f},{dt_stamp:.4f},{dt_wall:.4f}\n"
            )
            self._traj_prev_xyz = t
            self._traj_prev_stamp = stamp
            self._traj_prev_wall = wall
        except Exception as e:
            print(f"[TRAJ] logging disabled after error: {e}", flush=True)
            self._traj_file = None

    def unpack_map_to_tracker_noblock(self):
        with self._mapper2tracker_call:
            self._consume_map_queue()

    def _consume_map_queue(self):
        updated = False
        while not self._mapper2tracker_map_queue.empty():
            map_info = self._mapper2tracker_map_queue.get()
            self.last_mapper_frame_id = map_info["frame_id"]
            updated = True
        return updated

    def stop(self):
        self.finish.set()
