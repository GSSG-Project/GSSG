"""ROS 2 live-stream input for ZED2i: subscribes to rectified RGB + registered
depth + camera_info + odometry and exposes synced frames as CameraInfo records
matching the ROS 1 ``ROSStream`` API.

rclpy is a compiled module pinned to the ROS 2 distro's Python (Humble = 3.10);
importing this module from an env with a different Python ABI fails with a clear
error. Run GSSG from an env that has rclpy on its sys.path (source the ROS 2 and
Isaac setup files first).
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R_scipy

from gssg.dataset_reader.dataset_readers import CameraInfo

try:
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped as RosPoseCov
    from message_filters import ApproximateTimeSynchronizer
    from message_filters import Subscriber as MFSubscriber
    from nav_msgs.msg import Odometry as RosOdometry
    from rclpy.duration import Duration as RclpyDuration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time as RclpyTime
    from sensor_msgs.msg import CameraInfo as RosCameraInfo
    from sensor_msgs.msg import CompressedImage as RosCompressedImage
    from sensor_msgs.msg import Image as RosImage
    from std_msgs.msg import Empty as RosEmpty
    from tf2_ros import Buffer as TF2Buffer
    from tf2_ros import (
        ConnectivityException,
        ExtrapolationException,
        LookupException,
        TransformListener,
    )

    _RCLPY_AVAILABLE = True
    _RCLPY_IMPORT_ERROR = None
except ImportError as _e:
    _RCLPY_AVAILABLE = False
    _RCLPY_IMPORT_ERROR = _e


def _require_rclpy():
    if not _RCLPY_AVAILABLE:
        raise ImportError(
            "ros2_utils.ROS2Stream requires rclpy (ROS 2). Import failed with:\n"
            f"  {_RCLPY_IMPORT_ERROR}\n"
            "Fix:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source ~/workspaces/isaac_ros-dev/install/setup.bash\n"
            "  # then run GSSG with a Python that matches the ROS 2 distro's Python "
            "(Humble = 3.10). If your conda env Python doesn't match, switch to one that does."
        )


def _msg_to_ndarray(msg, encoding: str) -> np.ndarray:
    """Pure-Python sensor_msgs/Image decoder (drop-in for cv_bridge)."""
    if encoding in ("rgb8", "bgr8"):
        dtype, channels = np.uint8, 3
    elif encoding in ("rgba8", "bgra8"):
        dtype, channels = np.uint8, 4
    elif encoding == "mono8":
        dtype, channels = np.uint8, 1
    elif encoding == "16UC1":
        dtype, channels = np.uint16, 1
    elif encoding == "32FC1":
        dtype, channels = np.float32, 1
    elif "32F" in encoding:
        dtype, channels = np.float32, 1
    elif "16U" in encoding:
        dtype, channels = np.uint16, 1
    elif "rgba" in encoding or "bgra" in encoding:
        dtype, channels = np.uint8, 4
    elif "rgb" in encoding or "bgr" in encoding:
        dtype, channels = np.uint8, 3
    else:
        dtype, channels = np.uint8, 1
    arr = np.frombuffer(msg.data, dtype=dtype)
    if channels > 1:
        arr = arr.reshape((msg.height, msg.width, channels))
    else:
        arr = arr.reshape((msg.height, msg.width))
    return arr


def _decode_image_msg(msg) -> tuple[np.ndarray, str]:
    """Decode a raw or compressed sensor_msgs image -> (ndarray, encoding).

    CompressedImage covers the image_transport ``compressed`` plugin (jpeg/png),
    NOT ``compressedDepth`` (custom header) — record depth as png 16UC1.
    """
    if hasattr(msg, "format"):
        arr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"cv2.imdecode failed (format={msg.format!r})")
        if arr.ndim == 3:
            return arr, "bgr8"  # cv2 decodes color as BGR regardless of source encoding
        return arr, msg.format.split(";")[0].strip()
    return _msg_to_ndarray(msg, msg.encoding), msg.encoding


class ROS2Stream:
    """Live ZED2i + cuVSLAM stream. Mirrors ROSStream's public API:
    `get_next_frame(frame_id) -> CameraInfo | None`.
    """

    def __init__(self, args=None, world_frame: str | None = None):
        _require_rclpy()

        # --- Topic / frame configuration ---
        get = (lambda k, d: getattr(args, k, d)) if args is not None else (lambda k, d: d)
        self.color_topic = get("ros2_color_topic", "/zed/zed_node/rgb/color/rect/image")
        self.depth_topic = get("ros2_depth_topic", "/zed/zed_node/depth/depth_registered")
        self.camera_info_topic = get(
            "ros2_camera_info_topic", "/zed/zed_node/rgb/color/rect/camera_info"
        )
        # Pose source: directly subscribe to ZED SDK's odometry topic. The pose
        # in this message corresponds EXACTLY to its header timestamp, no TF
        # interpolation involved. child_frame_id is zed_camera_link; we compose
        # with a one-time-looked-up static TF to get pose at the optical frame.
        self.pose_topic = get("ros2_pose_topic", "/zed/zed_node/odom")
        self.world_frame = (
            world_frame if world_frame is not None else get("ros2_world_frame", "odom")
        )
        self.camera_frame = get("ros2_camera_frame", "zed_left_camera_frame_optical")
        # Quality gate: reject frames whose position covariance trace exceeds
        # this (in meters^2). 0.0 = accept all. ZED PT typically reports
        # ~1e-5 m^2 when tracking well, 1e-2+ when poor. 0.01 is a safe upper
        # bound that lets through good tracking but skips lost frames.
        self.pose_cov_max = float(get("ros2_pose_cov_max", 0.01))
        self.pose_rot_cov_max = float(get("ros2_pose_rot_cov_max", 0.0))  # rad^2; 0 = off
        # Motion-blur gate: skip frames whose FFT sharpness score falls below
        # this. 0.0 = disabled. Calibrate from the periodic "[ROS2Stream] blur"
        # log line — sharp handheld frames typically score well above blurred
        # ones; pick a threshold between the two clusters.
        self.blur_thres = float(get("online_blur_thres", 0.0))
        self._blur_scores: list = []
        self._blur_skipped = 0
        # Tracking-status gate: drop frames while the ZED reports broken VIO
        # (odometry_status != OK) or area-map relocalization (spatial SEARCHING).
        # Covariance alone misses these — the SDK can report small covariance while
        # coasting or relocating. No status messages (non-ZED sources) = gate inert.
        self.gate_pose_status = bool(get("ros2_gate_pose_status", True))
        self.pose_status_topic = get("ros2_pose_status_topic", "/zed/zed_node/pose/status")
        self.pose_status_max_age = float(get("ros2_pose_status_max_age", 1.0))
        self._pose_status = None
        self._pose_status_time = 0.0
        self._status_skipped = 0         # cumulative within one degraded episode
        self._status_skipped_window = 0  # since last drop-stats report
        self._status_was_ok = True

        # Frame-accounting counters (drop diagnosis). _frames_dropped_busy counts sensor
        # frames overwritten in the single-slot buffer before the consumer read them
        # (i.e. lost because perception/mapper was busy); _cov_skipped counts frames
        # rejected by the covariance gate. Reported+reset via get_and_reset_drop_stats().
        self._frames_received = 0
        self._frames_dropped_busy = 0
        self._cov_skipped = 0
        self._latest_consumed = True

        # --- rclpy init (guarded — safe to call from any process) ---
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("gssg_ros2_stream")

        # --- Intrinsics (filled by first CameraInfo message) ---
        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None
        self._intrinsics_event = threading.Event()

        self._camera_info_sub = self.node.create_subscription(
            RosCameraInfo,
            self.camera_info_topic,
            self._camera_info_cb,
            qos_profile_sensor_data,
        )

        # --- TF listener (one-time lookup of the static camera_link -> optical
        # transform; not used per-frame, since we get the pose directly from
        # the odom topic). ---
        self.tf_buffer = TF2Buffer(cache_time=RclpyDuration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        # Cached static TF camera_link -> optical (resolved once on first
        # frame, then reused). Set lazily because TF isn't available at init.
        self._T_camlink_to_optical: np.ndarray | None = None

        # --- Synced color + depth + pose ---
        # ApproximateTimeSynchronizer matches messages by their header.stamp,
        # so each tuple it delivers is guaranteed to be from the SAME camera
        # frame (color/depth/pose were all computed for the same instant by
        # the ZED SDK).
        def _img_type(topic):
            return RosCompressedImage if topic.endswith("/compressed") else RosImage

        self.color_sub = MFSubscriber(
            self.node,
            _img_type(self.color_topic),
            self.color_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = MFSubscriber(
            self.node,
            _img_type(self.depth_topic),
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        # Pose message type: nav_msgs/Odometry for the VIO odom topic (default), or
        # geometry_msgs/PoseWithCovarianceStamped for the area-memory-corrected map-frame
        # pose (…/pose_with_covariance). Both expose .pose.pose and .pose.covariance, so
        # everything downstream is type-agnostic. Point ros2_pose_topic at the map-frame
        # topic (+ ros2_world_frame: map) to build the GSSG map with drift-corrected poses.
        pose_msg_type = RosPoseCov if "pose_with_covariance" in self.pose_topic else RosOdometry
        self.pose_sub = MFSubscriber(
            self.node, pose_msg_type, self.pose_topic, qos_profile=qos_profile_sensor_data
        )
        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.pose_sub],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self._image_callback)

        # --- External stop signal: publish an Empty message to /gssg/stop to
        # ask GSSG to wind down (e.g. `ros2 topic pub --once /gssg/stop ...`). ---
        self.stop_topic = get("ros2_stop_topic", "/gssg/stop")
        self._stop_event = threading.Event()
        self._stop_sub = self.node.create_subscription(
            RosEmpty,
            self.stop_topic,
            self._stop_callback,
            1,
        )

        # --- ZED positional-tracking status (plain latest-value sub, not synced:
        # the gate only needs "is tracking healthy right now"). zed_msgs is optional;
        # without it (non-ZED setups) the gate simply stays inert. ---
        if self.gate_pose_status:
            try:
                from zed_msgs.msg import PosTrackStatus

                self._pose_status_sub = self.node.create_subscription(
                    PosTrackStatus,
                    self.pose_status_topic,
                    self._pose_status_cb,
                    qos_profile_sensor_data,
                )
            except ImportError:
                print("[ROS2Stream] zed_msgs not importable — pose-status gate disabled")
                self.gate_pose_status = False

        # --- Shared state ---
        self.latest_data = None
        self._lock = threading.Lock()
        self.has_received_first_frame = False
        self.last_message_time = time.time()

        # --- rclpy spin in background thread ---
        self._shutdown = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True, name="rclpy-spin")
        self._spin_thread.start()

        print("[ROS2Stream] node up:")
        print(f"  color : {self.color_topic}")
        print(f"  depth : {self.depth_topic}")
        print(f"  info  : {self.camera_info_topic}")
        print(f"  pose  : {self.pose_topic}  (frame: {self.world_frame})")
        print(f"  cov_max: {self.pose_cov_max} m^2  (0 = accept all)")
        if self.gate_pose_status:
            print(f"  status: {self.pose_status_topic}  (gate: odometry NOT_OK / SEARCHING)")
        print(f"  stop  : ros2 topic pub --once {self.stop_topic} std_msgs/msg/Empty '{{}}'")

    # ------------------------------------------------------------------
    # rclpy plumbing
    # ------------------------------------------------------------------
    def _spin(self):
        # spin_once raises ExternalShutdownException when rclpy.shutdown()
        # fires from another thread — that's the normal exit path, not an error.
        try:
            from rclpy.executors import ExternalShutdownException
        except ImportError:
            ExternalShutdownException = ()  # type: ignore
        try:
            while not self._shutdown.is_set() and rclpy.ok():
                try:
                    rclpy.spin_once(self.node, timeout_sec=0.1)
                except ExternalShutdownException:
                    break
        except Exception as e:
            if rclpy.ok():  # only complain if it wasn't a clean shutdown
                print(f"[ROS2Stream] spin error: {type(e).__name__}: {e}")

    def _stop_callback(self, _msg):
        if not self._stop_event.is_set():
            print(f"[ROS2Stream] received {self.stop_topic} — flagging shutdown")
            self._stop_event.set()

    def _pose_status_cb(self, msg):
        self._pose_status = msg
        self._pose_status_time = time.time()

    def _tracking_degraded(self) -> str | None:
        """Reason string when the latest ZED tracking status says the pose is not
        mappable; None when healthy, unknown, or stale (fail-open for non-ZED runs)."""
        if not self.gate_pose_status or self._pose_status is None:
            return None
        if (time.time() - self._pose_status_time) > self.pose_status_max_age:
            return None
        st = self._pose_status
        if st.odometry_status != 0:  # PosTrackStatus.OK
            return "odometry NOT_OK (VIO lost frame-to-frame tracking)"
        if st.spatial_memory_status == 2:  # PosTrackStatus.SEARCHING
            return "spatial_memory SEARCHING (relocating in area map)"
        return None

    def _is_blurry(self, raw_rgb: np.ndarray) -> bool:
        from gssg.dataset_reader.dataset_readers import blur_score_fft_gray

        # ~3x downsample keeps the FFT ~1-2 ms; threshold is calibrated against
        # scores at this same scale (logged below), so absolute scale is irrelevant.
        step = max(1, raw_rgb.shape[1] // 426)
        gray = raw_rgb[::step, ::step].mean(axis=-1).astype(np.float32)
        score = blur_score_fft_gray(gray)
        self._blur_scores.append(score)
        if len(self._blur_scores) >= 100:
            s = np.array(self._blur_scores)
            print(
                f"[ROS2Stream] blur: score min/median/max = "
                f"{s.min():.1f}/{np.median(s):.1f}/{s.max():.1f} over {len(s)} frames, "
                f"{self._blur_skipped} skipped (thres={self.blur_thres:.1f})"
            )
            self._blur_scores.clear()
            self._blur_skipped = 0
        if score < self.blur_thres:
            self._blur_skipped += 1
            return True
        return False

    def should_stop(self) -> bool:
        """True once an external stop signal (e.g. /gssg/stop) has fired."""
        return self._stop_event.is_set()

    def _camera_info_cb(self, msg: RosCameraInfo):
        if self.fx is not None:
            return  # only need it once
        # K is row-major 3x3: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self._intrinsics_event.set()
        print(
            f"[ROS2Stream] intrinsics: fx={self.fx:.2f} fy={self.fy:.2f} "
            f"cx={self.cx:.2f} cy={self.cy:.2f} @ {msg.width}x{msg.height}"
        )

    def _image_callback(self, color_msg, depth_msg, odom_msg: RosOdometry):
        with self._lock:
            self.has_received_first_frame = True
            self.last_message_time = time.time()
            self._frames_received += 1
            # The previous frame was overwritten before anyone consumed it -> a busy drop.
            if self.latest_data is not None and not self._latest_consumed:
                self._frames_dropped_busy += 1
            self._latest_consumed = False
            self.latest_data = {
                "color": color_msg,
                "depth": depth_msg,
                "odom": odom_msg,
                "timestamp": color_msg.header.stamp,
            }

    def _ensure_static_tf(self) -> bool:
        """Resolve and cache the static TF camera_link -> optical (one-time).
        Returns True once cached, False while still waiting on the TF tree."""
        if self._T_camlink_to_optical is not None:
            return True
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                "zed_camera_link",
                self.camera_frame,
                RclpyTime(),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        T = np.eye(4)
        T[:3, :3] = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        self._T_camlink_to_optical = T
        print(f"[ROS2Stream] static TF zed_camera_link -> {self.camera_frame} resolved.")
        return True

    # ------------------------------------------------------------------
    # Public API (mirrors ROSStream)
    # ------------------------------------------------------------------
    def get_next_frame(self, frame_id: int) -> CameraInfo | None:
        # Wait for intrinsics (first CameraInfo message) before publishing frames.
        if not self._intrinsics_event.is_set():
            return None

        with self._lock:
            # Hand out each sensor frame at most once. Without the _latest_consumed
            # check the last frame is re-delivered forever once the stream stops, so
            # the session never ends: run.py only reaches its idle-timeout branch when
            # this returns None, and the map keeps ingesting duplicates of the final
            # frame (SVO/bag replay past end-of-file).
            if self.latest_data is None or self._latest_consumed:
                return None
            local = dict(self.latest_data)
            self._latest_consumed = True  # this sensor frame has now been handed out

        # Tracking-status gate: refuse to map while the ZED itself says the pose is
        # broken (VIO failure) or being relocated (SEARCHING). Logged on transitions
        # so a long outage is two lines, not thousands.
        degraded = self._tracking_degraded()
        if degraded is not None:
            self._status_skipped += 1
            self._status_skipped_window += 1
            if self._status_was_ok:
                print(f"[ROS2Stream] ZED tracking degraded: {degraded} — dropping frames")
                self._status_was_ok = False
            return None
        if not self._status_was_ok:
            print(
                f"[ROS2Stream] ZED tracking recovered "
                f"({self._status_skipped} frames dropped while degraded)"
            )
            self._status_was_ok = True
            self._status_skipped = 0

        # Need the static camera_link -> optical TF before we can compose the pose.
        if not self._ensure_static_tf():
            return None

        # --- Pose from the synchronized odom message (NOT TF interpolation).
        # nav_msgs/Odometry pose is:
        #     parent  = msg.header.frame_id   (typically 'odom')
        #     child   = msg.child_frame_id    (typically 'zed_camera_link')
        # The pose corresponds EXACTLY to msg.header.stamp — same instant as the
        # color/depth this odom was synchronized to.
        odom = local["odom"]
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation

        T_world_camlink = np.eye(4)
        T_world_camlink[:3, :3] = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T_world_camlink[:3, 3] = [p.x, p.y, p.z]

        # Compose: world -> optical = world -> camera_link * camera_link -> optical
        c2w = T_world_camlink @ self._T_camlink_to_optical

        # Localization-confidence gate: if ZED reports high pose covariance the pose is
        # untrustworthy — skip this frame rather than poison the map. Position (x,y,z) trace
        # is checked always; rotational (rx,ry,rz) trace — spin-induced uncertainty on fast
        # turns — when ros2_pose_rot_cov_max is set. Caller retries with the next frame.
        if self.pose_cov_max > 0.0 or self.pose_rot_cov_max > 0.0:
            cov = odom.pose.covariance  # 36-element row-major 6x6 (x,y,z,rx,ry,rz)
            pos_trace = cov[0] + cov[7] + cov[14]
            rot_trace = cov[21] + cov[28] + cov[35]
            if (self.pose_cov_max > 0.0 and pos_trace > self.pose_cov_max) or (
                self.pose_rot_cov_max > 0.0 and rot_trace > self.pose_rot_cov_max
            ):
                self._cov_skipped += 1
                return None

        w2c = np.linalg.inv(c2w)
        R_w2c = np.transpose(w2c[:3, :3])
        T_w2c = w2c[:3, 3]

        # --- Decode color + depth ---
        try:
            color_msg = local["color"]
            raw_rgb, color_encoding = _decode_image_msg(color_msg)
            # ZED can publish bgra8 / rgba8 — drop alpha and convert BGR -> RGB.
            if raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
                raw_rgb = raw_rgb[..., :3]
            if raw_rgb.ndim == 3 and color_encoding.startswith("bgr"):
                raw_rgb = raw_rgb[..., ::-1].copy()  # BGR -> RGB

            # Motion-blur gate: drop blurred frames before they cost SAM/CLIP +
            # mapping and poison the map with smeared color/geometry. Same FFT
            # sharpness metric as the offline filter, on a downsampled gray.
            if self.blur_thres > 0.0 and self._is_blurry(raw_rgb):
                return None

            image_pil = PILImage.fromarray(raw_rgb)

            depth_msg = local["depth"]
            raw_depth, depth_encoding = _decode_image_msg(depth_msg)
            if "32F" in depth_encoding:
                raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0).astype(
                    np.float32
                )
            else:
                raw_depth = raw_depth.astype(np.float32) / 1000.0  # mm -> m
            depth_pil = PILImage.fromarray(raw_depth, mode="F")
        except Exception as e:
            print(f"[ROS2Stream] image decode error: {e}")
            return None

        h, w = raw_rgb.shape[:2]
        fov_x_rad = 2.0 * np.arctan(w / (2.0 * self.fx))
        fov_y_rad = 2.0 * np.arctan(h / (2.0 * self.fy))

        ts_sec = float(local["timestamp"].sec) + float(local["timestamp"].nanosec) * 1e-9

        return CameraInfo(
            height=h,
            width=w,
            uid=frame_id,
            R=R_w2c,
            T=T_w2c,
            FovX=fov_x_rad,
            FovY=fov_y_rad,
            image=image_pil,
            depth=depth_pil,
            image_name=f"ros2_{frame_id}",
            pose_gt=np.eye(4),
            cx=self.cx,
            cy=self.cy,
            timestamp=ts_sec,
            depth_scale=1000.0,
            image_path="ros2",
            depth_path="ros2",
        )

    def get_and_reset_drop_stats(self) -> dict:
        """Per-window frame accounting for the [FRAMES] diagnostic line. Returns sensor
        frames received, frames lost to a busy/blocked consumer, and covariance-gated
        frames since the last call, then resets those counters."""
        with self._lock:
            stats = {
                "received": self._frames_received,
                "dropped_busy": self._frames_dropped_busy,
                "cov_skipped": self._cov_skipped,
                "status_skipped": self._status_skipped_window,
            }
            self._frames_received = 0
            self._frames_dropped_busy = 0
            self._cov_skipped = 0
            self._status_skipped_window = 0
        return stats

    def release(self):
        self._shutdown.set()
        # Join the spin thread first so it stops calling into rclpy before
        # we tear the context down.
        try:
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if rclpy.ok():
                self.node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
