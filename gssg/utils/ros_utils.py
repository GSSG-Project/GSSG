import sys
import threading
import time

import numpy as np
from PIL import Image

from gssg.dataset_reader.dataset_readers import CameraInfo

# ROS 1 (noetic) packages live outside the conda env's site-packages.
try:
    sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
    import message_filters
    import rospy
    from sensor_msgs.msg import Image as RosImage
    from tf2_msgs.msg import TFMessage
except ImportError:
    print("ROS is not installed.")

from scipy.spatial.transform import Rotation as R_scipy


class PurePythonTFListener:
    """
    Manually builds the TF tree from /tf and /tf_static topics.
    Workaround for Conda environments incompatible with system libtf2.
    """

    def __init__(self):
        self.transforms = {}
        self.lock = threading.Lock()

        # Subscribe to standard TF topics
        self.tf_sub = rospy.Subscriber("/tf", TFMessage, self._callback)
        self.tf_static_sub = rospy.Subscriber("/tf_static", TFMessage, self._callback)

    def _callback(self, msg):
        with self.lock:
            for transform in msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id

                tx = transform.transform.translation.x
                ty = transform.transform.translation.y
                tz = transform.transform.translation.z

                rx = transform.transform.rotation.x
                ry = transform.transform.rotation.y
                rz = transform.transform.rotation.z
                rw = transform.transform.rotation.w

                self.transforms[child] = {
                    "parent": parent,
                    "trans": np.array([tx, ty, tz]),
                    "rot": np.array([rx, ry, rz, rw]),
                    "time": transform.header.stamp.to_sec(),
                }

    def lookup_transform(self, target_frame, source_frame):
        with self.lock:
            # 1. Build path from Source -> Root
            path_source = []
            curr = source_frame
            loop_safe = 0
            while curr in self.transforms and loop_safe < 100:
                path_source.append(curr)
                curr = self.transforms[curr]["parent"]
                loop_safe += 1
            path_source.append(curr)

            # 2. Build path from Target -> Root
            path_target = []
            curr = target_frame
            loop_safe = 0
            while curr in self.transforms and loop_safe < 100:
                path_target.append(curr)
                curr = self.transforms[curr]["parent"]
                loop_safe += 1
            path_target.append(curr)

            # 3. Find Common Ancestor
            common = None
            for node in path_source:
                if node in path_target:
                    common = node
                    break

            if common is None:
                return None, None

            # 4. Calculate Source -> Common
            mat_source_to_common = np.eye(4)
            curr = source_frame
            while curr != common:
                t_data = self.transforms[curr]
                T_step = np.eye(4)
                r = R_scipy.from_quat(t_data["rot"]).as_matrix()
                T_step[:3, :3] = r
                T_step[:3, 3] = t_data["trans"]
                mat_source_to_common = T_step @ mat_source_to_common
                curr = t_data["parent"]

            # 5. Calculate Target -> Common
            mat_target_to_common = np.eye(4)
            curr = target_frame
            while curr != common:
                t_data = self.transforms[curr]
                T_step = np.eye(4)
                r = R_scipy.from_quat(t_data["rot"]).as_matrix()
                T_step[:3, :3] = r
                T_step[:3, 3] = t_data["trans"]
                mat_target_to_common = T_step @ mat_target_to_common
                curr = t_data["parent"]

            # 6. Result: Target -> Source
            T_target_source = np.linalg.inv(mat_target_to_common) @ mat_source_to_common

            return T_target_source[:3, 3], T_target_source[:3, :3]


class ROSStream:
    def __init__(self, args=None, world_frame=None):
        if rospy.get_node_uri() is None:
            rospy.init_node("ros_stream", anonymous=True)

        def g(k, d):
            return getattr(args, k, d) if args is not None else d
        self.world_frame = world_frame or g("ros1_world_frame", "world")
        self.camera_frame = g("ros1_camera_frame", "camera_link")
        color_topic = g("ros1_color_topic", "/camera/color/image_raw")
        depth_topic = g("ros1_depth_topic", "/camera/depth/image_rect_raw")

        # Pinhole intrinsics. ROS 1 has no camera_info subscription here, so these
        # are config-driven (defaults = a generic RealSense-style camera). Override per camera
        # via ros1_fx / ros1_fy / ros1_cx / ros1_cy in the dataset config.
        self.fx = float(g("ros1_fx", 372.4634094238281))
        self.fy = float(g("ros1_fy", 371.5072021484375))
        self.cx = float(g("ros1_cx", 315.9902073557914))
        self.cy = float(g("ros1_cy", 254.59437002701452))

        # ROS (X-Forward, Z-Up) -> CV (Z-Forward, Y-Down)
        self.ros_to_cv_matrix = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

        self.tf_listener = PurePythonTFListener()
        self.color_sub = message_filters.Subscriber(color_topic, RosImage)
        self.depth_sub = message_filters.Subscriber(depth_topic, RosImage)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)

        self.latest_data = None
        self.lock = threading.Lock()
        self.has_received_first_frame = False
        self.last_message_time = time.time()

        print("ROSStream initialized. Waiting for topics...")

    def callback(self, color_msg, depth_msg):
        with self.lock:
            self.has_received_first_frame = True
            self.last_message_time = time.time()
            if self.camera_frame is None:
                self.camera_frame = color_msg.header.frame_id

            self.latest_data = {
                "color": color_msg,
                "depth": depth_msg,
                "timestamp": color_msg.header.stamp,
            }

    def manual_ros_to_cv2(self, msg, encoding):
        """Pure Python replacement for cv_bridge"""
        dtype_map = {
            "rgb8": np.uint8,
            "bgr8": np.uint8,
            "mono8": np.uint8,
            "16UC1": np.uint16,
            "32FC1": np.float32,
        }

        # Fallback for depth if encoding string varies
        if encoding not in dtype_map:
            if "32F" in encoding:
                dtype = np.float32
            elif "16U" in encoding:
                dtype = np.uint16
            else:
                dtype = np.uint8
        else:
            dtype = dtype_map[encoding]

        img_arr = np.frombuffer(msg.data, dtype=dtype)

        channels = 1
        if "rgb" in encoding or "bgr" in encoding:
            channels = 3

        try:
            if channels > 1:
                img_arr = img_arr.reshape((msg.height, msg.width, channels))
            else:
                img_arr = img_arr.reshape((msg.height, msg.width))
        except ValueError:
            img_arr = img_arr.reshape((msg.height, msg.width, -1))

        return img_arr

    def get_next_frame(self, frame_id):
        local_data = None
        with self.lock:
            if self.latest_data is None:
                if frame_id == 0 and int(time.time()) % 50 == 0:
                    print("ROSStream: Waiting for images...")
                return None
            local_data = self.latest_data

        if self.camera_frame is None:
            return None

        try:
            trans, rot_matrix = self.tf_listener.lookup_transform(
                self.world_frame, self.camera_frame
            )
        except Exception:
            return None

        if trans is None:
            if int(time.time()) % 50 == 0:
                print(f"ROSStream: Waiting for transform {self.world_frame} -> {self.camera_frame}")
            return None

        try:
            raw_rgb = self.manual_ros_to_cv2(local_data["color"], "rgb8")
            image_pil = Image.fromarray(raw_rgb)

            d_msg = local_data["depth"]
            raw_depth = self.manual_ros_to_cv2(d_msg, d_msg.encoding)

            if "32FC" in d_msg.encoding:
                raw_depth = np.nan_to_num(raw_depth)  # already meters
                depth_pil = Image.fromarray(raw_depth, mode="F")
            else:
                raw_depth = raw_depth.astype(np.float32) / 1000.0  # 16UC1 mm -> m
                depth_pil = Image.fromarray(raw_depth, mode="F")

        except Exception as e:
            print(f"Image Processing Error: {e}")
            return None

        h, w = raw_rgb.shape[:2]
        fov_x_rad = 2 * np.arctan(w / (2 * self.fx))
        fov_y_rad = 2 * np.arctan(h / (2 * self.fy))

        c2w = np.eye(4)
        c2w[:3, :3] = rot_matrix
        c2w[:3, 3] = trans
        w2c = np.linalg.inv(c2w)
        R_w2c = np.transpose(w2c[:3, :3])
        T_w2c = w2c[:3, 3]

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
            image_name=f"ros_{frame_id}",
            pose_gt=np.eye(4),
            cx=self.cx,
            cy=self.cy,
            timestamp=local_data["timestamp"].to_sec(),
            depth_scale=1000.0,
            image_path="ros",
            depth_path="ros",
        )

    def release(self):
        pass
