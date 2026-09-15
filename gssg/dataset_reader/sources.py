import glob
import json
import os

import numpy as np

from gssg.dataset_reader.dataset_readers import (
    SceneInfo,
    getNerfppNorm,
    natural_sort_key,
    readCameras,
    rot_compare,
    saveCfg,
    select_best_frame,
    trans_compare,
)

_VIEW_TRANS = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=float,
)

_REGISTRY = {}


def _register(cls):
    _REGISTRY[cls.dataset_type] = cls
    return cls


def build_source(args):
    cls = _REGISTRY.get(args.dataset_type)
    if cls is None:
        raise ValueError(f"Unknown dataset_type {args.dataset_type!r}. Known: {sorted(_REGISTRY)}")
    return cls(args)


class FrameSource:
    dataset_type = None
    is_streaming = False

    def __init__(self, args):
        self.args = args


class OfflineSource(FrameSource):
    color_glob = "results/frame*.jpg"
    depth_glob = "results/depth*.png"
    pose_layout = "single_traj"  # or "per_frame"
    flip_view = False  # OpenGL->OpenCV Y/Z flip default for this type
    natural_sort = False

    def _sorted(self, pattern):
        key = natural_sort_key if self.natural_sort else None
        return sorted(glob.glob(os.path.join(self.args.source_path, pattern)), key=key)

    def _pose_lines(self):
        if self.pose_layout == "single_traj":
            with open(os.path.join(self.args.source_path, "traj.txt")) as f:
                return f.readlines()
        files = sorted(
            glob.glob(os.path.join(self.args.source_path, "pose", "*.txt")),
            key=natural_sort_key,
        )
        return [open(p).readlines()[0].replace("\t", " ") for p in files]

    def _cam_params(self):
        path = os.path.join(self.args.source_path, "cam_params.json")
        if not os.path.exists(path):
            path = os.path.join(self.args.source_path, "..", "cam_params.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"cam_params.json not found in {self.args.source_path} or its parent"
            )
        with open(path) as f:
            return json.load(f)["camera"]

    def _flip(self):
        return getattr(self.args, "apply_view_transform", True) and self.flip_view

    def _select_indices(self, num_files, color_paths, pose_lines):
        args = self.args
        filter_blur = getattr(args, "filter_blur", False)
        radius = getattr(args, "blur_filter_radius", 5)
        if filter_blur:
            print(f"[INFO] Blur filtering enabled. Radius: {radius}")

        indices = []
        last = -1

        if not args.use_keyframe:
            for cand in range(args.frame_start, num_files, args.frame_step):
                if args.frame_max != -1 and len(indices) >= args.frame_max:
                    break
                idx = (
                    select_best_frame(cand, color_paths, num_files, last, radius)
                    if filter_blur
                    else cand
                )
                if idx <= last:
                    idx = last + 1
                    if idx >= num_files:
                        break
                indices.append(idx)
                last = idx
            print(
                f"[INFO] Loading {len(indices)} frames (Start: {args.frame_start}, "
                f"Step: {args.frame_step}, Filter Blur: {filter_blur})"
            )
            return indices

        prev_rot = prev_trans = None
        for i in range(args.frame_start, num_files):
            if args.frame_max != -1 and len(indices) >= args.frame_max:
                break
            if i <= last and indices:
                continue
            c2w = np.array(list(map(float, pose_lines[i].split()))).reshape(4, 4)
            is_kf = not indices
            if not is_kf:
                _, theta = rot_compare(prev_rot, c2w[:3, :3])
                _, l2 = trans_compare(prev_trans, c2w[:3, 3])
                is_kf = theta > args.keyframe_theta_thres or l2 > args.keyframe_trans_thres
            if is_kf:
                idx = (
                    select_best_frame(i, color_paths, num_files, last, radius) if filter_blur else i
                )
                if idx > last:
                    indices.append(idx)
                    last = idx
                    sel = np.array(list(map(float, pose_lines[idx].split()))).reshape(4, 4)
                    prev_rot, prev_trans = sel[:3, :3], sel[:3, 3]
        print(f"[INFO] Keyframe detection found {len(indices)} frames.")
        return indices

    def read_scene_info(self):
        args = self.args
        color_paths = self._sorted(self.color_glob)
        depth_paths = self._sorted(self.depth_glob)
        pose_lines = self._pose_lines()
        num_files = min(len(color_paths), len(pose_lines))
        config = self._cam_params()

        indices = self._select_indices(num_files, color_paths, pose_lines)

        view = _VIEW_TRANS if self._flip() else np.eye(4)
        poses = [
            np.array(list(map(float, pose_lines[i].split()))).reshape(4, 4) @ view for i in indices
        ]

        intrinsic = np.eye(3)
        intrinsic[0, 0], intrinsic[1, 1] = config["fx"], config["fy"]
        intrinsic[0, 2], intrinsic[1, 2] = config["cx"], config["cy"]

        cam_infos = readCameras(
            color_paths=[color_paths[i] for i in indices],
            depth_paths=[depth_paths[i] for i in indices],
            poses=poses,
            intrinsic=intrinsic,
            indices=list(range(len(poses))),
            depth_scale=config["scale"],
            timestamps=[i / 30.0 for i in indices],
        )
        saveCfg(poses, config, args.source_path)

        if args.eval:
            hold = args.eval_llff
            train = [c for i, c in enumerate(cam_infos) if (i + 1) % hold != 0]
            test = [c for i, c in enumerate(cam_infos) if (i + 1) % hold == 0]
        else:
            train, test = cam_infos, []

        scene = os.path.basename(args.source_path)
        return SceneInfo(
            point_cloud=None,
            train_cameras=train,
            test_cameras=test,
            nerf_normalization=getNerfppNorm(train),
            ply_path=None,
            mesh_path=os.path.join(args.source_path, f"{scene}.ply"),
        )


@_register
class ReplicaSource(OfflineSource):
    dataset_type = "replica"
    flip_view = False


@_register
class RealworldSource(OfflineSource):
    dataset_type = "realworld"
    flip_view = True


@_register
class RosOfflineSource(OfflineSource):
    dataset_type = "ros-offline"
    flip_view = True


@_register
class Hm3dSource(OfflineSource):
    dataset_type = "hm3d"
    color_glob = "rgb/*.png"
    depth_glob = "depth/*.png"
    pose_layout = "per_frame"
    flip_view = True
    natural_sort = True


class StreamingSource(FrameSource):
    is_streaming = True


@_register
class Ros1Source(StreamingSource):
    dataset_type = "ros"


@_register
class Ros2Source(StreamingSource):
    dataset_type = "ros2"
