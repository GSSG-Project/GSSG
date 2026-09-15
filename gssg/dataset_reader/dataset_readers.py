import json
import os
import re
import sys
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from gssg.utils.graphics_utils import (
    BasicPointCloud,
    focal2fov,
    getWorld2View2,
)


def blur_score_fft_gray(gray, size=60):
    """Sharpness score of a grayscale array (mean log-magnitude after removing the
    low-frequency FFT band); higher is sharper."""
    (h, w) = gray.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(gray)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon) + 1e-10)
    return float(np.mean(magnitude))


def detect_blur_fft(image_path, size=60):
    image = cv2.imread(image_path)
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return blur_score_fft_gray(gray, size)


_blur_cache = {}


def get_blur_score_cached(path):
    if path not in _blur_cache:
        _blur_cache[path] = detect_blur_fft(path)
    return _blur_cache[path]


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int
    depth: Image.Image
    depth_path: str
    pose_gt: np.array = np.eye(4)
    cx: float = -1
    cy: float = -1
    depth_scale: float = 1
    timestamp: float = -1


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    mesh_path: str


def rot_compare(R1, R2):
    R_diff = np.dot(R1, R2.T)
    trace = np.trace(R_diff)
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    theta = np.arccos(trace)
    return R_diff, theta


def trans_compare(T1, T2):
    diff = T1 - T2
    l2_diff = np.linalg.norm(diff)
    return diff, l2_diff


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    if not cam_centers:
        return {"translate": np.array([0.0, 0.0, 0.0]), "radius": 1.0}

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    if radius == 0:
        radius = 1
    return {"translate": translate, "radius": radius}


def saveCfg(poses, config, save_path):
    cameras = []
    for idx, c2w in enumerate(poses):
        width, height = config["w"], config["h"]

        R = c2w[:3, :3]
        T = c2w[:3, 3]
        position = T.tolist()
        rotation = [x.tolist() for x in R]

        img_name = f"frame_{idx:04d}"
        fx, fy = config["fx"], config["fy"]
        cameras.append(
            {
                "id": idx,
                "img_name": img_name,
                "width": width,
                "height": height,
                "position": position,
                "rotation": rotation,
                "fx": fx,
                "fy": fy,
            }
        )

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "cameras.json"), "w") as file:
        json.dump(cameras, file, indent=4)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]


def select_best_frame(candidate_idx, all_paths, num_files, last_chosen_idx, search_radius):
    start_search = max(last_chosen_idx + 1, candidate_idx - search_radius)
    end_search = min(num_files, candidate_idx + search_radius + 1)
    if start_search >= end_search:
        return candidate_idx

    best_idx = candidate_idx
    best_score = -1.0

    for i in range(start_search, end_search):
        score = get_blur_score_cached(all_paths[i])
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def readCameras(
    color_paths,
    depth_paths,
    poses,
    intrinsic,
    indices,
    depth_scale,
    timestamps,
    crop_edge=0,
    eval_=False,
):
    cam_infos = []

    for idx_ in range(len(indices)):
        idx = indices[idx_]

        if idx_ % 10 == 0 or idx_ == len(indices) - 1:
            sys.stdout.write(f"\r[INFO] Reading camera {idx_ + 1}/{len(indices)}")
            sys.stdout.flush()

        c2w = poses[idx]
        if np.isinf(c2w).any():
            continue

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image_color = Image.open(color_paths[idx])
        image_depth_raw = Image.open(depth_paths[idx])
        image_depth = np.asarray(image_depth_raw, dtype=np.float32) / depth_scale

        if image_color.size != image_depth_raw.size:
            image_color = image_color.resize(image_depth_raw.size)

        image_color_np = np.asarray(image_color)

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        if crop_edge > 0:
            image_color_np = image_color_np[crop_edge:-crop_edge, crop_edge:-crop_edge]
            image_depth = image_depth[crop_edge:-crop_edge, crop_edge:-crop_edge]
            cx -= crop_edge
            cy -= crop_edge

        height, width = image_color_np.shape[:2]
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)

        image_name = os.path.basename(color_paths[idx]).split(".")[0]

        cam_info = CameraInfo(
            uid=idx_,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=Image.fromarray(image_color_np),
            image_path=color_paths[idx],
            image_name=image_name,
            width=width,
            height=height,
            depth=Image.fromarray(image_depth),
            depth_path=depth_paths[idx],
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
            timestamp=timestamps[idx],
        )
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos
