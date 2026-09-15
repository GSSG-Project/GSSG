"""Render an MP4 of a finished run: the saved stable Gaussians rendered along the
recorded trajectory.

  ./bin/gssg-render output/replica_room0
  ./bin/gssg-render replica_room0 --fps 30 --speed 2.0
  ./bin/gssg-render replica_room0 --fx 640 --fy 640 --cx 640 --cy 360 --width 1280 --height 720

Poses come from <save_path>/trajectory.csv (c2w, wxyz quaternions, one row per mapped
frame) and are resampled at a uniform timestep in recorded-stamp time (slerp + lerp),
so the video plays back the traverse in real time (scaled by --speed). Intrinsics come
from manifest.json's "camera" block (written by current runs), or CLI flags for older
runs. Renderer/Gaussian settings are read from the run's resolved config.yaml.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

from gssg.dataset_reader.cameras import Camera
from gssg.map.gaussian_pointcloud import GaussianPointCloud
from gssg.map.render import Renderer
from gssg.utils.config_utils import read_config
from gssg.utils.graphics_utils import focal2fov


class _StubSceneGraph:
    def update_object_geometry(self, *args, **kwargs):
        pass

    def get_color_id(self, sid):
        return sid


def resolve_save_path(name):
    for p in (name, os.path.join("output", name)):
        if os.path.isdir(p):
            return p
    sys.exit(f"[gssg-render] no such run directory: {name}")


def load_manifest(save_path):
    path = os.path.join(save_path, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_ply(save_path, manifest, override):
    if override:
        return override
    rel = manifest.get("stable_ply")
    if rel and os.path.exists(os.path.join(save_path, rel)):
        return os.path.join(save_path, rel)
    import glob

    cands = sorted(glob.glob(os.path.join(save_path, "save_model", "frame_*", "iter_*_stable.ply")))
    if not cands:
        sys.exit(f"[gssg-render] no stable PLY found under {save_path}/save_model")
    return cands[-1]


def resolve_intrinsics(cli, manifest, source_path):
    if all(v is not None for v in (cli.width, cli.height, cli.fx, cli.fy)):
        cx = cli.cx if cli.cx is not None else cli.width / 2.0
        cy = cli.cy if cli.cy is not None else cli.height / 2.0
        return cli.width, cli.height, cli.fx, cli.fy, cx, cy
    cam = manifest.get("camera")
    if cam:
        return cam["width"], cam["height"], cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    # Replica-style dataset layout keeps a cam_params.json next to (or above) the scenes.
    for d in (source_path, os.path.dirname(source_path or "")):
        p = os.path.join(d, "cam_params.json") if d else ""
        if p and os.path.exists(p):
            with open(p) as f:
                c = json.load(f)["camera"]
            return c["w"], c["h"], c["fx"], c["fy"], c["cx"], c["cy"]
    sys.exit(
        "[gssg-render] no intrinsics: manifest has no 'camera' block (older run) and no "
        "cam_params.json found. Pass --width --height --fx --fy [--cx --cy]."
    )


def read_trajectory(path):
    """Return (stamps[N], c2w[N,4,4]) from trajectory.csv, dropping non-increasing stamps."""
    stamps, poses = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            stamp = float(row["stamp"])
            if stamps and stamp <= stamps[-1]:
                continue
            q = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
            t = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
            c2w = np.eye(4)
            c2w[:3, :3] = quat_wxyz_to_matrix(q)
            c2w[:3, 3] = t
            stamps.append(stamp)
            poses.append(c2w)
    if len(poses) < 2:
        sys.exit(f"[gssg-render] trajectory too short ({len(poses)} usable poses): {path}")
    return np.array(stamps), np.stack(poses)


def quat_wxyz_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp(q0, q1, alpha):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / s


def matrix_to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:
        # Rare near-180° case; fall back to the largest diagonal term.
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4.0
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
        return q / np.linalg.norm(q)
    q = np.array(
        [
            w,
            (R[2, 1] - R[1, 2]) / (4 * w),
            (R[0, 2] - R[2, 0]) / (4 * w),
            (R[1, 0] - R[0, 1]) / (4 * w),
        ]
    )
    return q / np.linalg.norm(q)


def resample_poses(stamps, poses, fps, speed):
    """Uniform video timeline over the recorded stamps: frame k sits at recorded time
    stamps[0] + k*speed/fps, with slerp/lerp between the surrounding logged poses."""
    quats = np.stack([matrix_to_quat_wxyz(p[:3, :3]) for p in poses])
    step = speed / fps
    times = np.arange(stamps[0], stamps[-1], step)
    idx = np.searchsorted(stamps, times, side="right") - 1
    idx = np.clip(idx, 0, len(stamps) - 2)
    out = []
    for t, i in zip(times, idx, strict=True):
        a = (t - stamps[i]) / max(1e-9, stamps[i + 1] - stamps[i])
        a = float(np.clip(a, 0.0, 1.0))
        c2w = np.eye(4)
        c2w[:3, :3] = quat_wxyz_to_matrix(slerp(quats[i], quats[i + 1], a))
        c2w[:3, 3] = (1 - a) * poses[i][:3, 3] + a * poses[i + 1][:3, 3]
        out.append(c2w)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("save_path", help="run directory (or its name under output/)")
    ap.add_argument(
        "--out", default=None, help="output mp4 (default <save_path>/trajectory_video.mp4)"
    )
    ap.add_argument("--ply", default=None, help="override the stable PLY to render")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed vs recorded time")
    ap.add_argument("--no-resample", action="store_true", help="one video frame per trajectory row")
    ap.add_argument("--depth", action="store_true", help="append a colormapped depth panel")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    cli = ap.parse_args()

    save_path = resolve_save_path(cli.save_path)
    manifest = load_manifest(save_path)
    config_path = os.path.join(save_path, "config.yaml")
    if not os.path.exists(config_path):
        sys.exit(f"[gssg-render] no config.yaml in {save_path} (older run?)")
    args = read_config(config_path)

    ply_path = resolve_ply(save_path, manifest, cli.ply)
    traj_path = os.path.join(save_path, "trajectory.csv")
    if not os.path.exists(traj_path):
        sys.exit(f"[gssg-render] no trajectory.csv in {save_path}")
    width, height, fx, fy, cx, cy = resolve_intrinsics(
        cli, manifest, getattr(args, "source_path", None)
    )
    out_path = cli.out or os.path.join(save_path, "trajectory_video.mp4")

    stamps, traj = read_trajectory(traj_path)
    if cli.no_resample:
        cams = list(traj)
    else:
        cams = resample_poses(stamps, traj, cli.fps, cli.speed)
    print(
        f"[gssg-render] {len(traj)} logged poses over {stamps[-1] - stamps[0]:.1f}s "
        f"-> {len(cams)} video frames @ {cli.fps:g} fps (speed x{cli.speed:g})"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpc = GaussianPointCloud(args, _StubSceneGraph(), name="render_video")
    print(f"[gssg-render] loading {ply_path}")
    gpc.load(ply_path)
    renderer = Renderer(args)
    gaussian_data = {
        "xyz": gpc.get_xyz,
        "opacity": gpc.get_opacity,
        "scales": gpc.get_scaling,
        "rotations": gpc.get_rotation,
        "shs": gpc.get_features,
        "normal": gpc.get_normal,
    }

    fov_x, fov_y = focal2fov(fx, width), focal2fov(fy, height)
    frame_w = width * 2 if cli.depth else width
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), cli.fps, (frame_w, height))
    if not writer.isOpened():
        sys.exit(f"[gssg-render] cv2.VideoWriter failed to open {out_path}")
    max_depth = float(getattr(args, "max_depth", 10.0))
    dummy_img = torch.zeros(3, height, width)

    for i, c2w in enumerate(tqdm(cams, desc="rendering")):
        w2c = np.linalg.inv(c2w)
        cam = Camera(
            colmap_id=i,
            R=np.transpose(w2c[:3, :3]),
            T=w2c[:3, 3],
            FoVx=fov_x,
            FoVy=fov_y,
            image=dummy_img,
            depth=None,
            gt_alpha_mask=None,
            image_name=f"video_{i}",
            uid=i,
            cx=cx,
            cy=cy,
            gpu_device=device,
        )
        with torch.no_grad():
            result = renderer.render(cam, gaussian_data)
        img = result["render"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        frame = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if cli.depth:
            d = result["depth"][0].clamp(0, max_depth).cpu().numpy() / max_depth
            d8 = (d * 255).astype(np.uint8)
            frame = np.hstack([frame, cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)])
        writer.write(frame)

    writer.release()
    print(f"[gssg-render] wrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
