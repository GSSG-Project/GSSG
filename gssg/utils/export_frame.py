import os

import torch

from gssg.utils.paths import repo_path


def export_frame_and_params(frame, params, output_dir):
    if output_dir is None:
        output_path = repo_path("scripts", "debug", "data.pt")
        frame_output_path = repo_path("scripts", "debug", "frame.pt")
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "data.pt")
        frame_output_path = os.path.join(output_dir, "frame.pt")
    torch.save(params, output_path)
    export_frame_data(frame, frame_output_path)


def export_frame_data(frame, path):
    data = {
        "world_view_transform": frame.world_view_transform,
        "full_proj_transform": frame.full_proj_transform,
        "camera_center": frame.camera_center,
    }
    torch.save(data, path)


def import_frame_data(path=None):
    if path is None:
        path = repo_path("scripts", "debug", "frame.pt")
    data = torch.load(path)
    return (
        data["world_view_transform"],
        data["full_proj_transform"],
        data["camera_center"],
    )
