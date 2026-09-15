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

from PIL import Image

from gssg.dataset_reader.cameras import Camera
from gssg.utils.general_utils import PILtoTorch

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    preload = args.preload
    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_image_depth = PILtoTorch(cam_info.depth, resolution, Image.NEAREST)
    gt_image = resized_image_rgb[:3, ...]
    gt_depth = resized_image_depth
    loaded_mask = None
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        depth=gt_depth,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        gpu_device=args.gpu_device,
        pose_gt=cam_info.pose_gt,
        cx=cam_info.cx / resolution_scale,
        cy=cam_info.cy / resolution_scale,
        timestamp=cam_info.timestamp,
        preload=preload,
        depth_scale=cam_info.depth_scale,
    )
