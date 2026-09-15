import os
import time

import numpy as np
import torch
from skimage.draw import polygon
from torchvision.ops import roi_align
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import save_image


def make_square_bboxes(
    frame_tensor,
    bboxes,
    masks,
    target_size=224,
    apply_mask: bool = False,
    debug=False,
    debug_dir="debug/debug_crops",
):
    """Produce square (K, 3, target_size, target_size) crops, one per bbox.

    Args:
        frame_tensor:  HWC / CHW / BCHW float tensor (uint8 also accepted).
        bboxes:        list / tensor of (x1, y1, x2, y2) bboxes.
        masks:         list of polygon arrays in (x, y) order. Ignored unless
                       apply_mask=True.
        target_size:   output H == W.
        apply_mask:    if True, composite each crop so pixels inside the
                       segmentation polygon stay sharp and pixels outside are
                       gaussian-blurred. If False (default), returns plain
                       RoIAlign'd crops.

    Returns:
        Tensor of shape (K, 3, target_size, target_size) on the same device
        as frame_tensor. Empty Tensor of shape (0, 3, T, T) when no kept bbox.
    """
    padding = 20

    # --- 1. Normalize frame_tensor to (C, H, W) float on its device ---
    if frame_tensor.ndim == 4:
        frame_tensor = frame_tensor[0]
    if frame_tensor.shape[-1] == 3 and frame_tensor.ndim == 3:
        frame_tensor = frame_tensor.permute(2, 0, 1)
    if frame_tensor.dtype == torch.uint8:
        frame_float = frame_tensor.float() / 255.0
    else:
        frame_float = frame_tensor

    C, H, W = frame_float.shape
    device = frame_float.device

    # --- 2. Normalize bboxes list ---
    if isinstance(bboxes, torch.Tensor):
        bboxes_list = bboxes.detach().cpu().tolist()
    else:
        bboxes_list = bboxes

    # --- 3. CPU pass: square crop coords (and optional polygon raster) ---
    crop_coords = []  # list of (x1, y1, x2, y2)
    masks_at_target_np = [] if apply_mask else None

    for idx, bbox in enumerate(bboxes_list):
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(W, int(bbox[2]))
        y2 = min(H, int(bbox[3]))
        bw, bh = x2 - x1, y2 - y1
        square_size = max(bw, bh) + (padding * 2)
        if square_size <= 0:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        sq_x1 = cx - square_size // 2
        sq_y1 = cy - square_size // 2
        sq_x2 = sq_x1 + square_size
        sq_y2 = sq_y1 + square_size

        # Shift the square back into the image if it overruns.
        if sq_x1 < 0:
            sq_x2 -= sq_x1
            sq_x1 = 0
        if sq_y1 < 0:
            sq_y2 -= sq_y1
            sq_y1 = 0
        if sq_x2 > W:
            sq_x1 -= sq_x2 - W
            sq_x2 = W
        if sq_y2 > H:
            sq_y1 -= sq_y2 - H
            sq_y2 = H
        sq_x1, sq_y1 = max(0, sq_x1), max(0, sq_y1)
        sq_x2, sq_y2 = min(W, sq_x2), min(H, sq_y2)

        crop_h = sq_y2 - sq_y1
        crop_w = sq_x2 - sq_x1
        if crop_h == 0 or crop_w == 0:
            continue

        crop_coords.append((sq_x1, sq_y1, sq_x2, sq_y2))

        if apply_mask:
            poly = masks[idx]
            if poly is None or len(poly) == 0:
                masks_at_target_np.append(np.zeros((target_size, target_size), dtype=bool))
                continue
            poly = np.asarray(poly)
            if poly.ndim == 3:
                poly = poly.squeeze(1)
            scale_x = target_size / float(crop_w)
            scale_y = target_size / float(crop_h)
            poly_xs_t = (poly[:, 0].astype(np.float32) - sq_x1) * scale_x
            poly_ys_t = (poly[:, 1].astype(np.float32) - sq_y1) * scale_y
            rr, cc = polygon(poly_ys_t, poly_xs_t, shape=(target_size, target_size))
            mask_np = np.zeros((target_size, target_size), dtype=bool)
            if rr.size:
                mask_np[rr, cc] = True
            masks_at_target_np.append(mask_np)

    if not crop_coords:
        return torch.empty((0, 3, target_size, target_size), device=device, dtype=frame_float.dtype)

    # --- 4. Single batched crop+resize via RoIAlign ---
    K = len(crop_coords)
    boxes_np = np.empty((K, 5), dtype=np.float32)
    boxes_np[:, 0] = 0.0
    for i, (sx1, sy1, sx2, sy2) in enumerate(crop_coords):
        boxes_np[i, 1] = float(sx1)
        boxes_np[i, 2] = float(sy1)
        boxes_np[i, 3] = float(sx2)
        boxes_np[i, 4] = float(sy2)
    boxes = torch.from_numpy(boxes_np).to(device)
    crops_batch = roi_align(
        frame_float.unsqueeze(0),
        boxes,
        output_size=(target_size, target_size),
        spatial_scale=1.0,
        aligned=True,
    )  # (K, 3, T, T)

    # --- 5. (Optional) mask-blur composite ---
    if apply_mask:
        masks_np_stack = np.stack(masks_at_target_np, axis=0)
        masks_batch = torch.from_numpy(masks_np_stack).to(device).unsqueeze(1)
        blurred_batch = gaussian_blur(crops_batch, kernel_size=21, sigma=8.0)
        crops_batch = torch.where(masks_batch, crops_batch, blurred_batch)

    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = int(time.time())
        for i in range(crops_batch.shape[0]):
            save_image(crops_batch[i], os.path.join(debug_dir, f"{timestamp}_crop_{i}.png"))

    return crops_batch
