import logging

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from gssg.utils.general_utils import devB, devF
from gssg.utils.loss_utils import l1_loss, l2_loss, ssim
from gssg.utils.rerun_utils import is_enabled as _rr_enabled

TOTAL_SAMPLE_COUNT = 0

# Objectness smoothing kernel (used by both objectness-weighted sampling and the
# objectness-weighted loss). Kept here as the single source of truth so the two
# paths can never diverge.
_OBJ_BLUR_KERNEL = 31
_OBJ_BLUR_SIGMA = 20

# Objectness sampling weights w(u) = M_conf(u)^gamma + beta, shared by the
# sampling and loss paths (same single-source-of-truth rationale as the kernel
# above). gamma=0 reduces to uniform sampling. Overridden at startup from the
# config (sample_gamma / sample_beta) via configure_objectness_weights().
_OBJ_GAMMA = 3.0
_OBJ_BETA = 0.3


def configure_objectness_weights(gamma, beta):
    global _OBJ_GAMMA, _OBJ_BETA
    _OBJ_GAMMA = float(gamma)
    _OBJ_BETA = float(beta)
    print(f"[sampling] objectness weights: gamma={_OBJ_GAMMA} beta={_OBJ_BETA}")


def smooth_objectness(objectness_map, device):
    """Gaussian-blur the per-pixel objectness map on ``device``.

    ``objectness_map`` (shape [H, W] or [H, W, 1]) is moved to ``device`` before
    the conv (much cheaper than blurring on CPU). Returns a [H, W] tensor on
    ``device``.
    """
    H, W = objectness_map.shape[:2]
    om = objectness_map.to(device, non_blocking=True).reshape(H, W)
    return TF.gaussian_blur(
        om.unsqueeze(0).unsqueeze(0),
        kernel_size=_OBJ_BLUR_KERNEL,
        sigma=_OBJ_BLUR_SIGMA,
    ).reshape(H, W)


def loss_update_v2(
    mapper, render_output, image_input, init_stat, update_args, render_mask=None, unstable=True
):
    pointcloud = mapper.active_gaussians if unstable else mapper.stable_gaussians
    opacity = pointcloud.opacity_activation(init_stat["opacity"])
    device = render_output["render"].device

    attach_mask = (opacity < 0.9).squeeze()
    attach_loss = torch.tensor(0, device=device)
    if attach_mask.sum() > 0:
        attach_loss = 1000 * (
            l2_loss(pointcloud._scaling[attach_mask], init_stat["scaling"][attach_mask])
            + l2_loss(pointcloud._xyz[attach_mask], init_stat["xyz"][attach_mask])
            + l2_loss(pointcloud._rotation[attach_mask], init_stat["rotation_raw"][attach_mask])
        )

    image = render_output["render"].permute(1, 2, 0)
    depth = render_output["depth"].permute(1, 2, 0)
    normal = render_output["normal"].permute(1, 2, 0)
    depth_index = render_output["depth_index_map"].permute(1, 2, 0)

    objectness_map = image_input.get("objectness_map")

    # gamma=0 -> constant weights = plain L1, skip the blur
    if objectness_map is not None and _OBJ_GAMMA > 0:
        H, W = objectness_map.shape[:2]
        smoothed_objectness = smooth_objectness(objectness_map, device)

        scores = torch.clamp(smoothed_objectness.flatten(), 0.0, 1.0)

        weights_flat = torch.pow(scores, _OBJ_GAMMA) + _OBJ_BETA
        loss_weights = weights_flat.view(H, W)

    else:
        loss_weights = torch.ones_like(image_input["depth_map"].squeeze(-1))

    ssim_loss = torch.tensor(0, device=device)
    normal_loss = torch.tensor(0, device=device)
    depth_loss = torch.tensor(0, device=device)

    if render_mask is None:
        render_mask = torch.ones(image.shape[:2], dtype=torch.bool, device=device)
        ssim_loss = 1 - ssim(image.permute(2, 0, 1), image_input["color_map"].permute(2, 0, 1))
    else:
        render_mask = render_mask.bool()

    loss_weights = loss_weights.to(render_mask.device)

    pixel_color_errors = torch.abs(image - image_input["color_map"]).sum(dim=-1)
    pixel_color_weights = loss_weights[render_mask]
    color_loss = (pixel_color_errors[render_mask] * pixel_color_weights).sum() / (
        pixel_color_weights.sum() + 1e-8
    )

    if depth is not None and update_args.depth_weight > 0:
        depth_error = torch.abs(depth - image_input["depth_map"])
        valid_depth_mask = (
            (depth_index != -1).squeeze()
            & (image_input["depth_map"] > 0).squeeze()
            & (depth_error < mapper.add_depth_thres).squeeze()
            & render_mask
        )
        pixel_depth_errors = depth_error.squeeze(-1)[valid_depth_mask]
        pixel_depth_weights = loss_weights[valid_depth_mask]
        depth_loss = (pixel_depth_errors * pixel_depth_weights).sum() / (
            pixel_depth_weights.sum() + 1e-8
        )

    if normal is not None and update_args.normal_weight > 0:
        cos_dist = 1 - F.cosine_similarity(normal, image_input["normal_map"], dim=-1)
        valid_normal_mask = (
            render_mask
            & (depth_index != -1).squeeze()
            & ~(image_input["normal_map"] == 0).all(dim=-1)
        )
        pixel_normal_errors = cos_dist[valid_normal_mask]
        pixel_normal_weights = loss_weights[valid_normal_mask]
        normal_loss = (pixel_normal_errors * pixel_normal_weights).sum() / (
            pixel_normal_weights.sum() + 1e-8
        )

    total_loss = (
        update_args.depth_weight * depth_loss
        + update_args.normal_weight * normal_loss
        + update_args.color_weight * color_loss
        + update_args.ssim_weight * ssim_loss
    )

    (total_loss + attach_loss).backward()
    mapper.optimizer.step()

    grad_mask = (pointcloud._features_dc.grad.abs() != 0).any(dim=-1)
    pointcloud._confidence[grad_mask] += 1

    report_losses = {
        "total_loss": total_loss.item(),
        "depth_loss": depth_loss.item(),
        "ssim_loss": ssim_loss.item(),
        "normal_loss": normal_loss.item(),
        "color_loss": color_loss.item(),
        "scale_loss": attach_loss.item(),
    }

    mapper.train_report(mapper.get_total_iter, report_losses)
    mapper.optimizer.zero_grad(set_to_none=True)

    return total_loss, report_losses


def loss_update_v1(
    mapper, render_output, image_input, init_stat, update_args, render_mask=None, unstable=True
):
    pointcloud = mapper.active_gaussians if unstable else mapper.stable_gaussians
    opacity = pointcloud.opacity_activation(init_stat["opacity"])

    attach_mask = (opacity < 0.9).squeeze()
    attach_loss = torch.tensor(0)

    if attach_mask.sum() > 0:
        attach_loss = 1000 * (
            l2_loss(pointcloud._scaling[attach_mask], init_stat["scaling"][attach_mask])
            + l2_loss(pointcloud._xyz[attach_mask], init_stat["xyz"][attach_mask])
            + l2_loss(pointcloud._rotation[attach_mask], init_stat["rotation_raw"][attach_mask])
        )

    image = render_output["render"].permute(1, 2, 0)
    depth = render_output["depth"].permute(1, 2, 0)
    normal = render_output["normal"].permute(1, 2, 0)
    depth_index = render_output["depth_index_map"].permute(1, 2, 0)

    ssim_loss = devF(torch.tensor(0))
    normal_loss = devF(torch.tensor(0))
    depth_loss = devF(torch.tensor(0))

    if render_mask is None:
        render_mask = devB(torch.ones(image.shape[:2]))
        ssim_loss = 1 - ssim(image.permute(2, 0, 1), image_input["color_map"].permute(2, 0, 1))
    else:
        render_mask = render_mask.bool()

    if mapper.dataset_type == "Scannetpp":
        render_mask = render_mask & (image_input["depth_map"] > 0).squeeze()

    color_loss = l1_loss(image[render_mask], image_input["color_map"][render_mask])

    if depth is not None and update_args.depth_weight > 0:
        depth_error = depth - image_input["depth_map"]
        valid_depth_mask = (
            (depth_index != -1).squeeze()
            & (image_input["depth_map"] > 0).squeeze()
            & (depth_error < mapper.add_depth_thres).squeeze()
            & render_mask
        )
        depth_loss = torch.abs(depth_error[valid_depth_mask]).mean()

    if normal is not None and update_args.normal_weight > 0:
        cos_dist = 1 - F.cosine_similarity(normal, image_input["normal_map"], dim=-1)
        valid_normal_mask = (
            render_mask
            & (depth_index != -1).squeeze()
            & ~(image_input["normal_map"] == 0).all(dim=-1)
        )
        normal_loss = cos_dist[valid_normal_mask].mean()

    total_loss = (
        update_args.depth_weight * depth_loss
        + update_args.normal_weight * normal_loss
        + update_args.color_weight * color_loss
        + update_args.ssim_weight * ssim_loss
    )

    loss = total_loss
    (loss + attach_loss).backward()
    mapper.optimizer.step()

    grad_mask = (pointcloud._features_dc.grad.abs() != 0).any(dim=-1)
    pointcloud._confidence[grad_mask] += 1

    report_losses = {
        "total_loss": total_loss.item(),
        "depth_loss": depth_loss.item(),
        "ssim_loss": ssim_loss.item(),
        "normal_loss": normal_loss.item(),
        "color_loss": color_loss.item(),
        "scale_loss": attach_loss.item(),
    }

    mapper.train_report(mapper.get_total_iter, report_losses)
    mapper.optimizer.zero_grad(set_to_none=True)

    return loss, report_losses


def sample_pixels_v1(
    vertex_map,
    normal_map,
    color_map,
    uniform_sample_num,
    select_mask=None,
    semantic_map=None,
    objectness_map=None,
    is_error_sampling=False,
    name=None,
    smoothed_objectness=None,  # unused in v1; accepted so callers can pass it uniformly
):
    global TOTAL_SAMPLE_COUNT
    assert uniform_sample_num >= 0
    if uniform_sample_num == 0:
        return (
            devF(torch.empty(0)),
            devF(torch.empty(0)),
            devF(torch.empty(0)),
            devF(torch.empty(0)) if semantic_map is not None else None,
        )

    H, W = vertex_map.shape[0], vertex_map.shape[1]
    coord_y, coord_x = torch.meshgrid(
        torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij"
    )
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()

    if select_mask is None:
        select_mask = devB(torch.ones([H, W, 1]))
    invalid_normal_mask = torch.where(normal_map.sum(dim=-1) == 0)
    select_mask[invalid_normal_mask] = False
    if uniform_sample_num > select_mask.sum():
        uniform_sample_num = select_mask.sum()

    select_mask = select_mask.flatten()
    vertexs = vertex_map.view(-1, 3)[select_mask]
    colors = color_map.view(-1, 3)[select_mask]
    normals = normal_map.view(-1, 3)[select_mask]

    if semantic_map is not None:
        semantic_map = semantic_map.to(select_mask.device)
        semantic_ids = semantic_map.view(-1, 1)[select_mask]
    else:
        return (
            devF(torch.empty(0)),
            devF(torch.empty(0)),
            devF(torch.empty(0)),
            devF(torch.empty(0)) if semantic_map is not None else None,
        )

    valid_indices = torch.where(select_mask)[0]
    samples = torch.randperm(valid_indices.shape[0])[:uniform_sample_num]
    sampled_valid_indices = valid_indices[samples]
    TOTAL_SAMPLE_COUNT += samples.numel()

    points_uniform = vertexs[samples]
    colors_uniform = colors[samples]
    normals_uniform = normals[samples]

    logging.debug(f"Sampling Function: {name}, Samples: {samples.numel()}")

    if _rr_enabled():
        rr.log("/sampling/num_selected", rr.Scalars(TOTAL_SAMPLE_COUNT))
        sampling_mask = torch.zeros(H * W, dtype=torch.bool, device=vertex_map.device)
        sampling_mask[sampled_valid_indices] = True
        mask = sampling_mask.view(H, W).cpu().numpy().astype(np.uint8) * 255
        sampling_mask_image = np.stack(
            [mask, mask * (not is_error_sampling), mask * (not is_error_sampling)], axis=-1
        )
        channel = "/sampling/density_error" if is_error_sampling else "/sampling/density_normal"
        rr.log(channel, rr.Image(sampling_mask_image))

    if semantic_map is not None:
        semantic_uniform = semantic_ids[samples]
        return (
            points_uniform.view(uniform_sample_num, 3),
            normals_uniform.view(uniform_sample_num, 3),
            colors_uniform.view(uniform_sample_num, 3),
            semantic_uniform.view(uniform_sample_num, 1),
        )
    else:
        return (
            points_uniform.view(uniform_sample_num, 3),
            normals_uniform.view(uniform_sample_num, 3),
            colors_uniform.view(uniform_sample_num, 3),
        )


def sample_pixels_v2(
    vertex_map,
    normal_map,
    color_map,
    uniform_sample_num,
    select_mask=None,
    semantic_map=None,
    objectness_map=None,
    is_error_sampling=False,
    name=None,
    smoothed_objectness=None,
):
    global TOTAL_SAMPLE_COUNT
    assert uniform_sample_num >= 0
    if uniform_sample_num == 0:
        return (
            devF(torch.empty(0)),
            devF(torch.empty(0)),
            devF(torch.empty(0)),
            devF(torch.empty(0)) if semantic_map is not None else None,
        )

    H, W = vertex_map.shape[0], vertex_map.shape[1]
    device = vertex_map.device

    if select_mask is None:
        select_mask = torch.ones([H, W, 1], dtype=torch.bool, device=device)

    invalid_normal_mask = normal_map.sum(dim=-1) == 0
    select_mask[invalid_normal_mask] = False

    num_selectable = select_mask.sum()
    final_sample_num = min(int(uniform_sample_num), int(num_selectable))

    select_mask_flat = select_mask.flatten()
    valid_indices = torch.where(select_mask_flat)[0]

    if valid_indices.shape[0] == 0 or final_sample_num <= 0:
        return (
            devF(torch.empty(0)),
            devF(torch.empty(0)),
            devF(torch.empty(0)),
            devF(torch.empty(0)) if semantic_map is not None else None,
        )

    vertexs = vertex_map.view(-1, 3)[select_mask_flat]
    colors = color_map.view(-1, 3)[select_mask_flat]
    normals = normal_map.view(-1, 3)[select_mask_flat]

    if semantic_map is not None:
        semantic_map = semantic_map.to(device)
        semantic_ids = semantic_map.view(-1, 1)[select_mask_flat]
    else:
        semantic_ids = None
    # gamma=0 -> weights are constant, skip the blur/multinomial and sample uniformly
    if objectness_map is not None and _OBJ_GAMMA > 0:
        # The blur depends only on the per-frame objectness map; callers may pass
        # a precomputed one to share it across sampling calls in a frame.
        if smoothed_objectness is None:
            smoothed_objectness = smooth_objectness(objectness_map, device)

        if _rr_enabled():
            rr.log("/sampling/smoothed_objectness", rr.Image(smoothed_objectness.cpu().numpy()))

        valid_scores = smoothed_objectness.flatten()[select_mask_flat]

        scores = torch.clamp(valid_scores, 0.0, 1.0)

        # _OBJ_GAMMA: higher (3-4) sharpens focus on objects, lower (1-2) is more
        # gradual; _OBJ_BETA: minimum sampling importance given to background pixels.
        weights = torch.pow(scores, _OBJ_GAMMA) + _OBJ_BETA
        samples = torch.multinomial(weights, final_sample_num, replacement=False)
    else:
        samples = torch.randperm(valid_indices.shape[0], device=device)[:final_sample_num]

    logging.debug(f"Sampling Function: {name}, Samples: {samples.numel()}")
    TOTAL_SAMPLE_COUNT += samples.numel()

    sampled_valid_indices = valid_indices[samples]

    if _rr_enabled():
        rr.log("/sampling/num_selected", rr.Scalars(TOTAL_SAMPLE_COUNT))
        sampling_mask = torch.zeros(H * W, dtype=torch.bool, device=device)
        sampling_mask[sampled_valid_indices] = True
        mask = sampling_mask.view(H, W).cpu().numpy().astype(np.uint8) * 255
        color_mask = np.stack(
            [
                mask if is_error_sampling else np.zeros_like(mask),
                mask if not is_error_sampling else np.zeros_like(mask),
                np.zeros_like(mask),
            ],
            axis=-1,
        )
        channel = "/sampling/density_error" if is_error_sampling else "/sampling/density_normal"
        rr.log(channel, rr.Image(color_mask))

    points_uniform = vertexs[samples]
    colors_uniform = colors[samples]
    normals_uniform = normals[samples]

    if semantic_map is not None and semantic_ids is not None:
        semantic_uniform = semantic_ids[samples]
        return (
            points_uniform.view(-1, 3),
            normals_uniform.view(-1, 3),
            colors_uniform.view(-1, 3),
            semantic_uniform.view(-1, 1),
        )
    else:
        return (
            points_uniform.view(-1, 3),
            normals_uniform.view(-1, 3),
            colors_uniform.view(-1, 3),
        )
