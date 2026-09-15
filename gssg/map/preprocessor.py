import torch

from gssg.utils.utils import (
    bilateralFilter_torch,
    compute_confidence_map,
    compute_normal_map,
    compute_vertex_map,
)


def preprocess(perception, frame, frame_id):
    # [H, W, C], the image is scaled by 255 in function "PILtoTorch"
    depth_map, color_map = (
        frame.original_depth.permute(1, 2, 0) * 255,
        frame.original_image.permute(1, 2, 0),
    )

    intrinsic = frame.get_intrinsic
    if perception.depth_filter:
        depth_map_filter = bilateralFilter_torch(depth_map, 5, 2, 2)
    else:
        depth_map_filter = depth_map

    valid_range_mask = (depth_map_filter > perception.min_depth) & (
        depth_map_filter < perception.max_depth
    )
    depth_map_filter[~valid_range_mask] = 0.0
    frame.original_depth = depth_map_filter.permute(2, 0, 1) / 255.0
    vertex_map_c = compute_vertex_map(depth_map_filter, intrinsic)
    normal_map_c = compute_normal_map(vertex_map_c)
    confidence_map = compute_confidence_map(normal_map_c, intrinsic)
    if perception.semantic_encoder is not None:
        semantic_map, semantic_results, objectness_map, frame_embedding = (
            get_segmentation_and_semantic_map(
                perception.semantic_encoder,
                frame.original_image,
                frame_id,
                perception.visualize,
            )
        )
    else:
        # Semantics off: every new Gaussian gets id 0; no objectness/embedding.
        H, W = frame.original_image.shape[1:]
        semantic_map = torch.zeros((H, W, 1), dtype=torch.int32)
        semantic_results, objectness_map, frame_embedding = {}, None, None

    # confidence_threshold tum: 0.5, others: 0.2
    invalid_confidence_mask = (normal_map_c == 0).all(dim=-1) | (
        confidence_map < perception.invalid_confidence_thresh
    )[..., 0]

    depth_map_filter[invalid_confidence_mask] = 0
    normal_map_c[invalid_confidence_mask] = 0
    vertex_map_c[invalid_confidence_mask] = 0
    confidence_map[invalid_confidence_mask] = 0

    perception.update_curr_status(
        frame,
        frame_id,
        depth_map,
        depth_map_filter,
        vertex_map_c,
        normal_map_c,
        color_map,
        semantic_map,
    )

    frame_map = {}
    frame_map["depth_map"] = depth_map_filter
    frame_map["color_map"] = color_map
    frame_map["normal_map_c"] = normal_map_c
    frame_map["vertex_map_c"] = vertex_map_c
    frame_map["confidence_map"] = confidence_map
    frame_map["invalid_confidence_mask"] = invalid_confidence_mask
    frame_map["time"] = frame_id
    frame_map["semantic_map"] = semantic_map
    frame_map["semantic_results"] = semantic_results
    frame_map["objectness_map"] = objectness_map
    frame_map["frame_embedding"] = frame_embedding

    return frame_map


def get_segmentation_and_semantic_map(semantic_encoder, frame, frame_id, visualize):
    return semantic_encoder.get_map(frame, visualize)
