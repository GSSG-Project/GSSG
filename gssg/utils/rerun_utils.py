import colorsys
import datetime

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

RECORDING_ID = "gssg-live"

# Per-process flag, set once this process has initialized a rerun recording.
# Log helpers and viz-only paths early-out via is_enabled() so visualize=False
# costs nothing.
_ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def set_enabled(value: bool) -> None:
    global _ENABLED
    _ENABLED = bool(value)


def rerun_init(dataset_name):
    # Shared recording_id so subprocesses log into the same recording as the
    # main process (where the blueprint is set up).
    rr.init(dataset_name or "rerun_example", recording_id=RECORDING_ID)
    # Larger server buffer than the 1 GiB default: a full SDK channel blocks the
    # logging (SLAM) thread, so extra buffer absorbs bursts.
    rr.spawn(memory_limit="50%", server_memory_limit="4GiB")
    rr.set_time("time", sequence=0)
    rr.log("log/status", rr.TextLog("Application started.", level=rr.TextLogLevel.INFO))

    bp = rrb.Blueprint(
        rrb.Grid(
            rrb.Tabs(
                rrb.Spatial2DView(origin="/scene/frame", name="RGB Frame"),
                rrb.Spatial2DView(
                    origin="/sampling/smoothed_objectness", name="Smoothed Objectness Map"
                ),
            ),
            rrb.Tabs(
                rrb.Spatial2DView(origin="/scene/semantic_map", name="Semantic Map"),
                rrb.Spatial2DView(origin="occupancy", name="Semantic Map"),
            ),
            rrb.Tabs(
                rrb.Spatial2DView(origin="/sampling/density_error", name="Sampling Density Error"),
                rrb.Spatial2DView(origin="/scene/objectness_map", name="Objectness Map"),
                rrb.TextLogView(origin="/log", name="Logs"),
            ),
            rrb.Tabs(
                rrb.Spatial3DView(origin="/world/map/point_cloud", name="Point Cloud"),
                rrb.Spatial2DView(
                    origin="/sampling/density_normal", name="Sampling Density Normal"
                ),
                rrb.TimeSeriesView(origin="/sampling/num_selected", name="Total Sample Count"),
            ),
        ),
        collapse_panels=False,
    )

    rr.send_blueprint(bp, make_active=True)
    set_enabled(True)


def rerun_log_frame(frame_id, image):
    if not _ENABLED:
        return
    rr.set_time("time", sequence=frame_id)
    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    rr.log(
        "log/status", rr.TextLog(f"[{now}] Processing item {frame_id}.", level=rr.TextLogLevel.INFO)
    )
    rr.log("/scene/frame", rr.Image(image))


def rerun_log_gaussians(mapper):
    if not _ENABLED:
        return
    entity_path = "/world/map/point_cloud"
    pointcloud_xyz = torch.cat([mapper.active_gaussians._xyz, mapper.stable_gaussians._xyz])
    pointcloud_sem = torch.cat(
        [mapper.active_gaussians._semantic, mapper.stable_gaussians._semantic]
    )
    pointcloud_scale = torch.cat(
        [mapper.active_gaussians._scaling, mapper.stable_gaussians._scaling]
    )
    if len(pointcloud_xyz) == 0:
        rr.log(entity_path, rr.Clear(recursive=False))
        return
    xyz = pointcloud_xyz.detach().cpu()
    semantics = pointcloud_sem.detach().cpu().squeeze().clone()
    mask = semantics != 0
    filtered_points = xyz[mask]
    points = filtered_points.numpy()

    semantic_ids = semantics[mask]
    colors = map_ids_to_colors(semantic_ids)
    radii = torch.max(torch.exp(pointcloud_scale.detach()), dim=1)[0]
    radii = radii.cpu().numpy()

    rr.log(entity_path, rr.Points3D(positions=points, colors=colors, radii=radii))

    unique_ids = torch.unique(semantics)

    label_positions = []
    label_texts = []

    for semantic_id_tensor in unique_ids:
        semantic_id = semantic_id_tensor.item()
        if semantic_id == 0:  # background
            continue

        obj = mapper.scene_graph.get_object_by_id(semantic_id)

        if obj:
            if obj.center is not None:
                label_positions.append(obj.center.tolist())
                label_texts.append(int(semantic_id))

    if label_texts:
        rr.log(
            f"{entity_path}/labels",
            rr.Points3D(
                positions=label_positions,
                colors=[(250, 0, 0) for i in label_positions],
                radii=0.005,
                labels=label_texts,
            ),
        )


_COLOR_MAP_CACHE = np.empty((0, 3), dtype=np.uint8)


def get_dynamic_color_map(num_colors: int) -> np.ndarray:
    # Color i depends only on i, so grow the palette as the max semantic id
    # climbs rather than rebuilding it every frame.
    global _COLOR_MAP_CACHE
    if num_colors <= len(_COLOR_MAP_CACHE):
        return _COLOR_MAP_CACHE[:num_colors]
    extra = []
    for i in range(len(_COLOR_MAP_CACHE), num_colors):
        hue = (i * 0.618033988749895) % 1.0  # golden-ratio conjugate spreads hues
        rgb_float = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        extra.append([int(c * 255) for c in rgb_float])
    _COLOR_MAP_CACHE = np.concatenate([_COLOR_MAP_CACHE, np.array(extra, dtype=np.uint8)])
    return _COLOR_MAP_CACHE[:num_colors]


def map_ids_to_colors(ids_tensor: torch.Tensor) -> np.ndarray:
    if ids_tensor is None or ids_tensor.numel() == 0:
        return np.empty((0, 3), dtype=np.uint8)

    ids_np = ids_tensor.squeeze().cpu().numpy().astype(int)
    max_id = ids_np.max()

    color_palette = get_dynamic_color_map(max_id + 1)
    ids_np = np.atleast_1d(ids_np)
    colors = np.full((len(ids_np), 3), [50, 50, 50], dtype=np.uint8)
    non_zero_mask = ids_np != 0
    if np.any(non_zero_mask):
        colors[non_zero_mask] = color_palette[ids_np[non_zero_mask] % len(color_palette)]

    return colors
