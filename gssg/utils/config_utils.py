import os

import yaml


class GroupParams:
    pass


_QUALITY_FILES = {
    "low": "configs/quality/low.yaml",
    "mid": "configs/quality/mid.yaml",
    "mid_bench": "configs/quality/mid_bench.yaml",  # = mid but ViT-H-14 (OpenLex3D-scorable)
    "high": "configs/quality/high.yaml",
    "live": "configs/quality/live.yaml",  # real-time budget + submapping (ROS streams)
    "replay": "configs/quality/replay.yaml",  # offline bag/SVO replay: live minus realtime caps
}


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_config(config_path, quality=None):
    if quality is not None and quality not in _QUALITY_FILES:
        raise ValueError(f"--quality must be one of {list(_QUALITY_FILES)}, got {quality!r}")
    qpath = _QUALITY_FILES[quality] if quality is not None else None

    base_config = _load(config_path)
    quality_swapped = False
    while base_config["parent"] != "None":
        # --quality swaps the FIRST quality-preset ancestor, keeping the dataset/scene
        # layers above it intact (presets may parent other presets, e.g. replay -> live).
        parent_path = base_config["parent"]
        if qpath is not None and not quality_swapped and parent_path.startswith("configs/quality/"):
            parent_path = qpath
            quality_swapped = True
        if not os.path.exists(parent_path):
            raise FileNotFoundError(
                f"config parent not found: {parent_path!r} (chain from {config_path!r})"
            )
        parent_config = _load(parent_path)
        parent_config_path = parent_config["parent"]
        parent_config.update(base_config)
        base_config = parent_config
        base_config["parent"] = parent_config_path

    group = GroupParams()
    for k, v in base_config.items():
        setattr(group, k.lstrip("_"), v)
    return group
