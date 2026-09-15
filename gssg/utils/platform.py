"""Platform detection and segmentation-backend resolution so one config runs
unchanged across desktop (x86_64 + TensorRT) and Jetson (aarch64)."""

import logging
import os
import platform
import re
from importlib.util import find_spec
from pathlib import Path

from gssg.utils.paths import REPO_ROOT

_ENGINES_DIR = REPO_ROOT / "thirdparty" / "bestsam" / "engines"


def arch() -> str:
    return platform.machine()


def is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    model = Path("/proc/device-tree/model")
    try:
        return model.exists() and "jetson" in model.read_text(errors="ignore").lower()
    except OSError:
        return False


def _has_tensorrt() -> bool:
    return find_spec("tensorrt") is not None


def _resolution(engine: Path) -> int:
    m = re.search(r"r(\d+)", engine.name)
    return int(m.group(1)) if m else 0


def _available_engines() -> list[Path]:
    return sorted(_ENGINES_DIR.glob("*.plan")) if _ENGINES_DIR.is_dir() else []


def _pick_engine(engines: list[Path]) -> Path:
    # Jetson favors the lightest (lowest-res) engine; desktop the sharpest (highest-res).
    return min(engines, key=_resolution) if is_jetson() else max(engines, key=_resolution)


def resolve_seg_backend(seg_model: str, engine_path: str) -> tuple[str, str]:
    """Return a (seg_model, engine_path) pair guaranteed to be runnable on this machine.

    - any backend other than bestsam/auto: returned unchanged.
    - "auto": bestsam if TensorRT and an engine are present, else sam3.
    - "bestsam": keep the configured engine if it exists; otherwise heal to a present
      engine, or to sam3 if none are built / TensorRT is missing.
    """
    if seg_model not in ("bestsam", "auto"):
        return seg_model, engine_path

    if not _has_tensorrt():
        if seg_model == "bestsam":
            logging.warning("seg_model=bestsam but TensorRT is unavailable; using sam3 instead.")
        return "sam3", engine_path

    configured = Path(os.path.expanduser(engine_path)) if engine_path else None
    if seg_model == "bestsam" and configured and configured.exists():
        return "bestsam", str(configured)

    engines = _available_engines()
    if engines:
        chosen = _pick_engine(engines)
        if seg_model == "bestsam":
            logging.warning(
                "BestSAM engine %s not found; using built engine %s (%s).",
                engine_path,
                chosen.name,
                "jetson" if is_jetson() else "desktop",
            )
        return "bestsam", str(chosen)

    if seg_model == "bestsam":
        logging.warning("No BestSAM engine built under %s; using sam3 instead.", _ENGINES_DIR)
    return "sam3", engine_path


def describe() -> dict:
    """Machine summary used by `bin/gssg-run doctor`."""
    return {
        "machine": arch(),
        "is_jetson": is_jetson(),
        "tensorrt": _has_tensorrt(),
        "engines": [e.name for e in _available_engines()],
    }
