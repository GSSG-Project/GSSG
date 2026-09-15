"""Central path resolver — import these instead of hardcoding absolute paths.

Defaults are repo-relative so a fresh clone works anywhere; each can be
overridden by an environment variable for machine-specific layouts (Jetson, etc.).
"""

import os
from pathlib import Path

# gssg/utils/paths.py -> parents[0]=utils, [1]=gssg, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else default


# Model checkpoints (FastSAM, MobileCLIP, CLIP). Override: GSSG_CHECKPOINTS.
# Default: <repo>/checkpoints.
CHECKPOINT_DIR = _env_path(
    "GSSG_CHECKPOINTS", REPO_ROOT / "checkpoints"
)

# Input datasets. Override: GSSG_DATA.
DATA_DIR = _env_path("GSSG_DATA", REPO_ROOT / "data")

# Run outputs / saved Gaussian models / scene graphs. Override: GSSG_OUTPUT.
OUTPUT_DIR = _env_path("GSSG_OUTPUT", REPO_ROOT / "output")


def checkpoint(name: str) -> str:
    """Absolute path to a model checkpoint by filename."""
    return str(CHECKPOINT_DIR / name)


def repo_path(*parts: str) -> str:
    """Absolute path rooted at the repository (e.g. repo_path('thirdparty', 'bestsam'))."""
    return str(REPO_ROOT.joinpath(*parts))
