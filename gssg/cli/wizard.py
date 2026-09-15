"""Interactive run wizard: ask a few questions, write a layered child config under
``configs/_runs/<name>.yaml``, and hand off to ``bin/gssg-run`` to launch it."""

import os
import sys
from pathlib import Path

from gssg.utils.config_utils import read_config
from gssg.utils.paths import REPO_ROOT
from gssg.utils.platform import is_jetson

_DATASETS_DIR = REPO_ROOT / "configs" / "datasets"
_RUNS_DIR = REPO_ROOT / "configs" / "_runs"
_GSSG_RUN = REPO_ROOT / "bin" / "gssg-run"
_QUALITIES = ["low", "mid", "high"]

C_Q = "\033[36m"  # prompt
C_H = "\033[2m"  # hint/default
C_R = "\033[0m"


# ---------------------------------------------------------------- prompts
def _ask(prompt: str, default: str) -> str:
    raw = input(f"{C_Q}{prompt}{C_R} {C_H}[{default}]{C_R} ").strip()
    return raw or default


def ask_yesno(prompt: str, default: bool) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{C_Q}{prompt}{C_R} {C_H}[{d}]{C_R} ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def ask_choice(prompt: str, options: list[tuple[str, str]], default: int = 0) -> str:
    print(f"\n{C_Q}{prompt}{C_R}")
    for i, (_key, label) in enumerate(options, 1):
        mark = " (default)" if i - 1 == default else ""
        print(f"  {i}) {label}{C_H}{mark}{C_R}")
    while True:
        raw = input(f"  choice {C_H}[{default + 1}]{C_R} ").strip()
        if not raw:
            return options[default][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]


def ask_quality() -> str:
    return ask_choice(
        "Quality preset?",
        [("low", "low — fastest, viewer off"), ("mid", "mid — balanced"), ("high", "high — best")],
        default=0,
    )


# ---------------------------------------------------------------- helpers
def _list_dataset_configs(exclude=()) -> list[str]:
    return sorted(p.stem for p in _DATASETS_DIR.glob("*.yaml") if p.stem not in exclude)


def _config_summary(stem: str) -> str:
    """First comment line of a dataset config, for the picker."""
    p = _DATASETS_DIR / f"{stem}.yaml"
    try:
        for line in p.read_text(errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("#") and len(s) > 1:
                return s.lstrip("# ").strip()[:70]
            if s and not s.startswith("#"):
                break
    except OSError:
        pass
    return ""


def _is_ros(config_relpath: str) -> bool:
    try:
        return read_config(config_relpath).dataset_type in ("ros", "ros2")
    except Exception:
        return False


def _launch(config_relpath: str, quality: str | None, mp: bool, no_ros: bool) -> None:
    cmd = [str(_GSSG_RUN), config_relpath]
    if quality:
        cmd += ["--quality", quality]
    cmd.append("--mp" if mp else "--single")
    if no_ros:
        cmd.append("--no-ros")
    print(f"\n{C_Q}launching:{C_R} {' '.join(cmd)}")
    print(f"{C_H}(re-run later non-interactively with the same command){C_R}\n")
    os.execv(str(_GSSG_RUN), cmd)


# ---------------------------------------------------------------- wizard
def main() -> None:
    if not sys.stdin.isatty():
        sys.exit("gssg-run new needs an interactive terminal. Use `gssg-run <config>` instead.")

    print(f"\n{C_Q}GSSG run wizard{C_R} {C_H}(Ctrl+C to cancel){C_R}")

    itype = ask_choice(
        "What are you running?",
        [
            ("zed", "ZED 2i — live SLAM over ROS 2"),
            ("dataset", "Offline dataset (Replica / HM3D / CLIO / …)"),
            ("existing", "An existing config, unchanged (advanced)"),
        ],
    )

    # Run an existing config straight through, no generated file.
    if itype == "existing":
        configs = _list_dataset_configs()
        stem = ask_choice(
            "Which config?", [(c, f"{c}  {C_H}{_config_summary(c)}") for c in configs]
        )
        rel = f"configs/datasets/{stem}.yaml"
        quality = ask_quality() if ask_yesno("Override quality preset?", False) else None
        ros = _is_ros(rel)
        _launch(rel, quality, mp=ros, no_ros=not ros)
        return

    # Pick the base config the generated run inherits from.
    if itype == "zed":
        base = "ros2_zed"
    else:
        configs = _list_dataset_configs(exclude=("ros2_zed",))
        base = ask_choice(
            "Which dataset?", [(c, f"{c}  {C_H}{_config_summary(c)}") for c in configs]
        )
    base_rel = f"configs/datasets/{base}.yaml"
    ros = _is_ros(base_rel)

    quality = ask_quality()
    viewer = ask_yesno("Live viewer (Rerun)?", default=not is_jetson())
    submap = ask_yesno(
        "Sub-mapping (out-of-core eviction, bounds VRAM to the local working set)?", False
    )
    name = _ask("Run name (-> output/<name>)", base)
    name = name.replace(" ", "_").strip("/")

    overrides = {"save_path": f"output/{name}", "visualize": viewer, "submapping": submap}
    if submap:
        overrides["submap_evict"] = True
    out = _write_config(name, base_rel, overrides)

    print(f"\n{C_Q}summary{C_R}")
    print(f"  config   : {out}  (parent: {base_rel})")
    print(f"  quality  : {quality}")
    print(f"  viewer   : {viewer}    sub-mapping: {submap}")
    print(f"  output   : output/{name}")
    print(f"  mode     : {'multi-process, ROS 2' if ros else 'single-process, no ROS'}")
    if not ask_yesno("\nStart now?", True):
        print(f"saved {out} — start later with: bin/gssg-run {out} --quality {quality}")
        return

    rel = os.path.relpath(out, REPO_ROOT)
    _launch(rel, quality, mp=ros, no_ros=not ros)


def _write_config(name: str, base_rel: str, overrides: dict) -> Path:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RUNS_DIR / f"{name}.yaml"
    lines = [
        "# Generated by `gssg-run new`. A normal layered config — edit or re-run freely:",
        f"#   bin/gssg-run configs/_runs/{name}.yaml --quality <low|mid|high>",
        f"parent: {base_rel}",
    ]
    for k, v in overrides.items():
        lines.append(f"{k}: {_yaml(v)}")
    out.write_text("\n".join(lines) + "\n")
    return out


def _yaml(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\ncancelled.")
        sys.exit(130)
