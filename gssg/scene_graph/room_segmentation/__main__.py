"""Run room segmentation on a saved Gaussian map, without re-running SLAM.

    python -m gssg.scene_graph.room_segmentation --input <stable.ply> --method <name> --out DIR

method: ours (default) | ours_v2 | hydra | hovsg | all (side-by-side comparison).
vertical_axis: 2 for Replica/ROS (default), 1 for HM3D. Splats are loaded with plain
numpy (no CUDA); prefer the run's final/global stable PLY. Each run writes
floorplan.json + floorplan.png (see floorplan.py for the format) plus debug rasters.
"""

import argparse
import json
import os
import time

from .cloud import load_splat_cloud
from .floorplan import coverage_stats, write_floorplan
from .methods import METHODS, segment


def _run(cloud, method, vertical_axis, out, source):
    started = time.perf_counter()
    result = segment(cloud, method=method, vertical_axis=vertical_axis, output_dir=out)
    elapsed = time.perf_counter() - started
    write_floorplan(result, out, method=method, vertical_axis=vertical_axis, source=source)

    stats = coverage_stats(result, cloud, vertical_axis)
    stats["runtime_s"] = round(elapsed, 2)
    stats["splats"] = len(cloud)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"[{method}] {elapsed:.1f}s  storeys: {len(result.storeys)}  rooms: {len(result.rooms)}")
    for rid, room in sorted(result.rooms.items()):
        print(f"  room {rid}: storey {room.storey}  area = {room.area_m2:.2f} m^2")
    print(f"  stats: {stats}")
    print(f"  output -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser(prog="python -m gssg.scene_graph.room_segmentation")
    ap.add_argument("ply", nargs="?", help="path to a saved Gaussian PLY (e.g. *_stable.ply)")
    ap.add_argument("--input", dest="input_ply", help="alias for the positional PLY path")
    ap.add_argument("--method", default="ours", choices=[*sorted(METHODS), "all"])
    ap.add_argument("--vertical-axis", type=int, default=2, help="2=Replica/ROS (default), 1=HM3D")
    ap.add_argument("--out", default=None, help="output dir (default: <ply_dir>/rooms/<method>)")
    args = ap.parse_args()

    ply = args.input_ply or args.ply
    if not ply:
        ap.error("missing PLY path (positional or --input)")
    base_out = args.out or os.path.join(os.path.dirname(os.path.abspath(ply)), "rooms")

    cloud = load_splat_cloud(ply)
    print(f"loaded {len(cloud):,} splats from {ply}")
    methods = sorted(METHODS) if args.method == "all" else [args.method]
    for method in methods:
        out = base_out if len(methods) == 1 else os.path.join(base_out, method)
        _run(cloud, method, args.vertical_axis, out, ply)


if __name__ == "__main__":
    main()
