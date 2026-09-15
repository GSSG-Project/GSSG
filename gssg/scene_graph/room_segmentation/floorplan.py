"""Floorplan output format, shared by every method.

A floorplan is written as two files in the output directory:

``floorplan.json``::

    {
      "format": "gssg-floorplan-v1",
      "method": "ours",                  # hydra | hovsg | ours | ours_v2
      "source": "/path/to/scene.ply",    # or null
      "vertical_axis": 2,
      "units": "meters",
      "storeys": [{"index": 0, "floor_height": -1.55, "room_ids": [1, 2]}],
      "rooms": [
        {"id": 1, "storey": 0, "floor_height": -1.55, "area_m2": 15.2, "label": null,
         "polygon": [{"exterior": [[u, v], ...], "holes": [[[u, v], ...], ...]}]}
      ]
    }

Room polygons are 2D in the plan axes (the two axes orthogonal to ``vertical_axis``),
world coordinates, one entry per part (MultiPolygons have several parts).

``floorplan.png``: one panel per storey, rooms filled and numbered.
"""

import json
import os

import numpy as np


def _polygon_parts(geometry):
    parts = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
    out = []
    for part in parts:
        if not hasattr(part, "exterior"):
            continue
        out.append(
            {
                "exterior": [[float(x), float(y)] for x, y in part.exterior.coords],
                "holes": [
                    [[float(x), float(y)] for x, y in ring.coords] for ring in part.interiors
                ],
            }
        )
    return out


def write_floorplan_json(result, path, *, method, vertical_axis, source=None):
    data = {
        "format": "gssg-floorplan-v1",
        "method": method,
        "source": os.path.abspath(source) if source else None,
        "vertical_axis": int(vertical_axis),
        "units": "meters",
        "storeys": [
            {
                "index": s.index,
                "floor_height": round(s.floor_height, 4),
                "room_ids": list(s.room_ids),
            }
            for s in result.storeys
        ],
        "rooms": [
            {
                "id": room.id,
                "storey": room.storey,
                "floor_height": round(room.floor_height, 4),
                "area_m2": round(room.area_m2, 3),
                "label": room.label,
                "polygon": _polygon_parts(room.polygon),
            }
            for _, room in sorted(result.rooms.items())
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def render_floorplan(result, path, *, method):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    storeys = result.storeys or []
    n = max(len(storeys), 1)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False)
    cmap = plt.get_cmap("tab20")

    for col, storey in enumerate(storeys):
        ax = axes[0][col]
        for room_id in storey.room_ids:
            room = result.rooms[room_id]
            color = cmap((room_id - 1) % 20)
            parts = room.polygon.geoms if hasattr(room.polygon, "geoms") else [room.polygon]
            for part in parts:
                if not hasattr(part, "exterior"):
                    continue
                xs, ys = part.exterior.xy
                ax.fill(xs, ys, color=color, alpha=0.55)
                ax.plot(xs, ys, color="black", linewidth=0.8)
                for ring in part.interiors:
                    hx, hy = ring.xy
                    ax.fill(hx, hy, color="white")
                    ax.plot(hx, hy, color="black", linewidth=0.8)
            anchor = room.polygon.representative_point()
            text = str(room.id) if room.label is None else f"{room.id}: {room.label}"
            ax.annotate(text, (anchor.x, anchor.y), ha="center", va="center", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.set_title(
            f"{method} — storey {storey.index} "
            f"(floor {storey.floor_height:.2f} m, {len(storey.room_ids)} rooms)"
        )
    if not storeys:
        axes[0][0].set_title(f"{method} — no rooms found")
        axes[0][0].set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_floorplan(result, output_dir, *, method, vertical_axis, source=None):
    """Write floorplan.json + floorplan.png; returns the JSON path."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "floorplan.json")
    write_floorplan_json(
        result, json_path, method=method, vertical_axis=vertical_axis, source=source
    )
    try:
        render_floorplan(result, os.path.join(output_dir, "floorplan.png"), method=method)
    except Exception as e:  # noqa: BLE001 - the JSON is the contract; the render is best-effort
        print(f"[room_segmentation] floorplan render failed: {e}")
    return json_path


def coverage_stats(result, cloud, vertical_axis):
    """Fraction of observed plan area covered by rooms + per-method quick stats."""
    from shapely.ops import unary_union

    if result.is_empty:
        return {"rooms": 0, "storeys": len(result.storeys), "covered_area_m2": 0.0}
    union = unary_union([r.polygon for r in result.rooms.values()])
    areas = np.array([r.area_m2 for r in result.rooms.values()])
    return {
        "rooms": len(result.rooms),
        "storeys": len(result.storeys),
        "covered_area_m2": round(float(union.area), 2),
        "room_area_min_m2": round(float(areas.min()), 2),
        "room_area_median_m2": round(float(np.median(areas)), 2),
        "room_area_max_m2": round(float(areas.max()), 2),
    }
