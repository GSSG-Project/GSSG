from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def export_object_room_map(
    scene_graph, save_path="object_rooms.png", show_rooms=True, point_size=40
):
    """Top-down 2D map of objects colored by their rooms.

    Projection plane matches scene_graph.vertical_axis:
      vertical_axis == 0  (X up):  floor = (Y, Z)
      vertical_axis == 1  (Y up):  floor = (X, Z)
      vertical_axis == 2  (Z up):  floor = (X, Y)
    """
    v_axis = int(getattr(scene_graph, "vertical_axis", 1))
    if v_axis == 1:
        u_idx, v_idx, u_label, v_label = 0, 2, "X", "Z"
    elif v_axis == 2:
        u_idx, v_idx, u_label, v_label = 0, 1, "X", "Y"
    elif v_axis == 0:
        u_idx, v_idx, u_label, v_label = 1, 2, "Y", "Z"
    else:
        raise ValueError(f"Unsupported vertical_axis={v_axis}; expected 0, 1, or 2.")

    room_points = defaultdict(list)
    for obj in scene_graph.all_objects.values():
        if getattr(obj, "center", None) is None:
            continue
        if not hasattr(obj, "room") or obj.room is None:
            continue
        c = obj.center.tolist()
        room_points[obj.room.id].append((c[u_idx], c[v_idx]))

    if not room_points:
        print("No objects with assigned rooms to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab20")

    if show_rooms:
        for idx, (_room_id, room) in enumerate(scene_graph.rooms.items()):
            if room.polygon is None:
                continue
            # A room polygon can be a MultiPolygon (disjoint pieces) — plot each part.
            geoms = (
                [room.polygon] if hasattr(room.polygon, "exterior") else list(room.polygon.geoms)
            )
            for geom in geoms:
                xs, ys = geom.exterior.xy
                ax.plot(xs, ys, color=cmap(idx), linewidth=2, alpha=0.6)

    for idx, (room_id, points) in enumerate(room_points.items()):
        pts = np.array(points)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=point_size,
            color=cmap(idx),
            label=f"Room {room_id}",
            edgecolors="k",
        )

    ax.set_aspect("equal")
    ax.set_xlabel(u_label)
    ax.set_ylabel(v_label)
    ax.set_title("Object–Room Map (Top-Down)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"Saved object room map to: {save_path}")
