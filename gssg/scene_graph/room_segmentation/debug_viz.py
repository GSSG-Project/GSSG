"""Debug artifacts. Never allowed to break segmentation — every saver is best-effort."""

import os

import cv2
import numpy as np


def save_storey_masks(output_dir, storey_index, wall, free, labels):
    try:
        suffix = f"_s{storey_index}"
        wall_img = wall.astype(np.uint8) * 255
        free_img = free.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, f"walls_debug{suffix}.png"), wall_img)
        cv2.imwrite(os.path.join(output_dir, f"free_debug{suffix}.png"), free_img)
        if labels.max() > 0:
            colored = cv2.applyColorMap(
                (labels * (255 // max(int(labels.max()), 1))).astype(np.uint8), cv2.COLORMAP_JET
            )
            colored[labels == 0] = 0
            cv2.imwrite(os.path.join(output_dir, f"rooms_raster{suffix}.png"), colored)
    except Exception as e:  # noqa: BLE001
        print(f"[room_segmentation] debug mask save failed: {e}")


def save_room_vectors(output_path, rooms):
    if not rooms:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        for uid, poly in rooms.items():
            geoms = [poly] if hasattr(poly, "exterior") else list(poly.geoms)
            for geom in geoms:
                x, y = geom.exterior.xy
                plt.fill(x, y, alpha=0.5, label=f"Room {uid}")
                plt.plot(x, y, "k-", linewidth=1)
        plt.axis("equal")
        plt.title("Room Polygons")
        plt.grid(True)
        plt.legend(loc="upper right", fontsize=8)
        plt.savefig(output_path)
        plt.close()
    except Exception as e:  # noqa: BLE001
        print(f"[room_segmentation] vector plot save failed: {e}")
