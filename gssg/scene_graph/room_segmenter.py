import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from plyfile import PlyData
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOM_SEGMENTATION_CONFIG = {
    "resolution": 0.03,  # meters per pixel
    "slice_min_height": 0.5,  # relative to floor
    "slice_max_height": 2.0,  # relative to floor
    "vertical_axis": 1,  # 0=X, 1=Y (HM3D), 2=Z (Replica)
    "opacity_threshold": 0.2,
    "density_threshold": 2,
    "min_room_area_m2": 5,
}


class RoomSegmenter:
    def __init__(self, args, data_source, output_dir):
        self.data_source = data_source
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.u_min = None
        self.v_min = None
        self.wall_id = None

        c = ROOM_SEGMENTATION_CONFIG
        self.resolution = getattr(args, "rs_resolution", c["resolution"])
        self.opacity_threshold = getattr(args, "rs_opacity_threshold", c["opacity_threshold"])
        self.vertical_axis = getattr(args, "vertical_axis", c["vertical_axis"])
        self.slice_min_height = getattr(args, "rs_slice_min_height", c["slice_min_height"])
        self.slice_max_height = getattr(args, "rs_slice_max_height", c["slice_max_height"])
        self.density_threshold = getattr(args, "rs_density_threshold", c["density_threshold"])
        self.min_room_area_m2 = getattr(args, "rs_min_room_area", c["min_room_area_m2"])

    def _pixel_to_real(self, c_x, c_y):
        real_u = self.u_min + (c_x * self.resolution)
        real_v = self.v_min + (c_y * self.resolution)
        return real_u, real_v

    def _vectorize_rooms(self, markers):
        rooms = {}
        unique_ids = np.unique(markers)
        for uid in unique_ids:
            if uid <= 0 or uid == self.wall_id:
                continue

            mask = (markers == uid).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys_in_room = []

            for cnt in contours:
                # Simplify geometry for cleaner vectors.
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                real_coords = []
                for point in approx[:, 0, :]:
                    r_u, r_v = self._pixel_to_real(point[0], point[1])
                    real_coords.append((r_u, r_v))

                if len(real_coords) >= 3:
                    polys_in_room.append(Polygon(real_coords))

            if polys_in_room:
                rooms[int(uid)] = unary_union(polys_in_room)
        return rooms

    def rasterize(self, markers, occ_grid):
        masked_markers = np.ma.masked_where(markers <= 0, markers)
        plt.figure(figsize=(10, 10))
        plt.imshow(
            occ_grid,
            cmap="gray_r",
            origin="lower",
            extent=[self.u_min, self.u_max, self.v_min, self.v_max],
            alpha=0.3,
        )
        plt.imshow(
            masked_markers,
            cmap="tab20",
            origin="lower",
            extent=[self.u_min, self.u_max, self.v_min, self.v_max],
            alpha=0.9,
        )
        plt.title("Segmented Rooms")
        plt.savefig(os.path.join(self.output_dir, "rooms_raster.png"))
        plt.close()

    def plot_vectors(self, output_path, rooms):
        plt.figure(figsize=(10, 10))
        if not rooms:
            return
        for uid, poly in rooms.items():
            geoms = [poly] if hasattr(poly, "exterior") else poly.geoms
            for geom in geoms:
                x, y = geom.exterior.xy
                plt.fill(x, y, alpha=0.5, label=f"Room {uid}")
                plt.plot(x, y, "k-", linewidth=1)
        plt.axis("equal")
        plt.title("Real-Scale Vector Room Map")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved vector map to {output_path}")

    def load_point_cloud(self, source):
        points = None

        if isinstance(source, str):
            print(f"Loading PLY from file: {source}")
            if not os.path.exists(source):
                raise FileNotFoundError(f"{source} does not exist.")

            plydata = PlyData.read(source)
            x = np.asarray(plydata.elements[0]["x"])
            y = np.asarray(plydata.elements[0]["y"])
            z = np.asarray(plydata.elements[0]["z"])
            points = np.stack((x, y, z), axis=1)

            if "opacity" in plydata.elements[0].data.dtype.names:
                opacities = np.asarray(plydata.elements[0]["opacity"])
                opacities = 1 / (1 + np.exp(-opacities))
                mask = opacities > self.opacity_threshold
                points = points[mask]

        elif isinstance(source, torch.Tensor):
            print("Loading from PyTorch Tensor")
            points = source.detach().cpu().numpy()

        elif isinstance(source, np.ndarray):
            print("Loading from Numpy Array")
            points = source

        else:
            raise ValueError(f"Unsupported input type: {type(source)}")

        if points.ndim > 2:
            points = points.reshape(-1, 3)

        if points.dtype.names:
            points = np.stack([points[n] for n in ["x", "y", "z"]], axis=1)

        print(f"Loaded {len(points)} points.")
        return points

    def generate_occupancy_grid(self, points):
        """Project 3D points onto a 2D grid and return the grid plus spatial metadata."""
        res = self.resolution
        v_axis = self.vertical_axis

        # Y-up (1): X->U, Z->V. Z-up (2): X->U, Y->V.
        if v_axis == 1:
            u_coords = points[:, 0]
            v_coords = points[:, 2]
            heights = points[:, 1]
        elif v_axis == 2:
            u_coords = points[:, 0]
            v_coords = points[:, 1]
            heights = points[:, 2]
        else:  # X is up
            u_coords = points[:, 1]
            v_coords = points[:, 2]
            heights = points[:, 0]

        h_mask = (heights >= self.slice_min_height) & (heights <= self.slice_max_height)
        u_coords = u_coords[h_mask]
        v_coords = v_coords[h_mask]

        if len(u_coords) == 0:
            print("No points found in the specified height slice.")
            return None, None, None, None, None

        u_min, u_max = u_coords.min(), u_coords.max()
        v_min, v_max = v_coords.min(), v_coords.max()

        padding = 0.5  # meters
        u_min -= padding
        u_max += padding
        v_min -= padding
        v_max += padding

        width = int(np.ceil((u_max - u_min) / res))
        height = int(np.ceil((v_max - v_min) / res))
        grid_density, _, _ = np.histogram2d(
            v_coords, u_coords, bins=[height, width], range=[[v_min, v_max], [u_min, u_max]]
        )

        occupancy_grid = (grid_density > self.density_threshold).astype(np.uint8) * 255
        kernel_close = np.ones((5, 5), np.uint8)
        occupancy_grid = cv2.morphologyEx(occupancy_grid, cv2.MORPH_CLOSE, kernel_close)
        kernel_dilate = np.ones((3, 3), np.uint8)
        occupancy_grid = cv2.dilate(occupancy_grid, kernel_dilate, iterations=1)
        cv2.imwrite(os.path.join(self.output_dir, "occupancy_debug.png"), occupancy_grid)

        return occupancy_grid, u_min, v_min, u_max, v_max

    def segment_rooms_watershed(self, occupancy_map):
        # Invert so wall=0, free=255.
        dist_input = cv2.bitwise_not(occupancy_map)
        dist = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)
        dist_threshold = 0.4 * dist.max()  # tunable, 0.3-0.6 works well
        _, sure_fg = cv2.threshold(dist, dist_threshold, 255, 0)
        sure_fg = np.uint8(sure_fg)
        sure_bg = cv2.dilate(occupancy_map, np.ones((3, 3), np.uint8), iterations=3)
        unknown = cv2.subtract(cv2.bitwise_not(sure_bg), sure_fg)
        unknown = cv2.subtract(dist_input, sure_fg)
        unknown[occupancy_map == 255] = 0
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        img_color = cv2.cvtColor(occupancy_map, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        min_area_pixels = self.min_room_area_m2 / (self.resolution**2)
        unique_markers = np.unique(markers)
        final_markers = np.zeros_like(markers)

        new_id = 1
        for m_id in unique_markers:
            if (
                m_id <= 1
            ):  # 0 is background/unknown, 1 is the background we added +1 to, -1 is boundary
                continue

            mask = (markers == m_id).astype(np.uint8)
            if cv2.countNonZero(mask) > min_area_pixels:
                final_markers[mask == 1] = new_id
                new_id += 1
        self.wall_id = new_id
        final_markers[occupancy_map == 255] = self.wall_id

        return final_markers, self.wall_id

    def process(self):
        points = self.load_point_cloud(self.data_source)

        print("Generating Occupancy Grid...")
        occ_grid = None
        occ_grid, self.u_min, self.v_min, self.u_max, self.v_max = self.generate_occupancy_grid(
            points
        )

        if occ_grid is None:
            return []

        print("Segmenting Rooms...")
        markers, self.wall_id = self.segment_rooms_watershed(occ_grid)

        print("Vectorizing...")
        rooms = self._vectorize_rooms(markers)

        print("Rasterizing...")
        self.rasterize(markers, occ_grid)

        return rooms


def process_room_data(data_source, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    tracker = RoomSegmenter(data_source, output_dir)
    rooms = tracker.process()
    tracker.plot_vectors(os.path.join(output_dir, "rooms_vector.png"), rooms)

    return tracker, (tracker.u_min, tracker.u_max, tracker.v_min, tracker.v_max)


if __name__ == "__main__":
    from gssg.utils.paths import OUTPUT_DIR as _OUT

    FILE_PATH = os.path.join(_OUT, "00829-QaLdnwvtxbs/save_model/frame_0100/iter_0219_stable.ply")
    OUTPUT_DIR = str(_OUT)

    print("--- RUNNING WITH FILE PATH ---")
    tracker, bounds = process_room_data(FILE_PATH, OUTPUT_DIR)
