"""Label grid -> simplified Shapely polygons in world coordinates."""

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .constants import POLY_SIMPLIFY_FRAC


def vectorize_rooms(labels, grid):
    """Returns {label: shapely geometry} for every room label in the grid."""
    rooms = {}
    for uid in np.unique(labels):
        if uid <= 0:
            continue
        mask = (labels == uid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            epsilon = POLY_SIMPLIFY_FRAC * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            coords = [grid.pixel_to_world(x, y) for x, y in approx[:, 0, :]]
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                polygons.append(polygon)
        if polygons:
            rooms[int(uid)] = unary_union(polygons)
    return rooms
