"""Hydra room detection (Hughes et al., RSS 2022), adapted to a saved map.

Reimplements the room finder of MIT-SPARK/Hydra (src/rooms/room_finder.cpp,
graph_filtration.cpp, graph_clustering.cpp) faithfully: a places graph whose nodes carry
free-space clearance and whose edges carry the minimum clearance along the connection,
an exact event-driven filtration (number of components at every distinct edge weight),
PLATEAU threshold selection inside the [0.5, 1.2) m dilation window, seed components at
the chosen threshold, and a max-clearance-first best-first flood for the remaining places.

Hydra's upstream (TSDF -> GVD -> sparse places) does not exist for a raw point cloud, so
the places graph is approximated per storey in 2D: occupancy from splat centers in a
door-height band, free space clipped to the observed footprint, EDT clearance, medial
axis as the GVD, nodes by 0.5 m grid compression. min_component_size/min_room_size are
rescaled (10 -> 3) for the sparser 1D skeleton (their counts assume a 3D medial surface
with roughly one place per m^2 of room; a medial axis yields a handful per room).
Rooms get polygons by flooding free space from their member places — Hydra itself leaves
room extents implicit in the place spheres.
"""

import heapq

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..classify import HORIZONTAL, classify_splats
from ..debug_viz import save_storey_masks
from ..floors import detect_floor_heights
from ..grid import BEVGrid
from ..partition import _geodesic_flood, _relabel_by_area
from ..types import Room, RoomSegmentationResult, Storey
from ..vectorize import vectorize_rooms

RESOLUTION_M = 0.1
OCC_BAND_ABOVE_FLOOR_M = (0.2, 2.0)
OCC_BAND_BELOW_CEILING_M = 0.3
OCC_MIN_POINTS = 3  # TSDF surrogate: isolated floaters do not register as surface
OBSERVED_CLOSE_PX = 3

COMPRESSION_M = 0.5
MIN_NODE_DISTANCE_M = 0.4
MIN_EDGE_DISTANCE_M = 0.25
RECONNECT_MAX_M = 2.0
RECONNECT_MIN_CLEARANCE_M = 0.5

MIN_DILATION_M = 0.5
MAX_DILATION_M = 1.2
PLATEAU_RATIO = 0.25
MIN_COMPONENT_SIZE = 3
MIN_ROOM_SIZE = 3
MIN_POLYGON_AREA_M2 = 0.5


class _DisjointSet:
    def __init__(self, n, min_size):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=np.int64)
        self.min_size = min_size
        self.big = 0

    def find(self, a):
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        big_before = int(self.size[ra] >= self.min_size) + int(self.size[rb] >= self.min_size)
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.big += int(self.size[ra] >= self.min_size) - big_before
        return True


def _line_min(dist_px, p0, p1):
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    rr = np.round(np.linspace(p0[0], p1[0], n)).astype(np.int64)
    cc = np.round(np.linspace(p0[1], p1[1], n)).astype(np.int64)
    return float(dist_px[rr, cc].min())


def _places_graph(free, dist_m, resolution):
    """Nodes: medial-axis pixels compressed on a COMPRESSION_M grid, clearance = EDT.
    Edges: bucket adjacency along the skeleton + short free-space reconnections;
    weight = min clearance along the straight line, capped by both endpoint clearances."""
    from skimage.morphology import medial_axis

    skeleton = medial_axis(free) & (dist_m >= MIN_NODE_DISTANCE_M)
    rows, cols = np.nonzero(skeleton)
    if len(rows) == 0:
        return np.zeros((0, 2), np.int64), np.zeros(0), {}, np.zeros(free.shape, np.int64) - 1

    pitch = max(int(round(COMPRESSION_M / resolution)), 1)
    bucket_key = (rows // pitch) * ((free.shape[1] // pitch) + 2) + (cols // pitch)
    order = np.lexsort((-dist_m[rows, cols], bucket_key))
    keys_sorted = bucket_key[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = keys_sorted[1:] != keys_sorted[:-1]

    node_of_key = {int(k): i for i, k in enumerate(keys_sorted[first])}
    reps = order[first]
    nodes_px = np.stack([rows[reps], cols[reps]], axis=1)
    node_dist = dist_m[rows[reps], cols[reps]]

    pixel_node = np.full(free.shape, -1, dtype=np.int64)
    pixel_node[rows, cols] = [node_of_key[int(k)] for k in bucket_key]

    dist_px = dist_m / resolution
    edges = {}

    def add_edge(a, b):
        if a == b:
            return
        key = (min(a, b), max(a, b))
        if key in edges:
            return
        w = min(
            _line_min(dist_px, nodes_px[a], nodes_px[b]) * resolution,
            float(node_dist[a]),
            float(node_dist[b]),
        )
        if w >= MIN_EDGE_DISTANCE_M:
            edges[key] = w

    pairs = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rows + dr, cols + dc
        ok = (r2 >= 0) & (r2 < free.shape[0]) & (c2 >= 0) & (c2 < free.shape[1])
        neighbor = pixel_node[r2[ok], c2[ok]]
        here = pixel_node[rows[ok], cols[ok]]
        distinct = (neighbor >= 0) & (here != neighbor)
        pairs.append(
            np.stack(
                [
                    np.minimum(here[distinct], neighbor[distinct]),
                    np.maximum(here[distinct], neighbor[distinct]),
                ],
                axis=1,
            )
        )
    for a, b in np.unique(np.concatenate(pairs), axis=0):
        add_edge(int(a), int(b))

    tree = cKDTree(nodes_px * resolution)
    for a, b in tree.query_pairs(RECONNECT_MAX_M):
        if (min(a, b), max(a, b)) not in edges:
            clearance = _line_min(dist_px, nodes_px[a], nodes_px[b]) * resolution
            if clearance >= RECONNECT_MIN_CLEARANCE_M:
                add_edge(a, b)

    return nodes_px, node_dist, edges, pixel_node


def _filtration(n_nodes, edges):
    """(d, #components >= MIN_COMPONENT_SIZE) at every distinct edge weight, ascending."""
    uf = _DisjointSet(n_nodes, MIN_COMPONENT_SIZE)
    out = []
    for w, a, b in sorted(((w, a, b) for (a, b), w in edges.items()), reverse=True):
        if uf.union(a, b):
            out.append((w, uf.big))
    out.reverse()
    return out


def _best_plateau(filtration):
    window = [(d, n) for d, n in filtration if MIN_DILATION_M <= d < MAX_DILATION_M]
    if not window:
        return None
    runs = []
    start = 0
    for i in range(1, len(window) + 1):
        if i == len(window) or window[i][1] != window[start][1]:
            runs.append((window[start][0], window[i - 1][0], window[start][1]))
            start = i
    max_lifetime = max(hi - lo for lo, hi, _ in runs)
    best = None
    for run in runs:
        if run[1] - run[0] >= PLATEAU_RATIO * max_lifetime and (best is None or run[2] > best[2]):
            best = run
    return (best or runs[0])[0]


def _seed_components(node_dist, edges, threshold):
    uf = _DisjointSet(len(node_dist), MIN_COMPONENT_SIZE)
    alive = node_dist > threshold
    for (a, b), w in edges.items():
        if w > threshold and alive[a] and alive[b]:
            uf.union(a, b)
    labels = np.full(len(node_dist), -1, dtype=np.int64)
    roots = {}
    for i in np.nonzero(alive)[0]:
        root = uf.find(i)
        if uf.size[root] >= MIN_COMPONENT_SIZE:
            labels[i] = roots.setdefault(root, len(roots))
    return labels


def _flood_neighbors(labels, edges):
    """Max-clearance-first best-first flood of unlabeled places (clusterGraphByNeighbors)."""
    adjacency = {}
    for (a, b), w in edges.items():
        adjacency.setdefault(a, []).append((b, w))
        adjacency.setdefault(b, []).append((a, w))
    heap = []
    for a in np.nonzero(labels >= 0)[0]:
        for b, w in adjacency.get(int(a), ()):
            if labels[b] < 0:
                heapq.heappush(heap, (-w, b, labels[a]))
    while heap:
        _, node, room = heapq.heappop(heap)
        if labels[node] >= 0:
            continue
        labels[node] = room
        for b, w in adjacency.get(node, ()):
            if labels[b] < 0:
                heapq.heappush(heap, (-w, b, room))
    return labels


def _segment_storey(splats, floor_h, ceiling, output_dir, storey_index):
    grid = BEVGrid.from_points(splats.u, splats.v, RESOLUTION_M)
    band_hi = min(floor_h + OCC_BAND_ABOVE_FLOOR_M[1], ceiling - OCC_BAND_BELOW_CEILING_M)
    in_band = (splats.h >= floor_h + OCC_BAND_ABOVE_FLOOR_M[0]) & (splats.h <= band_hi)
    occupied = grid.occupancy(splats.u[in_band], splats.v[in_band], min_count=OCC_MIN_POINTS)

    observed = grid.occupancy(splats.u, splats.v)
    structure = np.ones((OBSERVED_CLOSE_PX, OBSERVED_CLOSE_PX), dtype=bool)
    observed = ndimage.binary_closing(observed, structure=structure)
    free = observed & ~occupied

    dist_m = ndimage.distance_transform_edt(free) * grid.resolution
    nodes_px, node_dist, edges, pixel_node = _places_graph(free, dist_m, grid.resolution)
    if len(nodes_px) == 0 or not edges:
        return np.zeros(free.shape, np.int32), grid

    threshold = _best_plateau(_filtration(len(node_dist), edges))
    if threshold is None:
        return np.zeros(free.shape, np.int32), grid

    labels = _flood_neighbors(_seed_components(node_dist, edges, threshold), edges)

    room_sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.zeros(0)
    node_room = np.zeros(len(labels) + 1, dtype=np.int32)  # last slot: pixel_node == -1
    raster_id = {}
    for i in np.nonzero(labels >= 0)[0]:
        if room_sizes[labels[i]] >= MIN_ROOM_SIZE:
            node_room[i] = raster_id.setdefault(int(labels[i]), len(raster_id) + 1)
    seeds = node_room[pixel_node]
    raster = _relabel_by_area(_geodesic_flood(seeds, free)) if raster_id else seeds
    if output_dir:
        save_storey_masks(output_dir, storey_index, occupied, free, raster)
    return raster, grid


def segment(cloud, vertical_axis=2, output_dir=None) -> RoomSegmentationResult:
    result = RoomSegmentationResult()
    normals = cloud.normals if cloud.normals is not None else np.zeros_like(cloud.xyz)
    splats = classify_splats(cloud.xyz, normals, cloud.opacities, vertical_axis)
    if len(splats) == 0:
        return result

    horizontal = splats.h[splats.label == HORIZONTAL] if cloud.normals is not None else np.zeros(0)
    floors = detect_floor_heights(horizontal, splats.h)

    next_room_id = 1
    for index, floor_h in enumerate(floors):
        ceiling = floors[index + 1] if index + 1 < len(floors) else np.inf
        storey_splats = splats.subset((splats.h >= floor_h - 0.2) & (splats.h < ceiling - 0.2))
        if len(storey_splats) == 0:
            continue
        raster, grid = _segment_storey(storey_splats, floor_h, ceiling, output_dir, index)
        polygons = vectorize_rooms(raster, grid)

        storey = Storey(index=index, floor_height=float(floor_h))
        for local_id in sorted(polygons):
            if polygons[local_id].area < MIN_POLYGON_AREA_M2:
                continue
            room = Room(
                id=next_room_id,
                polygon=polygons[local_id],
                storey=index,
                floor_height=float(floor_h),
                area_m2=float(polygons[local_id].area),
            )
            result.rooms[room.id] = room
            storey.room_ids.append(room.id)
            next_room_id += 1
        result.storeys.append(storey)
    return result
