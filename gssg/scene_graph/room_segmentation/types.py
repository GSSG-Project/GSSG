from dataclasses import dataclass, field

from shapely.geometry.base import BaseGeometry


@dataclass
class Room:
    id: int
    polygon: BaseGeometry
    storey: int
    floor_height: float
    area_m2: float
    label: str | None = None


@dataclass
class Storey:
    index: int
    floor_height: float
    room_ids: list = field(default_factory=list)


@dataclass
class RoomSegmentationResult:
    storeys: list = field(default_factory=list)
    rooms: dict = field(default_factory=dict)  # {room_id: Room}

    @property
    def is_empty(self):
        return not self.rooms

    def room_polygons(self):
        """{room_id: shapely geometry} — the contract SceneGraph.update_object_rooms expects."""
        return {room_id: room.polygon for room_id, room in self.rooms.items()}
