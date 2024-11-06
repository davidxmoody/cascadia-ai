from copy import copy
from typing import Iterable, Self
from cascadia_ai.enums import Habitat
from cascadia_ai.positions import HexPosition
from cascadia_ai.tiles import Tile

AreaLabel = int
EdgeKey = int


def get_edge_key(pos: HexPosition, rot: int) -> EdgeKey:
    if 0 <= rot <= 2:
        return pos[0] * 1000 + pos[1] * 10 + rot
    if rot == 3:
        return pos[0] * 1000 + (pos[1] - 1) * 10
    if rot == 4:
        return (pos[0] - 1) * 1000 + pos[1] * 10 + 1
    if rot == 5:
        return (pos[0] - 1) * 1000 + (pos[1] + 1) * 10 + 2
    raise Exception("Invalid rotation")


edge_key_cache = dict[HexPosition, tuple[EdgeKey, ...]]()


def get_all_edge_keys(pos: HexPosition):
    if pos not in edge_key_cache:
        edge_key_cache[pos] = tuple(get_edge_key(pos, rot) for rot in range(6))
    return edge_key_cache[pos]


class HabitatLayer:
    habitat: Habitat
    largest_area: int

    _next_label: AreaLabel
    _areas: dict[AreaLabel, int]
    _edges: dict[EdgeKey, AreaLabel]

    _child_cache: dict[tuple[EdgeKey, ...], Self]

    def __init__(self, habitat: Habitat, tiles: dict[HexPosition, Tile] | None = None):
        self.habitat = habitat
        self.largest_area = 0

        self._next_label = 0
        self._areas = {}
        self._edges = {}

        self._child_cache = {}

        if tiles is not None:
            for pos, tile in tiles.items():
                self._place_edges(self._get_tile_edge_keys(pos, tile))

    @property
    def score(self):
        return self.largest_area + (2 if self.largest_area >= 7 else 0)

    def _get_tile_edge_keys(self, pos: HexPosition, tile: Tile):
        all_edge_keys = get_all_edge_keys(pos)
        return tuple(
            all_edge_keys[rot] for rot, h in enumerate(tile.edges) if h == self.habitat
        )

    def _get_edges_reward(self, edge_keys: Iterable[EdgeKey]):
        # TODO consider caching this too if necessary
        connected_area_labels = {
            self._edges[edge_key] for edge_key in edge_keys if edge_key in self._edges
        }
        new_area = 1
        for label in connected_area_labels:
            new_area += self._areas[label]

        reward = 0
        if new_area > self.largest_area:
            reward += new_area - self.largest_area
            if new_area >= 7 and not (self.largest_area >= 7):
                reward += 2

        return reward

    def get_tile_reward(self, pos: HexPosition, tile: Tile):
        return self._get_edges_reward(self._get_tile_edge_keys(pos, tile))

    def get_best_reward(self, pos: HexPosition):
        return self._get_edges_reward(get_all_edge_keys(pos))

    def _place_edges(self, edge_keys: tuple[EdgeKey, ...]):
        connected_areas = {
            self._edges[edge_key] for edge_key in edge_keys if edge_key in self._edges
        }

        if len(connected_areas) == 0:
            label = self._next_label
            self._next_label += 1

            size = 1
            self._areas[label] = size
            self.largest_area = max(self.largest_area, size)

            for edge_key in edge_keys:
                self._edges[edge_key] = label

        elif len(connected_areas) == 1:
            label = next(iter(connected_areas))

            size = self._areas[label] + 1
            self._areas[label] = size
            self.largest_area = max(self.largest_area, size)

            for edge_key in edge_keys:
                if edge_key in self._edges:
                    del self._edges[edge_key]
                else:
                    self._edges[edge_key] = label

        else:
            label, *labels_to_merge = connected_areas

            size = sum(self._areas[l] for l in connected_areas) + 1
            self._areas[label] = size
            self.largest_area = max(self.largest_area, size)

            for label_to_merge in labels_to_merge:
                del self._areas[label_to_merge]
            for edge_key, other_label in self._edges.items():
                if other_label in labels_to_merge:
                    self._edges[edge_key] = label

            for edge_key in edge_keys:
                if edge_key in self._edges:
                    del self._edges[edge_key]
                else:
                    self._edges[edge_key] = label

    def place_tile(self, pos: HexPosition, tile: Tile):
        if self.habitat not in tile.habitats:
            return self

        edge_keys = self._get_tile_edge_keys(pos, tile)

        if edge_keys not in self._child_cache:
            new_obj = self.copy()
            new_obj._place_edges(edge_keys)
            self._child_cache[edge_keys] = new_obj

        return self._child_cache[edge_keys]

    def copy(self):
        new_obj = copy(self)
        new_obj._areas = dict(new_obj._areas)
        new_obj._edges = dict(new_obj._edges)
        new_obj._child_cache = {}
        return new_obj
