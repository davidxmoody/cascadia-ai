from cascadia_ai.enums import Habitat
from cascadia_ai.environments import HexPosition, TileGrid
from cascadia_ai.tiles import Tile

AreaLabel = int
EdgeKey = tuple[int, int, int]


def get_edge_key(pos: HexPosition, rot: int) -> EdgeKey:
    if 0 <= rot <= 2:
        return (pos[0], pos[1], rot)
    if rot == 3:
        return (pos[0], pos[1] - 1, 0)
    if rot == 4:
        return (pos[0] - 1, pos[1], 1)
    if rot == 5:
        return (pos[0] - 1, pos[1] + 1, 2)
    raise Exception("Invalid rotation")


edge_key_cache = dict[HexPosition, list[EdgeKey]]()


def get_all_edge_keys(pos: HexPosition):
    if pos not in edge_key_cache:
        edge_key_cache[pos] = [get_edge_key(pos, rot) for rot in range(6)]
    return edge_key_cache[pos]


class HabitatAreas:
    habitat: Habitat
    largest_area: int

    _next_label: AreaLabel
    _areas: dict[AreaLabel, int]
    _edges: dict[EdgeKey, AreaLabel]

    def __init__(self, habitat: Habitat, tiles: TileGrid | None = None):
        self.habitat = habitat
        self.largest_area = 0

        self._next_label = 0
        self._areas = {}
        self._edges = {}

        if tiles is not None:
            for pos, tile in tiles.items():
                self.place_tile(pos, tile)

    def get_reward(self, edge_keys: list[EdgeKey]):
        new_area = 1
        for edge_key in edge_keys:
            if edge_key in self._edges:
                new_area += self._areas[self._edges[edge_key]]

        reward = 0
        if new_area > self.largest_area:
            reward += new_area - self.largest_area
            if new_area >= 7 and not (self.largest_area >= 7):
                reward += 2

        return reward

    def get_best_reward(self, pos: HexPosition):
        return self.get_reward(get_all_edge_keys(pos))

    def place_tile(self, pos: HexPosition, tile: Tile):
        edge_keys = {
            get_edge_key(pos, rot)
            for rot, habitat in enumerate(tile.edges)
            if habitat == self.habitat
        }

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
