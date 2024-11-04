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


class HabitatAreas:
    _next_label: AreaLabel
    _areas: dict[Habitat, dict[AreaLabel, int]]
    _edges: dict[Habitat, dict[EdgeKey, AreaLabel]]

    def __init__(self, tiles: TileGrid | None = None):
        self._next_label = 0
        self._areas = {h: {} for h in Habitat}
        self._edges = {h: {} for h in Habitat}

        if tiles is not None:
            for pos, tile in tiles.items():
                self.place_tile(pos, tile)

    def largest_area(self, habitat: Habitat):
        return max(self._areas[habitat].values())

    def get_tile_reward(self, pos: HexPosition, tile: Tile):
        pass

    def get_simple_tile_reward(self, pos: HexPosition, habitat: Habitat):
        connected_area_labels = set[AreaLabel]()
        for rot in range(6):
            edge_key = get_edge_key(pos, rot)
            if edge_key in self._edges[habitat]:
                connected_area_labels.add(self._edges[habitat][edge_key])

        new_area_size = 1 + sum(
            self._areas[habitat][label] for label in connected_area_labels
        )

        prev_largest_area_size = self.largest_area(habitat)

        reward = 0
        if new_area_size > prev_largest_area_size:
            reward += new_area_size - prev_largest_area_size
            if new_area_size >= 7 and not (prev_largest_area_size >= 7):
                reward += 2

        return reward

    def place_tile(self, pos: HexPosition, tile: Tile):
        edge_keys = {
            get_edge_key(pos, rot): habitat for rot, habitat in enumerate(tile.edges)
        }

        connected_areas = {h: set[AreaLabel]() for h in tile.unique_habitats}

        for edge_key, habitat in edge_keys.items():
            if edge_key in self._edges[habitat]:
                connected_areas[habitat].add(self._edges[habitat][edge_key])

        for habitat, area_labels in connected_areas.items():
            if len(area_labels) == 0:
                new_label = self._next_label
                self._next_label += 1
                area_labels.add(new_label)
                self._areas[habitat][new_label] = 0

            main_label, *labels_to_merge = area_labels

            if labels_to_merge:
                for label, size in list(self._areas[habitat].items()):
                    if label in labels_to_merge:
                        self._areas[habitat][main_label] += size
                        del self._areas[habitat][label]
                for edge_key, label in list(self._edges[habitat].items()):
                    if label in labels_to_merge:
                        self._edges[habitat][edge_key] = main_label

            self._areas[habitat][main_label] += 1

        for edge_key, habitat in edge_keys.items():
            is_touching = False

            for edges in self._edges.values():
                if edge_key in edges:
                    is_touching = True
                    del edges[edge_key]

            if not is_touching:
                self._edges[habitat][edge_key] = next(iter(connected_areas[habitat]))
