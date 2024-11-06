from collections import Counter
from copy import deepcopy
from cascadia_ai.areas import HabitatAreas
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.groups import WildlifeGroups
from cascadia_ai.positions import HexPosition, adjacent_positions
from cascadia_ai.tiles import Tile


def share_edge(
    pos1: HexPosition,
    tile1: Tile,
    pos2: HexPosition,
    tile2: Tile,
    habitat: Habitat | None = None,
):
    q1, r1 = pos1
    q2, r2 = pos2
    edge1 = tile1.get_edge((q2 - q1, r2 - r1))
    edge2 = tile2.get_edge((q1 - q2, r1 - r2))
    return edge1 == edge2 and (habitat is None or edge1 == habitat)


class Environment:
    nature_tokens: int
    tiles: dict[HexPosition, Tile]
    wildlife: dict[HexPosition, Wildlife]
    hareas: dict[Habitat, HabitatAreas]
    wgroups: dict[Wildlife, WildlifeGroups]

    def __init__(self, starting_tile_group: dict[HexPosition, Tile]):
        self.nature_tokens = 0
        self.tiles = dict(starting_tile_group)
        self.wildlife = {}
        self.hareas = {h: HabitatAreas(h, self.tiles) for h in Habitat}
        self.wgroups = {w: WildlifeGroups(w) for w in Wildlife}

    def adjacent_tiles(self, pos: HexPosition):
        return (
            (apos, self.tiles[apos])
            for apos in adjacent_positions(pos)
            if apos in self.tiles
        )

    def adjacent_wildlife(self, pos: HexPosition, filter: Wildlife | None = None):
        return (
            (apos, self.wildlife[apos])
            for apos in adjacent_positions(pos)
            if apos in self.wildlife
            and (filter is None or filter == self.wildlife[apos])
        )

    def has_adjacent_wildlife(self, pos: HexPosition, filter: Wildlife):
        for _ in self.adjacent_wildlife(pos, filter):
            return True
        return False

    def adjacent_wildlife_counts(self, pos: HexPosition):
        return Counter(w for _, w in self.adjacent_wildlife(pos))

    def adjacent_empty(self, pos: HexPosition):
        return (apos for apos in adjacent_positions(pos) if apos not in self.tiles)

    def all_adjacent_empty(self):
        return {apos for pos in self.tiles for apos in self.adjacent_empty(pos)}

    def unoccupied_tiles(self):
        for pos, tile in self.tiles.items():
            if pos not in self.wildlife:
                yield (pos, tile)

    def habitat_groups(self):
        connections: dict[Habitat, dict[HexPosition, set[HexPosition]]] = {
            h: {} for h in Habitat
        }

        for pos, tile in self.tiles.items():
            for h in set(tile.habitats):
                connections[h][pos] = {pos} | {
                    apos
                    for apos, atile in self.adjacent_tiles(pos)
                    if share_edge(pos, tile, apos, atile, h)
                }

        for h in Habitat:
            for pos in list(connections[h]):
                merged = False
                for other_pos, other_group in connections[h].items():
                    if pos != other_pos and pos in other_group:
                        connections[h][other_pos] = other_group | connections[h][pos]
                        merged = True
                if merged:
                    del connections[h][pos]

        return {
            h: sorted(groups_dict.values(), key=len, reverse=True)
            for h, groups_dict in connections.items()
        }

    def wildlife_groups(self, wildlife: Wildlife):
        positions = {p for p, w in self.wildlife.items() if w == wildlife}

        visited = set[HexPosition]()
        groups = list[set[HexPosition]]()

        for p in positions:
            if p in visited:
                continue

            current_group = set[HexPosition]()
            unexplored = {p}

            while len(unexplored):
                next = unexplored.pop()
                current_group.add(next)
                unexplored |= set(
                    apos
                    for apos in adjacent_positions(next)
                    if apos in positions and apos not in current_group
                )

            visited.update(current_group)
            groups.append(current_group)

        return sorted(groups, key=len, reverse=True)

    def can_place_tile(self, pos: HexPosition):
        return pos not in self.tiles and any(
            apos in self.tiles for apos in adjacent_positions(pos)
        )

    def can_place_wildlife(self, pos: HexPosition, wildlife: Wildlife):
        return (
            pos not in self.wildlife
            and pos in self.tiles
            and wildlife in self.tiles[pos].wildlife_slots
        )

    def place_tile(self, pos: HexPosition, tile: Tile):
        if not self.can_place_tile(pos):
            raise ValueError("Cannot place tile there")
        self.tiles[pos] = tile
        for h in Habitat:
            self.hareas[h] = self.hareas[h].place_tile(pos, tile)

    def place_wildlife(self, pos: HexPosition, wildlife: Wildlife):
        if not self.can_place_wildlife(pos, wildlife):
            raise ValueError("Cannot place wildlife there")
        self.wildlife[pos] = wildlife
        for w in Wildlife:
            self.wgroups[w] = self.wgroups[w].place_wildlife(pos, wildlife)

    def copy(self):
        return deepcopy(self)
