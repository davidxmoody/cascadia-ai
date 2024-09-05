from cascadia_ai.enums import Wildlife
from cascadia_ai.hex_grid import HexGrid, HexPosition
from cascadia_ai.tiles import Tile
from typing import NamedTuple


print_template = [
    r"   /0\   ",
    r" /5   0\ ",
    r"|4     1|",
    r"|4     1|",
    r" \3   2/ ",
    r"   \3/   ",
]


class RotatedTile(NamedTuple):
    tile: Tile
    rotation: int


starting_tiles: list[HexGrid[RotatedTile]] = [
    HexGrid(
        {
            (0, 0): RotatedTile(Tile.from_definition("MMb"), 0),
            (0, -1): RotatedTile(Tile.from_definition("FWehf"), 1),
            (1, -1): RotatedTile(Tile.from_definition("RPbs"), 2),
        }
    ),
]


class Environment:
    tiles: HexGrid[RotatedTile]
    wildlife: HexGrid[Wildlife]

    def __init__(self, starting_tile_group: HexGrid[RotatedTile]):
        self.tiles = starting_tile_group.copy()
        self.wildlife = HexGrid()

    def place_tile(self, position: HexPosition, tile: Tile, rotation: int):
        if len(list(self.tiles.adjacent(position))) == 0:
            raise ValueError("No adjacent tiles")
        self.tiles[position] = RotatedTile(tile, rotation)

    def place_wildlife(self, position: HexPosition, wildlife: Wildlife):
        rtile = self.tiles[position]
        if rtile is None or wildlife not in rtile.tile.wildlife_slots:
            raise ValueError("Cannot place wildlife")
        self.wildlife[position] = wildlife

    def print(self):
        chars: dict[tuple[int, int], str] = {}

        for (q, r), (tile, rotation) in self.tiles.items():
            hl, hr = tile.habitats
            sides = [hr, hr, hr, hl, hl, hl]

            x = q * 8 + r * 4
            y = r * -4

            for dy, line in enumerate(print_template):
                for dx, char in enumerate(line):
                    p = (x + dx, y + dy)
                    if char == " ":
                        continue
                    elif char.isdigit():
                        chars[p] = sides[int(char) - rotation].value
                    else:
                        chars[p] = char

        min_x = min(p[0] for p in chars)
        max_x = max(p[0] for p in chars)
        min_y = min(p[1] for p in chars)
        max_y = max(p[1] for p in chars)

        for y in range(min_y, max_y + 1):
            line = ""
            for x in range(min_x, max_x + 1):
                p = (x, y)
                line += chars[p] if p in chars else " "
            print(line)

        # TODO draw habitats in color (or add color later)
        # TODO add co-ords and/or wildlife and/or wildlife slots
