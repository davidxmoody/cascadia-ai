from cascadia_ai.enums import Wildlife
from cascadia_ai.hex_grid import HexGrid, HexPosition
from cascadia_ai.tiles import Tile
from typing import NamedTuple


print_template = [
    r"   /0\   ",
    r" /5   0\ ",
    r"|4  w  1|",
    r"|4     1|",
    r" \3   2/ ",
    r"   \3/   ",
]


class RotatedTile(NamedTuple):
    tile: Tile
    rotation: int

    def get_edge(self, dpos: tuple[int, int]):
        h1, h2 = self.tile.habitats
        rot = self.rotation

        match dpos:
            case (0, 1):
                return h1 if 1 <= rot <= 3 else h2
            case (1, 0):
                return h1 if 2 <= rot <= 4 else h2
            case (1, -1):
                return h1 if 3 <= rot <= 5 else h2
            case (0, -1):
                return h2 if 1 <= rot <= 3 else h1
            case (-1, 0):
                return h2 if 2 <= rot <= 4 else h1
            case (-1, 1):
                return h2 if 3 <= rot <= 5 else h1
            case _:
                raise Exception("Get edge called with non adjacent dpos")


starting_tile_defs: list[tuple[str, str, str]] = [
    ("MMb", "FWhef", "RPsb"),
    ("RRs", "PFseb", "MWfh"),
    ("PPf", "WRshf", "FMbe"),
    ("FFe", "MRheb", "WPfs"),
    ("WWh", "RFseh", "PMbf"),
]

starting_tiles = [
    HexGrid[RotatedTile](
        {
            (0, 0): RotatedTile(Tile.from_definition(a), 0),
            (0, -1): RotatedTile(Tile.from_definition(b), 1),
            (1, -1): RotatedTile(Tile.from_definition(c), 2),
        }
    )
    for a, b, c in starting_tile_defs
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
                    elif char == "w":
                        wildlife = self.wildlife[(q, r)]
                        if wildlife is not None:
                            chars[p] = wildlife.value
                    elif char.isdigit():
                        chars[p] = sides[int(char) - rotation].value
                    else:
                        chars[p] = char

        for y in range(min(p[1] for p in chars), max(p[1] for p in chars) + 1):
            line = ""
            for x in range(min(p[0] for p in chars), max(p[0] for p in chars) + 1):
                p = (x, y)
                line += chars[p] if p in chars else " "
            print(line)
