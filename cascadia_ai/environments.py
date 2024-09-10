from cascadia_ai.enums import Wildlife
from cascadia_ai.hex_grid import HexGrid, HexPosition
from cascadia_ai.tiles import Tile
from typing import NamedTuple


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
            (20, 21): RotatedTile(Tile.from_definition(a), 0),
            (20, 20): RotatedTile(Tile.from_definition(b), 1),
            (21, 20): RotatedTile(Tile.from_definition(c), 2),
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
            raise ValueError("Placed tile must be adjacent to another tile")
        self.tiles[position] = RotatedTile(tile, rotation)

    def place_wildlife(self, position: HexPosition, wildlife: Wildlife):
        rtile = self.tiles[position]
        if rtile is None:
            raise ValueError("Cannot place wildlife on a position with no tile")
        if wildlife not in rtile.tile.wildlife_slots:
            raise ValueError("Tile does not accept the given wildlife type")
        self.wildlife[position] = wildlife
