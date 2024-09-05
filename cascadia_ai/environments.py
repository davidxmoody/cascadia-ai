from cascadia_ai.enums import Wildlife
from cascadia_ai.hex_grid import HexGrid
from cascadia_ai.tiles import Tile
from typing import NamedTuple


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
