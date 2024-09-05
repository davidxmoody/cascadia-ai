from cascadia_ai.enums import Wildlife
from cascadia_ai.tiles import Tile
from typing import NamedTuple


class HexPosition(NamedTuple):
    q: int
    r: int


class RotatedTile(NamedTuple):
    tile: Tile
    rotation: int


starting_tiles: list[dict[HexPosition, RotatedTile]] = [
    {
        HexPosition(0, 0): RotatedTile(Tile.from_definition("MMb"), 0),
        HexPosition(0, -1): RotatedTile(Tile.from_definition("FWehf"), 1),
        HexPosition(1, -1): RotatedTile(Tile.from_definition("RPbs"), 2),
    }
]


class Environment:
    tiles: dict[HexPosition, RotatedTile]
    wildlife: dict[HexPosition, Wildlife]

    def __init__(self, starting_tile_group: dict[HexPosition, RotatedTile]):
        self.tiles = starting_tile_group.copy()
        self.wildlife = {}
