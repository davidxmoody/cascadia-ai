from cascadia_ai.enums import Wildlife
from cascadia_ai.tiles import Tile, tiles


class GameState:
    seed: int
    wildlife_bag: dict[Wildlife, int]
    tile_supply: list[Tile]
    display: list[tuple[Tile, Wildlife]]
    board: dict[tuple[int, int], tuple[Tile, int, Wildlife | None]]
    nature_tokens: int = 0

    def __init__(self, seed: int):
        self.seed = seed
        self.wildlife_bag = {w: 20 for w in Wildlife}
        self.tile_supply = list(tiles)

        # TODO finish init
