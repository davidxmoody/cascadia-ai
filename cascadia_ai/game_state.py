from random import Random
from cascadia_ai.enums import Wildlife
from cascadia_ai.environments import Environment, starting_tiles
from cascadia_ai.tiles import Tile, tiles


class GameState:
    _seed: int
    _rand: Random

    wildlife_bag: list[Wildlife]
    tile_supply: list[Tile]

    environment: Environment
    nature_tokens: int = 0

    def __init__(self, seed: int):
        self._seed = seed
        self._rand = Random(seed)

        self.wildlife_bag = list(Wildlife) * 20
        self._rand.shuffle(self.wildlife_bag)

        self.tile_supply = list(tiles)
        self._rand.shuffle(self.tile_supply)

        self.environment = Environment(self._rand.choice(starting_tiles))

    @property
    def available_pairs(self):
        return [(self.tile_supply[i], self.wildlife_bag[i]) for i in range(4)]
