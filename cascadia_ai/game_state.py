from random import Random
from typing import NamedTuple
from cascadia_ai.enums import Wildlife
from cascadia_ai.environments import Environment, starting_tiles
from cascadia_ai.hex_grid import HexPosition
from cascadia_ai.tiles import Tile, tiles


class Move(NamedTuple):
    tile_index: int
    tile_position: HexPosition
    tile_rotation: int
    wildlife_index: int
    wildlife_position: HexPosition | None


class GameState:
    _seed: int
    _rand: Random

    wildlife_bag: list[Wildlife]
    tile_supply: list[Tile]

    env: Environment
    nature_tokens: int = 0

    def __init__(self, seed: int):
        self._seed = seed
        self._rand = Random(seed)

        self.env = Environment(self._rand.choice(starting_tiles))

        self.tile_supply = list(tiles)
        self._rand.shuffle(self.tile_supply)

        self.wildlife_bag = list(Wildlife) * 20
        self._rand.shuffle(self.wildlife_bag)

        self._check_overpopulation()

    def _check_overpopulation(self):
        while len(set(self.wildlife_bag[:4])) == 1:
            first_four = self.wildlife_bag[:4]
            del self.wildlife_bag[:4]

            for w in first_four:
                index = self._rand.randint(4, len(self.wildlife_bag))
                self.wildlife_bag.insert(index, w)

    @property
    def available_pairs(self):
        return [(self.tile_supply[i], self.wildlife_bag[i]) for i in range(4)]

    @property
    def turns_remaining(self):
        return 23 - len(self.env.tiles)

    def validate_move(self, move: Move):
        if self.turns_remaining <= 0:
            raise Exception("No turns remaining")

        if move.tile_index < 0 or move.tile_index > 3:
            raise Exception("Tile index out of bounds")

        if move.wildlife_index < 0 or move.wildlife_index > 3:
            raise Exception("Wildlife index out of bounds")

        if move.tile_index != move.wildlife_index and self.nature_tokens <= 0:
            raise Exception("Cannot use mismatched pair without nature tokens")

        if self.env.tiles[move.tile_position] is not None:
            raise Exception("Tile already exists at given position")

        if len(list(self.env.tiles.adjacent(move.tile_position))) == 0:
            raise Exception("Tile must be placed adjacent to another tile")

        if move.wildlife_position is not None:
            wildlife = self.wildlife_bag[move.wildlife_index]
            if move.wildlife_position == move.tile_position:
                if wildlife not in self.tile_supply[move.tile_index].wildlife_slots:
                    raise Exception("Target tile does not accept this wildlife type")
            else:
                if self.env.wildlife[move.wildlife_position] is not None:
                    raise Exception("Wildlife already exists at given position")
                wildlife_target = self.env.tiles[move.wildlife_position]
                if wildlife_target is None:
                    raise Exception("Wildlife must be placed on a tile")
                if wildlife not in wildlife_target.tile.wildlife_slots:
                    raise Exception("Target tile does not accept this wildlife type")

    def available_moves(self):
        for tile_index in range(4):
            tile = self.tile_supply[tile_index]

            wildlife_index = tile_index  # TODO add nature token option
            wildlife = self.wildlife_bag[wildlife_index]

            for tile_position in self.env.tiles.all_empty_adjacent():
                wildlife_positions = {None} | {
                    p
                    for p, rtile in self.env.tiles.items()
                    if wildlife in rtile.tile.wildlife_slots
                    and p not in self.env.wildlife
                }

                if wildlife in tile.wildlife_slots:
                    wildlife_positions.add(tile_position)

                for wildlife_position in wildlife_positions:
                    for rotation in range(1 if tile.single_habitat else 6):
                        yield Move(
                            tile_index,
                            tile_position,
                            rotation,
                            wildlife_index,
                            wildlife_position,
                        )

    def make_move(self, move: Move):
        self.validate_move(move)

        if move.tile_index != move.wildlife_index:
            self.nature_tokens -= 1

        tile = self.tile_supply.pop(move.tile_index)
        wildlife = self.wildlife_bag.pop(move.wildlife_index)

        self.env.place_tile(move.tile_position, tile, move.tile_rotation)

        if move.wildlife_position is not None:
            self.env.place_wildlife(move.wildlife_position, wildlife)

        self.tile_supply.pop(0)
        self.wildlife_bag.pop(0)

        self._check_overpopulation()
