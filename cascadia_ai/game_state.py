from copy import deepcopy
from random import Random
from typing import NamedTuple
from cascadia_ai.enums import Wildlife
from cascadia_ai.environments import Environment, HexPosition, starting_tiles
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

    history: list[Move]

    def __init__(self, seed: int):
        self._seed = seed
        self._rand = Random(seed)

        self.history = []

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
        return 23 - self.env.num_tiles_placed

    def validate_move(self, move: Move):
        if self.turns_remaining <= 0:
            raise Exception("No turns remaining")

        if move.tile_index not in range(4):
            raise Exception("Tile index out of bounds")

        if move.wildlife_index not in range(4):
            raise Exception("Wildlife index out of bounds")

        if move.tile_index != move.wildlife_index and self.nature_tokens <= 0:
            raise Exception("Cannot use mismatched pair without nature tokens")

        if not self.env.can_accept_tile(move.tile_position):
            raise Exception("Cannot place tile there")

        if move.wildlife_position is not None:
            wildlife = self.wildlife_bag[move.wildlife_index]

            if move.wildlife_position == move.tile_position:
                if wildlife not in self.tile_supply[move.tile_index].wildlife_slots:
                    raise Exception("Cannot place wildlife there")

            else:
                if not self.env.can_accept_wildlife(move.wildlife_position, wildlife):
                    raise Exception("Cannot place wildlife there")

    def available_moves(self):
        for tile_index in range(4):
            tile = self.tile_supply[tile_index]

            wildlife_index = tile_index  # TODO add nature token option
            wildlife = self.wildlife_bag[wildlife_index]

            for tile_position in self.env.all_adjacent_empty():
                wildlife_positions = {None} | {
                    p
                    for p, rtile in self.env.unoccupied_tiles()
                    if wildlife in rtile.tile.wildlife_slots
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

        new_gs = deepcopy(self)

        if move.tile_index != move.wildlife_index:
            new_gs.nature_tokens -= 1

        tile = new_gs.tile_supply.pop(move.tile_index)
        wildlife = new_gs.wildlife_bag.pop(move.wildlife_index)

        new_gs.env.place_tile(move.tile_position, tile, move.tile_rotation)

        if move.wildlife_position is not None:
            new_gs.env.place_wildlife(move.wildlife_position, wildlife)

            if new_gs.env.tiles[move.wildlife_position].tile.nature_token_reward:
                new_gs.nature_tokens += 1

        new_gs.tile_supply.pop(0)
        new_gs.wildlife_bag.pop(0)

        new_gs._check_overpopulation()

        new_gs.history.append(move)

        return new_gs
