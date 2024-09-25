from collections import Counter
from copy import deepcopy
from random import Random, choice, randint
from typing import NamedTuple
from cascadia_ai.enums import Wildlife
from cascadia_ai.environments import Environment, HexPosition, starting_tiles
from cascadia_ai.tiles import Tile, tiles


class Action(NamedTuple):
    tile_index: int
    tile_position: HexPosition
    tile_rotation: int
    wildlife_index: int
    wildlife_position: HexPosition | None


class GameState:
    _seed: int
    _rand: Random

    # TODO make this hidden and only expose the currently displayed 4
    wildlife_bag: list[Wildlife]
    tile_supply: list[Tile]

    env: Environment
    nature_tokens: int = 0

    def __init__(self, seed: int | None = None):
        self._seed = seed if seed is not None else randint(0, 2**32)
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
        return 23 - self.env.num_tiles_placed

    def wildlife_in_supply(self):
        return Counter(self.wildlife_bag[4:])

    def tiles_in_supply(self):
        # TODO maybe do something to ensure that there's no cheating by looking at the order
        return self.tile_supply[4:]

    def validate_action(self, action: Action):
        if self.turns_remaining <= 0:
            raise Exception("No turns remaining")

        if action.tile_index not in range(4):
            raise Exception("Tile index out of bounds")

        if action.wildlife_index not in range(4):
            raise Exception("Wildlife index out of bounds")

        if action.tile_index != action.wildlife_index and self.nature_tokens <= 0:
            raise Exception("Cannot use mismatched pair without nature tokens")

        if not self.env.can_place_tile(action.tile_position):
            raise Exception("Cannot place tile there")

        if action.wildlife_position is not None:
            wildlife = self.wildlife_bag[action.wildlife_index]

            if action.wildlife_position == action.tile_position:
                if wildlife not in self.tile_supply[action.tile_index].wildlife_slots:
                    raise Exception("Cannot place wildlife there")

            else:
                if not self.env.can_place_wildlife(action.wildlife_position, wildlife):
                    raise Exception("Cannot place wildlife there")

    def available_actions(self):
        actions: list[Action] = []

        for tile_index in range(4):
            tile = self.tile_supply[tile_index]

            wildlife_index = tile_index  # TODO add nature token option
            wildlife = self.wildlife_bag[wildlife_index]

            for tile_position in self.env.all_adjacent_empty():
                wildlife_positions = {None} | {
                    p
                    for p, tile in self.env.unoccupied_tiles()
                    if wildlife in tile.wildlife_slots
                }

                if wildlife in tile.wildlife_slots:
                    wildlife_positions.add(tile_position)

                for wildlife_position in wildlife_positions:
                    for rotation in range(1 if tile.single_habitat else 6):
                        actions.append(
                            Action(
                                tile_index,
                                tile_position,
                                rotation,
                                wildlife_index,
                                wildlife_position,
                            )
                        )

        return actions

    def get_random_action(self):
        tile_index = choice(range(4))
        tile = self.tile_supply[tile_index]

        wildlife_index = tile_index  # TODO add nature token option
        wildlife = self.wildlife_bag[wildlife_index]

        tile_position = choice(list(self.env.all_adjacent_empty()))

        wildlife_positions = [
            p
            for p, tile in self.env.unoccupied_tiles()
            if wildlife in tile.wildlife_slots
        ]
        if wildlife in tile.wildlife_slots:
            wildlife_positions.append(tile_position)

        wildlife_position = (
            choice(wildlife_positions) if len(wildlife_positions) else None
        )

        rotation = choice(range(1 if tile.single_habitat else 6))

        return Action(
            tile_index,
            tile_position,
            rotation,
            wildlife_index,
            wildlife_position,
        )

    def get_all_next_states(self):
        return [self.take_action(action) for action in self.available_actions()]

    def get_random_next_state(self):
        return self.take_action(choice(self.available_actions()))

    def take_action(self, action: Action):
        self.validate_action(action)

        new_state = deepcopy(self)

        if action.tile_index != action.wildlife_index:
            new_state.nature_tokens -= 1

        tile = new_state.tile_supply.pop(action.tile_index)
        wildlife = new_state.wildlife_bag.pop(action.wildlife_index)

        new_state.env.place_tile(action.tile_position, tile.rotate(action.tile_rotation))

        if action.wildlife_position is not None:
            new_state.env.place_wildlife(action.wildlife_position, wildlife)

            if new_state.env.tiles[action.wildlife_position].nature_token_reward:
                new_state.nature_tokens += 1

        new_state.tile_supply.pop(0)
        new_state.wildlife_bag.pop(0)

        new_state._check_overpopulation()

        return new_state
