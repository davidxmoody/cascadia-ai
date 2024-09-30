from copy import deepcopy
from random import choice, sample
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
    _tile_supply: list[Tile]
    _wildlife_supply: dict[Wildlife, int]

    tile_display: list[Tile]
    wildlife_display: list[Wildlife]

    env: Environment
    nature_tokens: int = 0

    def __init__(self):
        self.env = Environment(choice(starting_tiles))

        self._tile_supply = tiles[:]
        self.tile_display = self._draw_tiles(4)

        self._wildlife_supply = {w: 20 for w in Wildlife}
        self.wildlife_display = self._draw_wildlife(4)

        self._check_overpopulation()

    def _draw_tiles(self, num=1):
        result = sample(self._tile_supply, k=num)
        for t in result:
            self._tile_supply.remove(t)
        return result

    def _draw_wildlife(self, num=1):
        result = sample(
            list(self._wildlife_supply.keys()),
            counts=list(self._wildlife_supply.values()),
            k=num,
        )
        for w in result:
            self._wildlife_supply[w] -= 1
        return result

    def _check_overpopulation(self):
        while len(set(self.wildlife_display)) == 1:
            self._replace_wildlife(list(range(4)))

    def _replace_wildlife(self, indexes: list[int]):
        removed = [self.wildlife_display[i] for i in indexes]
        for i in indexes:
            self.wildlife_display[i] = self._draw_wildlife()[0]
        for w in removed:
            self._wildlife_supply[w] += 1

    @property
    def turns_remaining(self):
        return 23 - self.env.num_tiles_placed

    def copy(self):
        return deepcopy(self)

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
            wildlife = self.wildlife_display[action.wildlife_index]

            if action.wildlife_position == action.tile_position:
                if wildlife not in self.tile_display[action.tile_index].wildlife_slots:
                    raise Exception("Cannot place wildlife there")

            else:
                if not self.env.can_place_wildlife(action.wildlife_position, wildlife):
                    raise Exception("Cannot place wildlife there")

    # TODO remove this method
    def available_actions(self):
        actions: list[Action] = []

        for tile_index in range(4):
            tile = self.tile_display[tile_index]

            wildlife_index = tile_index  # TODO add nature token option
            wildlife = self.wildlife_display[wildlife_index]

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

    def take_action(self, action: Action):
        self.validate_action(action)

        if action.tile_index != action.wildlife_index:
            self.nature_tokens -= 1

        tile = self.tile_display.pop(action.tile_index)
        wildlife = self.wildlife_display.pop(action.wildlife_index)

        self.env.place_tile(action.tile_position, tile.rotate(action.tile_rotation))

        if action.wildlife_position is not None:
            self.env.place_wildlife(action.wildlife_position, wildlife)

            if self.env.tiles[action.wildlife_position].nature_token_reward:
                self.nature_tokens += 1

        self.tile_display.pop(0)
        self.tile_display.extend(self._draw_tiles(2))

        self.wildlife_display.pop(0)
        self.wildlife_display.extend(self._draw_wildlife(2))

        self._check_overpopulation()
