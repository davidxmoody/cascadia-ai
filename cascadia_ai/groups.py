from copy import copy
from typing import Self
from cascadia_ai.ai.actions import (
    calculate_bear_reward,
    calculate_elk_reward,
    calculate_hawk_reward,
    calculate_salmon_reward,
)
from cascadia_ai.enums import Wildlife
from cascadia_ai.positions import HexPosition, adjacent_positions

GroupLabel = int


class WildlifeGroups:
    wildlife: Wildlife
    count: int
    groups: list[frozenset[HexPosition]]
    _child_cache: dict[HexPosition, Self]

    def __init__(
        self, wildlife: Wildlife, wgrid: dict[HexPosition, Wildlife] | None = None
    ):
        self.wildlife = wildlife
        self.count = 0
        self.groups = []
        self._child_cache = {}

        if wgrid is not None:
            for pos, w in wgrid.items():
                if w == wildlife:
                    self._place_wildlife(pos)

    def get_reward(self, pos: HexPosition):
        match self.wildlife:
            case Wildlife.BEAR:
                return calculate_bear_reward(self.groups, pos) or 0

            case Wildlife.ELK:
                return calculate_elk_reward(self.groups, pos) or 0

            case Wildlife.SALMON:
                return calculate_salmon_reward(self.groups, pos) or 0

            case Wildlife.HAWK:
                return calculate_hawk_reward(self.groups, pos) or 0

            case Wildlife.FOX:
                return 0
                # raise Exception("Fox reward not supported")

    def _place_wildlife(self, pos: HexPosition):
        self.count += 1

        adjacent_pos = adjacent_positions(pos)

        new_groups: list[frozenset[HexPosition]] = []
        new_group = {pos}

        for group in self.groups:
            if group.isdisjoint(adjacent_pos):
                new_groups.append(group)
            else:
                new_group.update(group)

        new_groups.append(frozenset(new_group))
        new_groups.sort(key=len, reverse=True)

        self.groups = new_groups

    def place_wildlife(self, pos: HexPosition, wildlife: Wildlife):
        if wildlife != self.wildlife:
            return self

        if pos not in self._child_cache:
            new_obj = self.copy()
            new_obj._place_wildlife(pos)
            self._child_cache[pos] = new_obj

        return self._child_cache[pos]

    def copy(self):
        new_obj = copy(self)
        new_obj.groups = list(new_obj.groups)
        new_obj._child_cache = {}
        return new_obj
