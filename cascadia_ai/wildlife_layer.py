from copy import copy
from typing import Self
from cascadia_ai.enums import Wildlife
from cascadia_ai.positions import HexPosition, adjacent_positions
from cascadia_ai.score import (
    get_max_elk_group_score,
    is_valid_salmon_run,
    scoring_bear_pairs,
    scoring_salmon_run,
    scoring_hawk_singles,
)


class WildlifeLayer:
    score: int
    count: int
    _wildlife: Wildlife
    _groups: list[frozenset[HexPosition]]
    _reward_cache: dict[HexPosition, int]
    _child_cache: dict[HexPosition, Self]

    def __init__(
        self,
        wildlife: Wildlife,
        wgrid: dict[HexPosition, Wildlife] | None = None,
    ):
        self.score = 0
        self.count = 0
        self._wildlife = wildlife
        self._groups = []
        self._reward_cache = {}
        self._child_cache = {}

        if wgrid is not None:
            for pos, w in wgrid.items():
                if w == wildlife:
                    self.score += self.get_reward(pos, w)
                    self._place_wildlife(pos)
            self._reward_cache = {}

    def get_reward(self, pos: HexPosition, wildlife: Wildlife):
        if wildlife != self._wildlife:
            return 0

        if pos in self._reward_cache:
            return self._reward_cache[pos]

        connected_groups = self._get_connected_groups(pos)

        match self._wildlife:
            case Wildlife.BEAR:
                if len(connected_groups) == 0:
                    reward = 0

                else:
                    num_bear_pairs_before = sum(len(g) == 2 for g in self._groups)
                    num_bear_pairs_after = num_bear_pairs_before

                    if len(connected_groups) == 1 and len(connected_groups[0]) == 1:
                        num_bear_pairs_after += 1

                    else:
                        for group in connected_groups:
                            if len(group) == 2:
                                num_bear_pairs_after -= 1

                    reward = (
                        scoring_bear_pairs[num_bear_pairs_after]
                        - scoring_bear_pairs[num_bear_pairs_before]
                    )

            case Wildlife.ELK:
                partial_score_before = sum(
                    get_max_elk_group_score(group) for group in connected_groups
                )

                partial_score_after = get_max_elk_group_score(
                    set.union({pos}, *connected_groups)
                )

                reward = partial_score_after - partial_score_before

            case Wildlife.SALMON:
                new_group = set.union({pos}, *connected_groups)

                if not is_valid_salmon_run(new_group):
                    reward = -1 * sum(
                        scoring_salmon_run[len(group)]
                        for group in connected_groups
                        if is_valid_salmon_run(group)
                    )

                else:
                    partial_score_before = sum(
                        scoring_salmon_run[len(group)] for group in connected_groups
                    )

                    partial_score_after = scoring_salmon_run[len(new_group)]

                    reward = partial_score_after - partial_score_before

            case Wildlife.HAWK:
                num_singles_before = sum(len(g) == 1 for g in self._groups)
                num_singles_after = num_singles_before

                if len(connected_groups):
                    num_singles_after -= sum(len(g) == 1 for g in connected_groups)
                else:
                    num_singles_after += 1

                reward = (
                    scoring_hawk_singles[num_singles_after]
                    - scoring_hawk_singles[num_singles_before]
                )

            case Wildlife.FOX:
                raise Exception("Fox reward not supported")

        self._reward_cache[pos] = reward
        return reward

    def _get_connected_groups(self, pos: HexPosition):
        adjacent_pos = adjacent_positions(pos)
        return [group for group in self._groups if not group.isdisjoint(adjacent_pos)]

    def _place_wildlife(self, pos: HexPosition):
        self.count += 1

        adjacent_pos = adjacent_positions(pos)

        new_groups: list[frozenset[HexPosition]] = []
        new_group = {pos}

        for group in self._groups:
            if group.isdisjoint(adjacent_pos):
                new_groups.append(group)
            else:
                new_group.update(group)

        new_groups.append(frozenset(new_group))
        # TODO is this necessary?
        # new_groups.sort(key=len, reverse=True)

        self._groups = new_groups

    def place_wildlife(self, pos: HexPosition, wildlife: Wildlife):
        if wildlife != self._wildlife:
            return self

        if pos not in self._child_cache:
            reward = self.get_reward(pos, wildlife)
            if reward is None:
                raise Exception("Attempted to place wildlife with None reward")
            new_obj = self.copy()
            new_obj._place_wildlife(pos)
            new_obj.score += reward
            self._child_cache[pos] = new_obj

        return self._child_cache[pos]

    def copy(self):
        new_obj = copy(self)
        new_obj._groups = list(new_obj._groups)
        new_obj._reward_cache = {}
        new_obj._child_cache = {}
        return new_obj


class FoxLayer:
    score: int
    count: int
    _wgrid: dict[HexPosition, Wildlife]
    _reward_cache: dict[tuple[HexPosition, Wildlife], int]
    _child_cache: dict[tuple[HexPosition, Wildlife], Self]

    def __init__(self, wgrid: dict[HexPosition, Wildlife] | None = None):
        self.score = 0
        self.count = 0
        self._wgrid = {}
        self._reward_cache = {}
        self._child_cache = {}

        if wgrid is not None:
            for pos, w in wgrid.items():
                self.score += self.get_reward(pos, w)
                self._place_wildlife(pos, w)
            self._reward_cache = {}

    def get_reward(self, pos: HexPosition, wildlife: Wildlife):
        key = (pos, wildlife)
        if key in self._reward_cache:
            return self._reward_cache[key]

        reward = 0

        adjacent_pos = adjacent_positions(pos)

        if wildlife == Wildlife.FOX:
            reward += len(
                set(self._wgrid[apos] for apos in adjacent_pos if apos in self._wgrid)
            )

        for apos in adjacent_pos:
            if self._wgrid.get(apos) == Wildlife.FOX:
                if not any(
                    self._wgrid.get(apos2) == wildlife
                    for apos2 in adjacent_positions(apos)
                ):
                    reward += 1

        self._reward_cache[key] = reward
        return reward

    def _place_wildlife(self, pos: HexPosition, wildlife: Wildlife):
        if wildlife == Wildlife.FOX:
            self.count += 1
        self._wgrid[pos] = wildlife

    def place_wildlife(self, pos: HexPosition, wildlife: Wildlife):
        key = (pos, wildlife)
        if key not in self._child_cache:
            reward = self.get_reward(pos, wildlife)
            new_obj = self.copy()
            new_obj._place_wildlife(pos, wildlife)
            new_obj.score += reward
            self._child_cache[key] = new_obj
        return self._child_cache[key]

    def copy(self):
        new_obj = copy(self)
        new_obj._wgrid = dict(new_obj._wgrid)
        new_obj._reward_cache = {}
        new_obj._child_cache = {}
        return new_obj
