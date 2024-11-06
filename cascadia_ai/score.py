from collections.abc import Set
from typing import Generator, NamedTuple
from cascadia_ai.enums import Habitat, Wildlife
# from cascadia_ai.environment import Environment
from cascadia_ai.positions import HexPosition, adjacent_positions


class ScoreTable:
    def __init__(self, data: list[int]):
        self._data = data

    def __getitem__(self, i: int):
        return self._data[i if i < len(self._data) else -1]


scoring_bear_pairs = ScoreTable([0, 4, 11, 19, 27])
scoring_elk_line = ScoreTable([0, 2, 5, 9, 13])
scoring_salmon_run = ScoreTable([0, 2, 5, 8, 12, 16, 20, 25])
scoring_hawk_singles = ScoreTable([0, 2, 5, 8, 11, 14, 18, 22, 26])


# def calculate_habitat_scores(env: Environment):
#     largest_group_sizes = {
#         h: len(state[0] if len(state) else [])
#         for h, state in env.habitat_groups().items()
#     }
#     return {h: v + 2 if v >= 7 else v for h, v in largest_group_sizes.items()}


# def calculate_bear_score(env: Environment):
#     groups = env.wildlife_groups(Wildlife.BEAR)
#     num_valid_groups = sum(len(g) == 2 for g in groups)
#     return scoring_bear_pairs[num_valid_groups]


hex_steps: list[tuple[int, int]] = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]


# TODO try caching this globally
def iter_elk_line_options(
    group: Set[HexPosition], acc: list[int] = []
) -> Generator[list[int], None, None]:
    if len(group) == 0:
        yield acc

    for starting_point in group:
        adjacent_steps = [
            step
            for step in hex_steps
            if ((starting_point[0] + step[0], starting_point[1] + step[1]) in group)
        ]

        if len(adjacent_steps) == 0:
            yield acc + [1]

        for dq, dr in adjacent_steps:
            q, r = starting_point
            line = set[HexPosition]()

            while (q, r) in group:
                line.add((q, r))
                q += dq
                r += dr

            yield from iter_elk_line_options(group - line, acc + [len(line)])


def get_max_elk_group_score(group: Set[HexPosition]):
    return max(
        sum(scoring_elk_line[line] for line in lines)
        for lines in iter_elk_line_options(group)
    )


# def calculate_elk_score(env: Environment):
#     return sum(
#         get_max_elk_group_score(group) for group in env.wildlife_groups(Wildlife.ELK)
#     )


def is_valid_salmon_run(group: set[HexPosition]):
    for pos in group:
        num_adj = 0
        for adj in adjacent_positions(pos):
            if adj in group:
                num_adj += 1
            if num_adj > 2:
                return False
    return True


# def calculate_salmon_score(env: Environment):
#     return sum(
#         scoring_salmon_run[len(group)]
#         for group in env.wildlife_groups(Wildlife.SALMON)
#         if is_valid_salmon_run(group)
#     )


# def calculate_hawk_score(env: Environment):
#     num_valid_groups = sum(len(g) == 1 for g in env.wildlife_groups(Wildlife.HAWK))
#     return scoring_hawk_singles[num_valid_groups]


# def calculate_fox_score(env: Environment):
#     return sum(
#         len(env.adjacent_wildlife_counts(pos))
#         for pos, w in env.wildlife.items()
#         if w == Wildlife.FOX
#     )


# def calculate_wildlife_score(env: Environment):
#     return {
#         Wildlife.BEAR: calculate_bear_score(env),
#         Wildlife.ELK: calculate_elk_score(env),
#         Wildlife.SALMON: calculate_salmon_score(env),
#         Wildlife.HAWK: calculate_hawk_score(env),
#         Wildlife.FOX: calculate_fox_score(env),
#     }


class Score(NamedTuple):
    wildlife: dict[Wildlife, int]
    habitat: dict[Habitat, int]
    nature_tokens: int

    @property
    def total(self):
        return (
            sum(self.wildlife.values())
            + sum(self.habitat.values())
            + self.nature_tokens
        )

    def __repr__(self):
        return ", ".join(
            [
                *(f"{k.value}:{v:>2}" for k, v in self.wildlife.items()),
                *(f"{k.value}:{v:>2}" for k, v in self.habitat.items()),
                f"nt:{self.nature_tokens:>2}",
                f"total:{self.total:>3}",
            ]
        )
