from collections.abc import Set
from typing import Generator, NamedTuple
from cascadia_ai.enums import Habitat, Wildlife
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


hex_steps: list[tuple[int, int]] = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]


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


# TODO try caching this
def get_max_elk_group_score(group: Set[HexPosition]):
    if len(group) == 1:
        return scoring_elk_line[1]

    if len(group) == 2:
        return scoring_elk_line[2]

    return max(
        sum(scoring_elk_line[line] for line in lines)
        for lines in iter_elk_line_options(group)
    )


def is_valid_salmon_run(group: Set[HexPosition]):
    for pos in group:
        num_adj = 0
        for adj in adjacent_positions(pos):
            if adj in group:
                num_adj += 1
            if num_adj > 2:
                return False
    return True


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
