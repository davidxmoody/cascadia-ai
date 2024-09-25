from typing import NamedTuple
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import Environment, HexPosition, hex_steps
from cascadia_ai.game_state import GameState


def calculate_habitat_scores(env: Environment):
    largest_group_sizes = {
        h: len(state[0] if len(state) else []) for h, state in env.habitat_groups().items()
    }
    return {h: v + 2 if v >= 7 else v for h, v in largest_group_sizes.items()}


def score_lookup(score_table: list[int], value: int):
    return score_table[min(value, len(score_table) - 1)]


def calculate_bear_score(env: Environment):
    groups = env.wildlife_groups(Wildlife.BEAR)
    num_valid_groups = sum(len(g) == 2 for g in groups)
    return score_lookup([0, 4, 11, 19, 27], num_valid_groups)


def iter_elk_line_options(group: set[HexPosition], acc: list[int] = []):
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


def calculate_elk_score(env: Environment):
    total_score = 0

    for group in env.wildlife_groups(Wildlife.ELK):
        max_group_score = 0

        for lines in iter_elk_line_options(group):
            max_group_score = max(
                max_group_score,
                sum(score_lookup([0, 2, 5, 9, 13], line) for line in lines),
            )

        total_score += max_group_score

    return total_score


def calculate_salmon_score(env: Environment):
    return sum(
        score_lookup([0, 2, 5, 8, 12, 16, 20, 25], len(group))
        for group in env.wildlife_groups(Wildlife.SALMON)
        if all(env.adjacent_wildlife_counts(pos)[Wildlife.SALMON] <= 2 for pos in group)
    )


def calculate_hawk_score(env: Environment):
    num_valid_groups = sum(len(g) == 1 for g in env.wildlife_groups(Wildlife.HAWK))
    return score_lookup([0, 2, 5, 8, 11, 14, 18, 22, 26], num_valid_groups)


def calculate_fox_score(env: Environment):
    return sum(
        len(env.adjacent_wildlife_counts(pos))
        for pos, w in env.wildlife.items()
        if w == Wildlife.FOX
    )


def calculate_wildlife_score(env: Environment):
    return {
        Wildlife.BEAR: calculate_bear_score(env),
        Wildlife.ELK: calculate_elk_score(env),
        Wildlife.SALMON: calculate_salmon_score(env),
        Wildlife.HAWK: calculate_hawk_score(env),
        Wildlife.FOX: calculate_fox_score(env),
    }


class Score(NamedTuple):
    wildlife: dict[Wildlife, int]
    habitat: dict[Habitat, int]
    nature_tokens: int
    total: int


def calculate_score(state: GameState):
    wildlife = calculate_wildlife_score(state.env)

    habitat = calculate_habitat_scores(state.env)

    return Score(
        wildlife=wildlife,
        habitat=habitat,
        nature_tokens=state.nature_tokens,
        total=sum(wildlife.values()) + sum(habitat.values()) + state.nature_tokens,
    )
