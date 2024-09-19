from typing import NamedTuple
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import Environment, HexPosition, RotatedTile, hex_steps
from cascadia_ai.game_state import GameState


def share_edge(
    habitat: Habitat,
    pos1: HexPosition,
    rtile1: RotatedTile,
    pos2: HexPosition,
    rtile2: RotatedTile,
):
    q1, r1 = pos1
    q2, r2 = pos2
    edge1 = rtile1.get_edge((q2 - q1, r2 - r1))
    edge2 = rtile2.get_edge((q1 - q2, r1 - r2))
    return edge1 == edge2 == habitat


def calculate_habitat_score(habitat: Habitat, env: Environment):
    # TODO consider adding "habitat_groups()" method to env
    groups: dict[HexPosition, set[HexPosition]] = {}

    for pos, rtile in env.tiles.items():
        if habitat not in rtile.tile.habitats:
            continue

        groups[pos] = {pos} | {
            apos
            for apos, artile in env.adjacent_tiles(pos)
            if share_edge(habitat, pos, rtile, apos, artile)
        }

    for pos in list(groups.keys()):
        merged = False
        for other_pos, other_group in groups.items():
            if pos != other_pos and pos in other_group:
                groups[other_pos] = other_group | groups[pos]
                merged = True
        if merged:
            del groups[pos]

    score = 0 if len(groups) == 0 else max(len(group) for group in groups.values())

    if score >= 7:
        score += 2

    return score


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


class Score(NamedTuple):
    wildlife: dict[str, int]
    habitat: dict[str, int]
    nature_tokens: int
    total: int


def calculate_score(gs: GameState):
    wildlife = {
        Wildlife.BEAR.value: calculate_bear_score(gs.env),
        Wildlife.ELK.value: calculate_elk_score(gs.env),
        Wildlife.SALMON.value: calculate_salmon_score(gs.env),
        Wildlife.HAWK.value: calculate_hawk_score(gs.env),
        Wildlife.FOX.value: calculate_fox_score(gs.env),
    }

    habitat = {h.value: calculate_habitat_score(h, gs.env) for h in Habitat}

    return Score(
        wildlife=wildlife,
        habitat=habitat,
        nature_tokens=gs.nature_tokens,
        total=sum(wildlife.values()) + sum(habitat.values()) + gs.nature_tokens,
    )
