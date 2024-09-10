from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import RotatedTile
from cascadia_ai.game_state import GameState
from cascadia_ai.hex_grid import HexGrid, HexPosition, hex_steps


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


def calculate_habitat_score(habitat: Habitat, tiles: HexGrid[RotatedTile]):
    groups: dict[HexPosition, set[HexPosition]] = {}

    for pos, rtile in tiles.items():
        if habitat not in rtile.tile.habitats:
            continue

        groups[pos] = {pos} | {
            apos
            for apos, artile in tiles.adjacent(pos)
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


def find_groups(grid: HexGrid[Wildlife]):
    visited = set[HexPosition]()
    groups = list[set[HexPosition]]()

    for p in grid.keys():
        if p in visited:
            continue

        current_group = set[HexPosition]()
        unexplored = {p}

        while len(unexplored):
            next = unexplored.pop()
            current_group.add(next)
            adjacent = set(a for a, _ in grid.adjacent(next))
            unexplored |= adjacent - current_group

        visited.update(current_group)
        groups.append(current_group)

    return groups


def calculate_bear_score(wgrid: HexGrid[Wildlife]):
    groups = find_groups(wgrid.filter(Wildlife.BEAR))
    num_valid_groups = sum(len(g) == 2 for g in groups)
    return score_lookup([0, 4, 11, 19, 27], num_valid_groups)


def iter_elk_line_options(points: set[HexPosition], acc: list[int] = []):
    if len(points) == 0:
        yield acc

    for starting_point in points:
        adjacent_steps = [
            step
            for step in hex_steps
            if ((starting_point[0] + step[0], starting_point[1] + step[1]) in points)
        ]

        if len(adjacent_steps) == 0:
            yield acc + [1]

        for dq, dr in adjacent_steps:
            q, r = starting_point
            line = set[HexPosition]()

            while (q, r) in points:
                line.add((q, r))
                q += dq
                r += dr

            yield from iter_elk_line_options(points - line, acc + [len(line)])


def calculate_elk_score(wgrid: HexGrid[Wildlife]):
    total_score = 0

    groups = find_groups(wgrid.filter(Wildlife.ELK))

    for group in groups:
        max_group_score = 0

        for lines in iter_elk_line_options(group):
            max_group_score = max(
                max_group_score,
                sum(score_lookup([0, 2, 5, 9, 13], line) for line in lines),
            )

        total_score += max_group_score

    return total_score


def calculate_salmon_score(wgrid: HexGrid[Wildlife]):
    salmon_score = 0

    salmon = wgrid.filter(Wildlife.SALMON)

    for group in find_groups(salmon):
        valid_run = all(len(list(salmon.adjacent(p))) <= 2 for p in group)
        if valid_run:
            salmon_score += score_lookup([0, 2, 5, 8, 12, 16, 20, 25], len(group))

    return salmon_score


def calculate_hawk_score(wgrid: HexGrid[Wildlife]):
    groups = find_groups(wgrid.filter(Wildlife.HAWK))
    num_valid_groups = sum(len(g) == 1 for g in groups)
    return score_lookup([0, 2, 5, 8, 11, 14, 18, 22, 26], num_valid_groups)


def calculate_fox_score(wgrid: HexGrid[Wildlife]):
    fox_score = 0
    for p, w in wgrid.items():
        if w == Wildlife.FOX:
            unique_adjacent = {w2 for _, w2 in wgrid.adjacent(p)}
            fox_score += len(unique_adjacent)
    return fox_score


def calculate_score(gs: GameState):
    wildlife = {
        Wildlife.BEAR.value: calculate_bear_score(gs.env.wildlife),
        Wildlife.ELK.value: calculate_elk_score(gs.env.wildlife),
        Wildlife.SALMON.value: calculate_salmon_score(gs.env.wildlife),
        Wildlife.HAWK.value: calculate_hawk_score(gs.env.wildlife),
        Wildlife.FOX.value: calculate_fox_score(gs.env.wildlife),
    }

    habitat = {h.value: calculate_habitat_score(h, gs.env.tiles) for h in Habitat}

    return {
        "wildlife": wildlife,
        "habitat": habitat,
        "nature_tokens": gs.nature_tokens,
        "total": sum(wildlife.values()) + sum(habitat.values()) + gs.nature_tokens,
    }
