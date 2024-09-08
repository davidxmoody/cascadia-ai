from typing import Counter
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import RotatedTile
from cascadia_ai.game_state import GameState
from cascadia_ai.hex_grid import HexGrid, HexPosition


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


def calculate_habitat_score(tiles: HexGrid[RotatedTile]):
    score: dict[Habitat, int] = {}

    for habitat in Habitat:
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

        score[habitat] = (
            0 if len(groups) == 0 else max(len(group) for group in groups.values())
        )

        if score[habitat] >= 7:
            score[habitat] += 2

    return score


def calculate_bear_score(wildlife: HexGrid[Wildlife]):
    bears = {p for p, w in wildlife.items() if w == Wildlife.BEAR}
    potential_pairs = Counter[tuple[HexPosition, HexPosition]]()

    for p in bears:
        adjacent = [a for a, _ in wildlife.adjacent(p) if a in bears]
        if len(adjacent) == 1:
            p1, p2 = sorted([p, adjacent[0]])
            potential_pairs[(p1, p2)] += 1

    num_real_pairs = sum(1 for count in potential_pairs.values() if count == 2)

    bear_scores = [0, 4, 11, 19, 27]

    return bear_scores[min(num_real_pairs, len(bear_scores) - 1)]


def calculate_elk_score(wildlife: HexGrid[Wildlife]):
    return 0


def calculate_salmon_score(wildlife: HexGrid[Wildlife]):
    salmon_scores = [0, 2, 5, 8, 12, 16, 20, 25]

    total_salmon_score = 0

    salmon = {p for p, w in wildlife.items() if w == Wildlife.SALMON}

    visited = set[HexPosition]()

    for starting_point in salmon:
        if starting_point in visited:
            continue

        valid_run = True
        current_run = set[HexPosition]()
        unexplored = {starting_point}

        while len(unexplored):
            next = unexplored.pop()
            adjacent = {p for p, w in wildlife.adjacent(next) if w == Wildlife.SALMON}
            if len(adjacent) > 2:
                valid_run = False
            current_run.add(next)
            unexplored.update(adjacent - current_run)

        visited.update(current_run)

        if valid_run:
            run_length = len(current_run)
            total_salmon_score += salmon_scores[min(run_length, len(salmon_scores) - 1)]

    return total_salmon_score


def calculate_hawk_score(wildlife: HexGrid[Wildlife]):
    hawks = {p for p, w in wildlife.items() if w == Wildlife.HAWK}
    num_isolated_hawks = 0

    for p in hawks:
        num_adjacent_hawks = sum(w == Wildlife.HAWK for _, w in wildlife.adjacent(p))
        if num_adjacent_hawks == 0:
            num_isolated_hawks += 1

    hawk_scores = [0, 2, 5, 8, 11, 14, 18, 22, 26]

    return hawk_scores[min(num_isolated_hawks, len(hawk_scores) - 1)]


def calculate_fox_score(wildlife: HexGrid[Wildlife]):
    fox_score = 0
    foxes = {p for p, w in wildlife.items() if w == Wildlife.FOX}

    for p in foxes:
        unique_adjacent = {w for _, w in wildlife.adjacent(p)}
        fox_score += len(unique_adjacent)

    return fox_score


def calculate_wildlife_score(wildlife: HexGrid[Wildlife]):
    return {
        Wildlife.BEAR: calculate_bear_score(wildlife),
        Wildlife.ELK: calculate_elk_score(wildlife),
        Wildlife.SALMON: calculate_salmon_score(wildlife),
        Wildlife.HAWK: calculate_hawk_score(wildlife),
        Wildlife.FOX: calculate_fox_score(wildlife),
    }


def calculate_score(gs: GameState):
    ws = calculate_wildlife_score(gs.environment.wildlife)
    hs = calculate_habitat_score(gs.environment.tiles)

    return sum(ws.values()) + sum(hs.values()) + gs.nature_tokens
