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

        score[habitat] = max(len(group) for group in groups.values())

        # TODO add test for bonus
        if score[habitat] >= 7:
            score[habitat] += 2

    return score


def calculate_wildlife_score(wildlife: HexGrid[Wildlife]):
    return {}


def calculate_score(gs: GameState):
    ws = calculate_wildlife_score(gs.environment.wildlife)
    hs = calculate_habitat_score(gs.environment.tiles)

    return sum(ws.values()) + sum(hs.values()) + gs.nature_tokens
