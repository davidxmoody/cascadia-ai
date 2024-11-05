from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import Environment, share_edge
from cascadia_ai.game_state import Action, GameState
from cascadia_ai.positions import HexPosition, adjacent_positions
from cascadia_ai.tiles import Tile
from cascadia_ai.score import (
    scoring_bear_pairs,
    scoring_salmon_run,
    scoring_hawk_singles,
    get_max_elk_group_score,
    is_valid_salmon_run,
)


def find_connected_groups(groups: list[set[HexPosition]], pos: HexPosition):
    adjacent = set(adjacent_positions(pos))
    return [g for g in groups if not g.isdisjoint(adjacent)]


def pick_best_rotations(
    state: GameState,
    tile: Tile,
    pos: HexPosition,
    hgroups: dict[Habitat, list[set[HexPosition]]],
):
    trotations = list(range(1 if tile.single_habitat else 6))
    trewards = [
        calculate_treward(state, hgroups, tile, pos, trot) for trot in trotations
    ]
    max_treward = max(trewards)
    return [
        (trot, treward)
        for trot, treward in zip(trotations, trewards)
        if treward == max_treward
    ]


def calculate_treward(
    state: GameState,
    hgroups: dict[Habitat, list[set[HexPosition]]],
    tile: Tile,
    pos: HexPosition,
    rotation: int,
):
    treward = 0

    for h in set(tile.habitats):
        largest_group_size = len(hgroups[h][0])

        new_group_size = 1

        connected_positions = {
            apos
            for apos, atile in state.env.adjacent_tiles(pos)
            if share_edge(pos, tile.rotate(rotation), apos, atile, h)
        }

        connected_groups = [
            g for g in hgroups[h] if not connected_positions.isdisjoint(g)
        ]

        new_group_size = 1 + sum(len(g) for g in connected_groups)

        if new_group_size > largest_group_size:
            treward += new_group_size - largest_group_size
            if new_group_size >= 7 and not (largest_group_size >= 7):
                treward += 2

    return treward


def tile_options(state: GameState):
    hgroups = state.env.habitat_groups()

    for tpos in state.env.all_adjacent_empty():
        for tindex in range(4):
            tile = state.tile_display[tindex]
            for trot, treward in pick_best_rotations(state, tile, tpos, hgroups):
                yield (tindex, tpos, trot, treward)


def windex_options(state: GameState, tindex: int):
    yield (tindex, 0)

    if state.nature_tokens > 0:
        considered_wildlife = {state.wildlife_display[tindex]}

        for i in range(4):
            if state.wildlife_display[i] not in considered_wildlife:
                considered_wildlife.add(state.wildlife_display[i])
                yield (i, -1)


def wpos_options(state: GameState, tindex: int, tpos: HexPosition, windex: int):
    wildlife = state.wildlife_display[windex]
    placed_tile = state.tile_display[tindex]

    if wildlife in placed_tile.wildlife_slots:
        yield (tpos, int(placed_tile.nature_token_reward))

    for pos, tile in state.env.unoccupied_tiles():
        if wildlife in tile.wildlife_slots:
            yield (pos, int(tile.nature_token_reward))


def calculate_bear_reward(groups: list[set[HexPosition]], pos: HexPosition):
    connected_groups = find_connected_groups(groups, pos)

    if len(connected_groups) == 0:
        return 0

    elif len(connected_groups) == 1 and len(connected_groups[0]) == 1:
        num_bear_pairs_before = sum(len(g) == 2 for g in groups)

        return (
            scoring_bear_pairs[num_bear_pairs_before + 1]
            - scoring_bear_pairs[num_bear_pairs_before]
        )

    return None


def calculate_elk_reward(groups: list[set[HexPosition]], pos: HexPosition):
    connected_groups = find_connected_groups(groups, pos)

    partial_score_before = sum(
        get_max_elk_group_score(group) for group in connected_groups
    )

    partial_score_after = get_max_elk_group_score(set.union({pos}, *connected_groups))

    return partial_score_after - partial_score_before


def calculate_salmon_reward(groups: list[set[HexPosition]], pos: HexPosition):
    connected_groups = find_connected_groups(groups, pos)

    new_group = set.union({pos}, *connected_groups)

    if not is_valid_salmon_run(new_group):
        return None

    partial_score_before = sum(
        scoring_salmon_run[len(group)] for group in connected_groups
    )

    partial_score_after = scoring_salmon_run[len(new_group)]

    return partial_score_after - partial_score_before


def calculate_hawk_reward(groups: list[set[HexPosition]], pos: HexPosition):
    connected_groups = find_connected_groups(groups, pos)

    if len(connected_groups):
        return None

    num_singles_before = sum(len(g) == 1 for g in groups)

    return (
        scoring_hawk_singles[num_singles_before + 1]
        - scoring_hawk_singles[num_singles_before]
    )


def calculate_fox_reward(env: Environment, pos: HexPosition):
    return len(set(w for _, w in env.adjacent_wildlife(pos)))


def calculate_adjacent_fox_reward(
    env: Environment, pos: HexPosition, wildlife: Wildlife
):
    adjacent_fox_reward = 0
    for fox_pos, _ in env.adjacent_wildlife(pos, Wildlife.FOX):
        if not env.has_adjacent_wildlife(fox_pos, wildlife):
            adjacent_fox_reward += 1
    return adjacent_fox_reward


# TODO change argument order to match other functions
def calculate_wreward(state: GameState, wildlife: Wildlife, pos: HexPosition):
    match wildlife:
        case Wildlife.BEAR:
            groups = state.env.wildlife_groups(Wildlife.BEAR)
            reward = calculate_bear_reward(groups, pos)

        case Wildlife.ELK:
            groups = state.env.wildlife_groups(Wildlife.ELK)
            reward = calculate_elk_reward(groups, pos)

        case Wildlife.SALMON:
            groups = state.env.wildlife_groups(Wildlife.SALMON)
            reward = calculate_salmon_reward(groups, pos)

        case Wildlife.HAWK:
            groups = state.env.wildlife_groups(Wildlife.HAWK)
            reward = calculate_hawk_reward(groups, pos)

        case Wildlife.FOX:
            reward = calculate_fox_reward(state.env, pos)

    if reward is None:
        return None

    return reward + calculate_adjacent_fox_reward(state.env, pos, wildlife)


def get_actions_and_rewards(state: GameState) -> tuple[list[Action], list[int]]:
    actions: list[Action] = []
    rewards: list[int] = []

    wcache: dict[tuple[Wildlife, HexPosition], int | None] = {}

    for tindex, tpos, trot, treward in tile_options(state):
        wplaced = False

        for windex, ntcost in windex_options(state, tindex):
            for wpos, ntreward in wpos_options(state, tindex, tpos, windex):
                wildlife = state.wildlife_display[windex]

                key = (wildlife, wpos)
                if key in wcache:
                    wreward = wcache[key]
                else:
                    wreward = calculate_wreward(state, wildlife, wpos)
                    wcache[key] = wreward

                if wreward is not None:
                    wplaced = True
                    actions.append(Action(tindex, tpos, trot, windex, wpos))
                    rewards.append(treward + ntreward + ntcost + wreward)

        if not wplaced:
            actions.append(Action(tindex, tpos, trot, tindex, None))
            rewards.append(treward)

    return (actions, rewards)
