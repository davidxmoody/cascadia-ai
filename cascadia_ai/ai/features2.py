from enum import Enum
from typing import Iterable
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import (
    Environment,
    HexPosition,
    adjacent_positions,
    share_edge,
)
from cascadia_ai.game_state import GameState, Action


def encode(enum: type[Enum], *values: Enum):
    return [float(values.count(v)) for v in enum]


def format_nums(nums: Iterable[int], length: int = 3):
    sorted_nums = sorted(nums, reverse=True)[:length]
    floats = [float(num) for num in sorted_nums]
    while len(floats) < length:
        floats.append(0.0)
    return floats


def habitat_group_sizes(env: Environment):
    for _, groups in env.habitat_groups().items():
        group_sizes = [len(g) for g in groups]
        yield from format_nums(group_sizes)
        yield 2.0 if max(group_sizes, default=0) >= 7 else 0.0


def wildlife_group_counts(env: Environment, wildlife: Wildlife):
    groups = env.wildlife_groups(wildlife)
    group_sizes = [len(g) for g in groups]
    ones = group_sizes.count(1)
    twos = group_sizes.count(2)
    others = len(group_sizes) - ones - twos
    return [float(ones), float(twos), float(others)]


def wildlife_group_sizes(env: Environment, wildlife: Wildlife):
    groups = env.wildlife_groups(wildlife)
    group_sizes = [len(g) for g in groups]
    return format_nums(group_sizes)


def fox_counts(env: Environment):
    num_foxes = 0.0
    num_adjacent = 0.0
    num_unoccupied = 0.0
    num_empty = 0.0

    for pos, w in env.wildlife.items():
        if w == Wildlife.FOX:
            num_foxes += 1.0
            for apos in adjacent_positions(pos):
                if apos in env.tiles:
                    if apos in env.wildlife:
                        num_adjacent += 1.0
                    else:
                        num_unoccupied += 1.0
                else:
                    num_empty += 1.0

    return [num_foxes, num_adjacent, num_unoccupied, num_empty]


def unoccupied_info(env: Environment):
    wslots = (w for _, t in env.unoccupied_tiles() for w in t.wildlife_slots)
    return encode(Wildlife, *wslots)


def remaining_counts(state: GameState):
    habitats = (h for t in state._tile_supply for h in t.habitats)
    wslots = (w for t in state._tile_supply for w in t.wildlife_slots)

    yield from encode(Habitat, *habitats)
    yield from encode(Wildlife, *wslots)
    yield from encode(Wildlife, *state._wildlife_supply)


def get_state_features(state: GameState) -> list[float]:
    return [
        float(state.turns_remaining),
        float(state.nature_tokens),
        float(state.env.num_tiles_placed - state.env.num_wildlife_placed),
        *habitat_group_sizes(state.env),
        *wildlife_group_counts(state.env, Wildlife.BEAR),
        *wildlife_group_sizes(state.env, Wildlife.ELK),
        *wildlife_group_sizes(state.env, Wildlife.SALMON),
        *wildlife_group_counts(state.env, Wildlife.HAWK),
        *fox_counts(state.env),
        *unoccupied_info(state.env),
        *remaining_counts(state),
    ]


def matching_edges(state: GameState, action: Action):
    matching = 0
    tile = state.tile_display[action.tile_index]

    for apos, atile in state.env.adjacent_tiles(action.tile_position):
        if share_edge(action.tile_position, tile, apos, atile):
            matching += 1

    return float(matching)


def adjacent_wildlife(state: GameState, pos: HexPosition | None):
    if pos is None:
        return encode(Wildlife) + encode(Wildlife)

    return encode(Wildlife, *(w for _, w in state.env.adjacent_wildlife(pos))) + encode(
        Wildlife,
        *(
            w
            for _, t in state.env.adjacent_unoccupied_tiles(pos)
            for w in t.wildlife_slots
        )
    )


def get_action_features(state: GameState, action: Action) -> list[float]:
    tile = state.tile_display[action.tile_index]
    wildlife = state.wildlife_display[action.wildlife_index]

    return [
        float(matching_edges(state, action)),
        float(action.tile_index == action.wildlife_index),
        float(action.tile_position == action.wildlife_position),
        float(action.wildlife_position is not None),
        *encode(Wildlife, wildlife),
        *encode(Wildlife, *tile.wildlife_slots),
        *encode(Habitat, *tile.habitats),
        *adjacent_wildlife(state, action.tile_position),
        *adjacent_wildlife(state, action.wildlife_position),
    ]


def get_transitions(state: GameState):
    state_features = get_state_features(state)

    actions: list[Action] = []
    action_features: list[list[float]] = []

    for action in state.available_actions():
        actions.append(action)
        action_features.append(get_action_features(state, action))

    return (state_features, action_features, actions)
