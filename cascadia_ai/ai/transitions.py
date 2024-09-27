# TODO list:
# - Iterate over every possible action
#   (and later include spending nature tokens and refreshing the display)
# - Eliminate actions that are functionally identical or are clearly suboptimal (mostly tile rotations)
# - Re-use scoring for sections of the score that do not change
#   (such as wildlife scoring for types that weren't just placed and habitat scoring for far away tiles)
# - Generate features for each potential next state (sharing where possible)
# - Return lists for actions, rewards and features (of the resulting state)

from typing import Counter
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import HexPosition, adjacent_positions, share_edge
from cascadia_ai.game_state import GameState
from cascadia_ai.score import (
    calculate_bear_score,
    calculate_elk_score,
    calculate_fox_score,
    calculate_hawk_score,
    calculate_salmon_score,
    calculate_wildlife_score,
)
import numpy as np

feature_names = [
    "turns_remaining",
    "nature_tokens",
    *(f"{h.value}_{suffix}" for suffix in ["largest", "bonus"] for h in Habitat),
    *(
        f"{w.value}_{suffix}"
        for suffix in ["score", "count", "slots"]
        for w in Wildlife
    ),
    "bear_pairs",
    "bear_singles",
    "hawk_singles",
]

F = {name: i for i, name in enumerate(feature_names)}


def get_transitions(state: GameState):
    base_features = np.zeros(len(feature_names), dtype=np.float32)
    base_features[F["turns_remaining"]] = state.turns_remaining
    base_features[F["nature_tokens"]] = state.nature_tokens

    wcache: dict[tuple[Wildlife, Wildlife, HexPosition], int] = {}
    wscore = calculate_wildlife_score(state.env)

    for w in Wildlife:
        base_features[F[f"{w.value}_score"]] = wscore[w]

    hgroups = state.env.habitat_groups()
    for h in Habitat:
        largest = len(hgroups[h][0])
        base_features[F[f"{h.value}_largest"]] = largest
        if largest >= 7:
            base_features[F[f"{h.value}_bonus"]] = 2

    wcounts = Counter(state.env.wildlife.values())
    for w, c in wcounts.items():
        base_features[F[f"{w.value}_count"]] = c

    for _, tile in state.env.unoccupied_tiles():
        for w in tile.wildlife_slots:
            base_features[F[f"{w.value}_slots"]] += 1

    bear_groups = state.env.wildlife_groups(Wildlife.BEAR)
    bear_group_sizes = Counter(len(g) for g in bear_groups)
    base_features[F["bear_pairs"]] = bear_group_sizes[2]
    base_features[F["bear_singles"]] = bear_group_sizes[1]

    hawk_groups = state.env.wildlife_groups(Wildlife.HAWK)
    base_features[F["hawk_singles"]] = sum(len(g) == 1 for g in hawk_groups)

    actions = state.available_actions()

    rewards: list[int] = []
    features_array = np.tile(base_features, (len(actions), 1))

    for i, action in enumerate(actions):
        reward = 0
        features_array[i, F["turns_remaining"]] -= 1

        tile = state.tile_supply[action.tile_index]
        wildlife = state.wildlife_bag[action.wildlife_index]

        if action.wildlife_position is None:
            wildlife_target = None
        elif action.wildlife_position == action.tile_position:
            wildlife_target = tile
        else:
            wildlife_target = state.env.tiles[action.wildlife_position]

        if action.tile_index != action.wildlife_index:
            reward -= 1
            features_array[i, F["nature_tokens"]] -= 1
        if wildlife_target is not None and wildlife_target.nature_token_reward:
            reward += 1
            features_array[i, F["nature_tokens"]] += 1

        for h in Habitat:
            largest_group_size = len(hgroups[h][0])

            if h in tile.habitats:
                new_group_size = 1

                connected_positions = [
                    apos
                    for apos, atile in state.env.adjacent_tiles(action.tile_position)
                    if share_edge(
                        action.tile_position,
                        tile.rotate(action.tile_rotation),
                        apos,
                        atile,
                        h,
                    )
                ]

                for group in hgroups[h]:
                    if any(cpos in group for cpos in connected_positions):
                        new_group_size += len(group)

                if new_group_size > largest_group_size:
                    reward += new_group_size - largest_group_size
                    features_array[i, F[f"{h.value}_largest"]] = new_group_size

                    if new_group_size >= 7 and not (largest_group_size >= 7):
                        reward += 2
                        features_array[i, F[f"{h.value}_bonus"]] = (
                            2 if new_group_size >= 7 else 0
                        )

        features_array[i, F[f"{wildlife.value}_count"]] += 1

        if (
            wildlife_target is not None
            and action.tile_position != action.wildlife_position
        ):
            for w in wildlife_target.wildlife_slots:
                features_array[i, F[f"{w.value}_slots"]] -= 1
            for w in tile.wildlife_slots:
                features_array[i, F[f"{w.value}_slots"]] += 1

        if action.wildlife_position is not None and wildlife == Wildlife.BEAR:
            adjacent_pos = set(adjacent_positions(action.wildlife_position))
            adjacent_groups = [g for g in bear_groups if not g.isdisjoint(adjacent_pos)]

            if len(adjacent_groups) == 0:
                features_array[i, F["bear_singles"]] += 1

            elif len(adjacent_groups) == 1 and len(adjacent_groups[0]) == 1:
                features_array[i, F["bear_singles"]] -= 1
                features_array[i, F["bear_pairs"]] += 1

            else:
                for g in adjacent_groups:
                    if len(g) == 1:
                        features_array[i, F["bear_singles"]] -= 1
                    elif len(g) == 2:
                        features_array[i, F["bear_pairs"]] -= 1

        if action.wildlife_position is not None and wildlife == Wildlife.HAWK:
            adjacent_pos = set(adjacent_positions(action.wildlife_position))
            adjacent_groups = [g for g in hawk_groups if not g.isdisjoint(adjacent_pos)]

            if len(adjacent_groups) == 0:
                features_array[i, F["hawk_singles"]] += 1

            else:
                for g in adjacent_groups:
                    if len(g) == 1:
                        features_array[i, F["hawk_singles"]] -= 1

        is_fox_adjacent = action.wildlife_position is not None and any(
            w == Wildlife.FOX
            for _, w in state.env.adjacent_wildlife(action.wildlife_position)
        )

        for w in Wildlife:
            if action.wildlife_position is not None:
                if w == wildlife or (w == Wildlife.FOX and is_fox_adjacent):
                    key = (w, wildlife, action.wildlife_position)

                    if key in wcache:
                        new_wildlife_score = wcache[key]

                    else:
                        # TODO this could be optimised by not needing to copy everything
                        # or by handling each wildlife case individually with separate logic

                        env_copy = state.env.copy()
                        env_copy.place_tile(action.tile_position, tile)
                        env_copy.place_wildlife(action.wildlife_position, wildlife)

                        match w:
                            case Wildlife.BEAR:
                                new_wildlife_score = calculate_bear_score(env_copy)
                            case Wildlife.ELK:
                                new_wildlife_score = calculate_elk_score(env_copy)
                            case Wildlife.SALMON:
                                new_wildlife_score = calculate_salmon_score(env_copy)
                            case Wildlife.HAWK:
                                new_wildlife_score = calculate_hawk_score(env_copy)
                            case Wildlife.FOX:
                                new_wildlife_score = calculate_fox_score(env_copy)

                        wcache[key] = new_wildlife_score

                    reward += new_wildlife_score - wscore[w]
                    features_array[i, F[f"{w.value}_score"]] = new_wildlife_score

        rewards.append(reward)

        # TODO list for unexplored features:
        # - habitat/wildlife/wildlife_slots remaining in the display
        # - wildlife_slots in environment
        # - counts for single bears
        # - unclaimed nature token rewards in environment
        # - number of tiles without wildlife
        # - counts for wildlife/tiles remaining in supply
        # - something to account for potential scoring that could be gained from
        #   current empty slots or anything else that captures unclaimed potential score

    return (actions, rewards, features_array)
