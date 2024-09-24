# TODO list:
# - Iterate over every possible action
#   (and later include spending nature tokens and refreshing the display)
# - Eliminate actions that are functionally identical or are clearly suboptimal (mostly tile rotations)
# - Re-use scoring for sections of the score that do not change
#   (such as wildlife scoring for types that weren't just placed and habitat scoring for far away tiles)
# - Generate features for each potential next state (sharing where possible)
# - Return lists for actions, rewards and features (of the resulting state)

from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import HexPosition, share_edge
from cascadia_ai.game_state import GameState
from cascadia_ai.score import (
    calculate_bear_score,
    calculate_elk_score,
    calculate_fox_score,
    calculate_hawk_score,
    calculate_salmon_score,
    calculate_wildlife_score,
)


def get_transitions(state: GameState):
    wcache: dict[tuple[Wildlife, Wildlife, HexPosition], int] = {}
    wscore = calculate_wildlife_score(state.env)

    hgroups = state.env.habitat_groups()

    actions = state.available_actions()

    rewards: list[int] = []
    features_list: list[list[float]] = []

    for action in actions:
        reward = 0
        features = [float(state.turns_remaining - 1)]

        tile = state.tile_supply[action.tile_index]
        wildlife = state.wildlife_bag[action.wildlife_index]

        if action.wildlife_position is None:
            wildlife_target = None
        elif action.wildlife_position == action.tile_position:
            wildlife_target = tile
        else:
            wildlife_target = state.env.tiles[action.wildlife_position]

        nature_tokens = state.nature_tokens
        if action.tile_index != action.wildlife_index:
            nature_tokens -= 1
            reward -= 1
        if wildlife_target is not None and wildlife_target.nature_token_reward:
            nature_tokens += 1
            reward += 1
        features.append(float(nature_tokens))

        for h in Habitat:
            largest_group_size = len(hgroups[h][0])

            # TODO this could be cached too or could be calculated independently to wildlife stuff and then merged

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

                # if h == Habitat.RIVERS and :
                #     print(connected_positions)

                for group in hgroups[h]:
                    if any(cpos in group for cpos in connected_positions):
                        new_group_size += len(group)

                if new_group_size > largest_group_size:
                    reward += new_group_size - largest_group_size
                    if new_group_size >= 7 and not (largest_group_size >= 7):
                        reward += 2
                    largest_group_size = new_group_size

            features.append(float(largest_group_size))
            features.append(float(2 if largest_group_size >= 7 else 0))

        is_fox_adjacent = action.wildlife_position is not None and any(
            w == Wildlife.FOX
            for _, w in state.env.adjacent_wildlife(action.wildlife_position)
        )

        for w in Wildlife:
            wildlife_score = wscore[w]

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

                    reward += new_wildlife_score - wildlife_score
                    wildlife_score = new_wildlife_score

            features.append(float(wildlife_score))

        rewards.append(reward)
        features_list.append(features)

        # TODO list for unexplored features:
        # - habitat/wildlife/wildlife_slots remaining in the display
        # - wildlife_slots in environment
        # - counts for single bears
        # - unclaimed nature token rewards in environment
        # - number of tiles without wildlife
        # - counts for wildlife/tiles remaining in supply
        # - something to account for potential scoring that could be gained from
        #   current empty slots or anything else that captures unclaimed potential score

    return (actions, rewards, features_list)
