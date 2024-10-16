from collections import Counter
import numpy as np
from cascadia_ai.ai.actions import calculate_treward, calculate_wreward
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import HexPosition, share_edge
from cascadia_ai.game_state import Action, GameState
from cascadia_ai.tiles import Tile


feature_names = [
    "turns_remaining",
    "nature_tokens",
    "unclaimed_nt_rewards",
    *(f"{w.value}_num_unoccupied_slots" for w in Wildlife),
    *(f"{h.value}_largest_area" for h in Habitat),
    "num_bear_pairs",
    "num_bear_singles",
    "num_elk",
    "num_salmon",
    "num_hawks",
    "num_foxes",
    "usable_spaces",
    *(f"{w.value}_best_unoccupied_reward" for w in Wildlife),
    *(f"{w.value}_best_adjacent_reward" for w in Wildlife),
    *(f"{h.value}_best_reward" for h in Habitat),
]

F = {name: i for i, name in enumerate(feature_names)}


def calculate_usable_spaces(
    state: GameState, last_placement: tuple[HexPosition, Wildlife] | None = None
):
    num_usable = 0

    for pos, tile in state.env.unoccupied_tiles():
        if Wildlife.ELK in tile.wildlife_slots or Wildlife.FOX in tile.wildlife_slots:
            num_usable += 1

        elif (
            Wildlife.HAWK in tile.wildlife_slots
            and not state.env.has_adjacent_wildlife(pos, Wildlife.HAWK)
        ):
            num_usable += 1

        elif Wildlife.SALMON in tile.wildlife_slots and all(
            len(list(state.env.adjacent_wildlife(apos, Wildlife.SALMON))) < 2
            for apos, _ in state.env.adjacent_wildlife(pos, Wildlife.SALMON)
        ):
            num_usable += 1

        elif Wildlife.BEAR in tile.wildlife_slots and all(
            len(list(state.env.adjacent_wildlife(apos, Wildlife.BEAR))) == 0
            for apos, _ in state.env.adjacent_wildlife(pos, Wildlife.BEAR)
        ):
            num_usable += 1

    return num_usable


class StateFeatures:
    def __init__(self, state: GameState):
        data = np.zeros(len(feature_names), dtype=np.float32)

        data[F["turns_remaining"]] = state.turns_remaining
        data[F["nature_tokens"]] = state.nature_tokens

        for _, t in state.env.unoccupied_tiles():
            data[F["unclaimed_nt_rewards"]] += t.nature_token_reward
            for w in t.wildlife_slots:
                data[F[f"{w.value}_num_unoccupied_slots"]] += 1

        hgroups = state.env.habitat_groups()
        for h, groups in hgroups.items():
            data[F[f"{h.value}_largest_area"]] = len(groups[0])

        bsizes = Counter(len(g) for g in state.env.wildlife_groups(Wildlife.BEAR))
        data[F["num_bear_pairs"]] = bsizes[2]
        data[F["num_bear_singles"]] = bsizes[1]

        wcounts = Counter(state.env.wildlife.values())
        data[F["num_elk"]] = wcounts[Wildlife.ELK]
        data[F["num_salmon"]] = wcounts[Wildlife.SALMON]
        data[F["num_hawks"]] = wcounts[Wildlife.HAWK]
        data[F["num_foxes"]] = wcounts[Wildlife.FOX]

        data[F["usable_spaces"]] = calculate_usable_spaces(state)

        # best_wrewards = []
        # for pos, tile in state.env.unoccupied_tiles():
        #     best_wreward = 0
        #     for wildlife in tile.wildlife_slots:
        #         wreward = calculate_wreward(state, wildlife, pos)
        #         if wreward is not None:
        #             if tile.nature_token_reward:
        #                 wreward += 1
        #             best_wreward = max(best_wreward, wreward)
        #     best_wrewards.append(best_wreward)

        # best_wrewards.sort(reverse=True)

        # for i in range(3):
        #     data[F[f"best_unoccupied_wreward_{i}"]] = best_wrewards[i]

        surrounding_empty = {pos for pos in state.env.all_adjacent_empty()}

        for w in Wildlife:
            data[F[f"{w.value}_best_unoccupied_reward"]] = max(
                (
                    (calculate_wreward(state, w, pos) or -1)
                    + int(tile.nature_token_reward)
                    for pos, tile in state.env.unoccupied_tiles()
                    if w in tile.wildlife_slots
                ),
                default=0,
            )

        for w in Wildlife:
            data[F[f"{w.value}_best_adjacent_reward"]] = max(
                (calculate_wreward(state, w, pos) or 0) for pos in surrounding_empty
            )

        for h in Habitat:
            data[F[f"{h.value}_best_reward"]] = max(
                calculate_treward(state, hgroups, Tile((h, h), frozenset()), pos, 0)
                for pos in surrounding_empty
            )

        self._data = data
        self._state = state
        self._hgroups = hgroups

    def get_next_features(self, actions: list[Action]):
        hcache: dict[tuple[int, HexPosition, int], dict[int, int]] = {}

        features_array = np.tile(self._data, (len(actions), 1))

        for i, action in enumerate(actions):
            features_array[i, F["turns_remaining"]] -= 1
            placed_tile = self._state.tile_display[action.tile_index]

            if action.wildlife_position is None:
                wildlife_target = None
            elif action.wildlife_position == action.tile_position:
                wildlife_target = placed_tile
            else:
                wildlife_target = self._state.env.tiles[action.wildlife_position]

            if action.nt_spent:
                features_array[i, F["nature_tokens"]] -= 1

            if placed_tile.nature_token_reward:
                features_array[i, F["unclaimed_nt_rewards"]] += 1

            if wildlife_target is not None and wildlife_target.nature_token_reward:
                features_array[i, F["nature_tokens"]] += 1
                features_array[i, F["unclaimed_nt_rewards"]] -= 1

            hkey = (action.tile_index, action.tile_position, action.tile_rotation)
            if hkey in hcache:
                hvalues = hcache[hkey]
            else:
                hvalues: dict[int, int] = {}
                for h in set(placed_tile.habitats):
                    connected_positions = {
                        apos
                        for apos, atile in self._state.env.adjacent_tiles(
                            action.tile_position
                        )
                        if share_edge(
                            action.tile_position,
                            placed_tile.rotate(action.tile_rotation),
                            apos,
                            atile,
                            h,
                        )
                    }

                    new_group_size = 1
                    for group in self._hgroups[h]:
                        if not group.isdisjoint(connected_positions):
                            new_group_size += len(group)

                    if new_group_size > len(self._hgroups[h][0]):
                        hvalues[F[f"{h.value}_largest_area"]] = new_group_size

                hcache[hkey] = hvalues

            for k, v in hvalues.items():
                features_array[i, k] = v

            if action.tile_position != action.wildlife_position:
                for w in placed_tile.wildlife_slots:
                    features_array[i, F[f"{w.value}_num_unoccupied_slots"]] += 1
                if wildlife_target is not None:
                    for w in wildlife_target.wildlife_slots:
                        features_array[i, F[f"{w.value}_num_unoccupied_slots"]] -= 1

            if action.wildlife_position is not None:
                placed_wildlife = self._state.wildlife_display[action.wildlife_index]

                match placed_wildlife:
                    case Wildlife.BEAR:
                        # Assume that actions that would invalidate existing groups aren't passed in
                        if self._state.env.has_adjacent_wildlife(
                            action.wildlife_position, Wildlife.BEAR
                        ):
                            features_array[i, F["num_bear_pairs"]] += 1
                            features_array[i, F["num_bear_singles"]] -= 1
                        else:
                            features_array[i, F["num_bear_singles"]] += 1

                    case Wildlife.ELK:
                        features_array[i, F["num_elk"]] += 1

                    case Wildlife.SALMON:
                        features_array[i, F["num_salmon"]] += 1

                    case Wildlife.HAWK:
                        features_array[i, F["num_hawks"]] += 1

                    case Wildlife.FOX:
                        features_array[i, F["num_foxes"]] += 1

        return features_array

    def __repr__(self):
        return "\n".join(
            [
                "StateFeatures({",
                *(f"  '{fn}': {self._data[F[fn]]}," for fn in feature_names),
                "})",
            ]
        )
