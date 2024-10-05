from collections import Counter
import numpy as np
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import HexPosition, share_edge
from cascadia_ai.game_state import Action, GameState


feature_names = [
    "turns_remaining",
    "nature_tokens",
    "unclaimed_nt_rewards",
    *(
        f"{h.value}_{suffix}"
        for h in Habitat
        for suffix in [
            "largest_area",
            "remaining_fraction",
        ]
    ),
    *(
        f"{w.value}_{suffix}"
        for w in Wildlife
        for suffix in [
            "group_size_0",
            "group_size_1",
            "group_size_2",
            "group_size_3",
            "group_size_4",
            "group_size_5",
            "group_size_6",
            "group_size_7",
            "num_unoccupied_slots",
            "remaining_fraction",
        ]
    ),
]

F = {name: i for i, name in enumerate(feature_names)}

# TODO add something to indicate if a single bear can no longer be reached
# (i.e. if it's surrounded by other things)

# TODO add something to indicate fox score potential
# (e.g. for each fox, how much could its score be increased)

# TODO consider encoding group sizes differently for salmon/elk vs bears/hawks
# maybe encode number of groups of each size vs ordered group sizes

# TODO include something about adjacency of empty slots with other things


class StateFeatures:
    def __init__(self, state: GameState):
        self._state = state
        self._data = np.zeros(len(feature_names), dtype=np.float32)

        self._hgroups = state.env.habitat_groups()
        self._wgroups = {w: state.env.wildlife_groups(w) for w in Wildlife}
        self._unoccupied = list(state.env.unoccupied_tiles())

        self._remaining_wcounts = Counter(state._wildlife_supply)
        self._remaining_wcounts.update(state.wildlife_display)

        self._remaining_hcounts = Counter(
            h for t in (state._tile_supply + state.tile_display) for h in t.habitats
        )

        self["turns_remaining"] = state.turns_remaining
        self["nature_tokens"] = state.nature_tokens
        self["unclaimed_nt_rewards"] = sum(
            t.nature_token_reward for _, t in self._unoccupied
        )

        for h in Habitat:
            self[f"{h.value}_largest_area"] = len(self._hgroups[h][0])

            self[f"{h.value}_remaining_fraction"] = (
                self._remaining_hcounts[h] / self._remaining_hcounts.total()
            )

        for w in Wildlife:
            group_sizes = [len(g) for g in self._wgroups[w]]

            for i in range(1, 8):
                self[f"{w.value}_group_size_{i}"] = sum(gs == i for gs in group_sizes)

            self[f"{w.value}_num_unoccupied_slots"] = sum(
                w in t.wildlife_slots for _, t in self._unoccupied
            )

            self[f"{w.value}_remaining_fraction"] = (
                self._remaining_wcounts[w] / self._remaining_wcounts.total()
            )

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

            remaining_hcounts = self._remaining_hcounts.copy()
            for ti in [action.tile_index, 1 if action.tile_index == 0 else 0]:
                for h in self._state.tile_display[ti].habitats:
                    remaining_hcounts[h] -= 1
            for h in Habitat:
                features_array[i, F[f"{h.value}_remaining_fraction"]] = (
                    remaining_hcounts[h] / remaining_hcounts.total()
                )

            remaining_wcounts = self._remaining_wcounts.copy()
            for wi in [action.wildlife_index, 1 if action.wildlife_index == 0 else 0]:
                remaining_wcounts[self._state.wildlife_display[wi]] -= 1
            for w in Wildlife:
                features_array[i, F[f"{w.value}_remaining_fraction"]] = (
                    remaining_wcounts[w] / remaining_wcounts.total()
                )

            if action.tile_position != action.wildlife_position:
                for w in placed_tile.wildlife_slots:
                    features_array[i, F[f"{w.value}_num_unoccupied_slots"]] += 1
                if wildlife_target is not None:
                    for w in wildlife_target.wildlife_slots:
                        features_array[i, F[f"{w.value}_num_unoccupied_slots"]] -= 1

            # TODO cache this and work out a better way to cache things
            if action.wildlife_position is not None:
                placed_wildlife = self._state.wildlife_display[action.wildlife_index]

                connected_positions = {
                    pos
                    for pos, _ in self._state.env.adjacent_wildlife(
                        action.wildlife_position, placed_wildlife
                    )
                }

                group_sizes = [1]
                for group in self._wgroups[placed_wildlife]:
                    if group.isdisjoint(connected_positions):
                        group_sizes.append(len(group))
                    else:
                        group_sizes[0] += len(group)
                group_sizes.sort(reverse=True)

                for j in range(1, 8):
                    features_array[i, F[f"{placed_wildlife.value}_group_size_{j}"]] = (
                        sum(gs == j for gs in group_sizes)
                    )

        return features_array

    def __getitem__(self, feature: str) -> float:
        if feature not in feature_names:
            raise KeyError(f"Feature '{feature}' not found")
        return self._data[feature_names.index(feature)]

    def __setitem__(self, feature: str, value: float | int | bool) -> None:
        if feature not in feature_names:
            raise KeyError(f"Feature '{feature}' not found")
        self._data[feature_names.index(feature)] = value

    def __repr__(self):
        return "\n".join(
            [
                "StateFeatures({",
                *(f"  '{fn}': {self[fn]}," for fn in feature_names),
                "})",
            ]
        )


# # %%
# s = GameState()

# # %%
# actions = [a for a, _ in get_actions(s)]

# sf = StateFeatures(s)

# next_states = [s.copy().take_action(a) for a in actions]

# next_features = sf.get_next_features(actions)

# # %%
# for i, ns in enumerate(next_states):
#     nsf = StateFeatures(ns)
#     nsfd = nsf._data
#     nsfd2 = next_features[i]

#     if not np.array_equal(nsfd, nsfd2):
#         print(i)
#         print(actions[i])
#         for fi, fn in enumerate(feature_names):
#             print(f"{fn:<24}: {nsfd[fi]:>4} {nsfd2[fi]:>4} {nsfd[fi] == nsfd2[fi]}")
#         break
