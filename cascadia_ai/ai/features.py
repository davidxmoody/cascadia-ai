import numpy as np
from cascadia_ai.ai.actions import get_actions
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import HexPosition, share_edge
from cascadia_ai.game_state import Action, GameState


feature_names = [
    "turns_remaining",
    "nature_tokens",
    "num_unoccupied",
    "unclaimed_nt_rewards",
    *(
        f"{h.value}_{suffix}"
        for h in Habitat
        for suffix in [
            "group_size_0",
            "group_size_1",
            "group_size_2",
            "scoring_bonus",
            "in_display_0",
            "in_display_1",
            "num_in_remaining",
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
            "in_display_0",
            "in_display_1",
            "slots_in_display_0",
            "slots_in_display_1",
            "num_in_remaining",
            "num_slots_in_remaining",
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

        self["turns_remaining"] = state.turns_remaining
        self["nature_tokens"] = state.nature_tokens
        self["num_unoccupied"] = len(self._unoccupied)
        self["unclaimed_nt_rewards"] = sum(
            t.nature_token_reward for _, t in self._unoccupied
        )

        for h in Habitat:
            group_sizes = [len(g) for g in self._hgroups[h]]

            for i in range(3):
                self[f"{h.value}_group_size_{i}"] = (
                    group_sizes[i] if i < len(group_sizes) else 0
                )

            self[f"{h.value}_scoring_bonus"] = (
                2 if len(group_sizes) and group_sizes[0] >= 7 else 0
            )

            for i in range(2):
                self[f"{h.value}_in_display_{i}"] = sum(
                    h == h2 for h2 in state.tile_display[i].habitats
                )

            self[f"{h.value}_num_in_remaining"] = sum(
                h == h2
                for tile in (state._tile_supply + state.tile_display[2:])
                for h2 in tile.habitats
            )

        for w in Wildlife:
            group_sizes = [len(g) for g in self._wgroups[w]]

            for i in range(8):
                self[f"{w.value}_group_size_{i}"] = (
                    group_sizes[i] if i < len(group_sizes) else 0
                )

            self[f"{w.value}_num_unoccupied_slots"] = sum(
                w in t.wildlife_slots for _, t in self._unoccupied
            )

            for i in range(2):
                self[f"{w.value}_in_display_{i}"] = w == state.wildlife_display[i]
                self[f"{w.value}_slots_in_display_{i}"] = (
                    w in state.tile_display[i].wildlife_slots
                )

            self[f"{w.value}_num_in_remaining"] = state._wildlife_supply[w] + sum(
                w == w2 for w2 in state.wildlife_display[2:]
            )

            self[f"{w.value}_num_slots_in_remaining"] = sum(
                w in tile.wildlife_slots
                for tile in (state._tile_supply + state.tile_display[2:])
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
                if i == 0:
                    print("nt spent")
                    print(action)
                features_array[i, F["nature_tokens"]] -= 1

            if placed_tile.nature_token_reward:
                features_array[i, F["unclaimed_nt_rewards"]] += 1

            if wildlife_target is not None and wildlife_target.nature_token_reward:
                features_array[i, F["nature_tokens"]] += 1
                features_array[i, F["unclaimed_nt_rewards"]] -= 1

            if wildlife_target is None:
                features_array[i, F["num_unoccupied"]] += 1

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

                    group_sizes = [1]
                    for group in self._hgroups[h]:
                        if group.isdisjoint(connected_positions):
                            group_sizes.append(len(group))
                        else:
                            group_sizes[0] += len(group)
                    group_sizes.sort(reverse=True)

                    for j in range(3):
                        hvalues[F[f"{h.value}_group_size_{j}"]] = (
                            group_sizes[j] if j < len(group_sizes) else 0
                        )

                    hvalues[F[f"{h.value}_scoring_bonus"]] = (
                        2 if len(group_sizes) and group_sizes[0] >= 7 else 0
                    )

                hcache[hkey] = hvalues

            for k, v in hvalues.items():
                features_array[i, k] = v

            for tile in self._state.tile_display[2:]:
                for h in tile.habitats:
                    features_array[i, F[f"{h.value}_num_in_remaining"]] -= 1
                for w in tile.wildlife_slots:
                    features_array[i, F[f"{w.value}_num_slots_in_remaining"]] -= 1

            new_tile_display = self._state.tile_display[:]
            new_tile_display.pop(action.tile_index),
            new_tile_display.pop(0),
            for j, tile in enumerate(new_tile_display):
                for h in Habitat:
                    features_array[i, F[f"{h.value}_in_display_{j}"]] = sum(
                        h == h2 for h2 in new_tile_display[j].habitats
                    )
                for w in Wildlife:
                    features_array[i, F[f"{w.value}_slots_in_display_{j}"]] = (
                        w in tile.wildlife_slots
                    )

            for w in self._state.wildlife_display[2:]:
                features_array[i, F[f"{w.value}_num_in_remaining"]] -= 1

            new_wildlife_display = self._state.wildlife_display[:]
            new_wildlife_display.pop(action.wildlife_index)
            new_wildlife_display.pop(0)
            for j in range(2):
                for w in Wildlife:
                    features_array[i, F[f"{w.value}_in_display_{j}"]] = (
                        w == new_wildlife_display[j]
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

                if i == 3:
                    print(connected_positions)

                group_sizes = [1]
                for group in self._wgroups[placed_wildlife]:
                    if group.isdisjoint(connected_positions):
                        group_sizes.append(len(group))
                    else:
                        group_sizes[0] += len(group)
                group_sizes.sort(reverse=True)

                for j in range(8):
                    features_array[i, F[f"{placed_wildlife.value}_group_size_{j}"]] = (
                        group_sizes[j] if j < len(group_sizes) else 0
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


# %%
s = GameState()

# %%
actions = [a for a, _ in get_actions(s)]

sf = StateFeatures(s)

next_states = [s.copy().take_action(a) for a in actions]

next_features = sf.get_next_features(actions)

# %%
for i, ns in enumerate(next_states):
    nsf = StateFeatures(ns)
    nsfd = nsf._data
    nsfd2 = next_features[i]

    if not np.array_equal(nsfd, nsfd2):
        print(i)
        print(actions[i])
        for fi, fn in enumerate(feature_names):
            print(f"{fn:<24}: {nsfd[fi]:>4} {nsfd2[fi]:>4} {nsfd[fi] == nsfd2[fi]}")
        break
