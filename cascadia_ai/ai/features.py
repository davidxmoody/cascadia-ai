import numpy as np
from cascadia_ai.enums import Habitat, Wildlife
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
