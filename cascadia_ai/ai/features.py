import numpy as np
from collections import Counter
from cascadia_ai.ai.actions import calculate_treward, calculate_wreward
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.game_state import Action, GameState
from cascadia_ai.tiles import Tile


features_shapes = [[13], [37, 11], [34, 5]]


def pad_list(data: list[list], length: int):
    while len(data) < length:
        data.append([0] * len(data[0]))
    return data


def get_features(s: GameState, a: Action | None = None):
    if a is not None:
        s = s.copy().take_action(a)

    hgroups = s.env.habitat_groups()
    dummy_tiles = {h: Tile((h, h), frozenset()) for h in Habitat}
    bsizes = Counter(len(g) for g in s.env.wildlife_groups(Wildlife.BEAR))
    wcounter = Counter(s.env.wildlife.values())

    global_features = [
        s.turns_remaining,
        s.nature_tokens,
        bsizes[2],
        bsizes[1],
        wcounter[Wildlife.ELK],
        wcounter[Wildlife.SALMON],
        wcounter[Wildlife.HAWK],
        wcounter[Wildlife.FOX],
        *(len(groups[0]) for groups in hgroups.values()),
    ]

    wildlife_rewards = []
    tile_rewards = []

    for pos, tile in s.env.unoccupied_tiles():
        wildlife_rewards.append(
            [
                int(tile.nature_token_reward),
                *(int(w in tile.wildlife_slots) for w in Wildlife),
                *(
                    (
                        0
                        if w not in tile.wildlife_slots
                        else (calculate_wreward(s, w, pos) or 0)
                    )
                    for w in Wildlife
                ),
            ]
        )

    for pos in s.env.all_adjacent_empty():
        wildlife_rewards.append(
            [
                0,
                *(0 for _ in Wildlife),
                *((calculate_wreward(s, w, pos) or 0) for w in Wildlife),
            ]
        )

        tile_rewards.append(
            [
                calculate_treward(s, hgroups, tile, pos, 0)
                for tile in dummy_tiles.values()
            ]
        )

    return (
        np.array(global_features, dtype=np.float32),
        np.array(pad_list(wildlife_rewards, 37), dtype=np.float32),
        np.array(pad_list(tile_rewards, 34), dtype=np.float32),
    )
