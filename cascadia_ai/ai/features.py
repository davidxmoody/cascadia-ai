from copy import deepcopy
import numpy as np
from collections import Counter, defaultdict
from cascadia_ai.ai.actions import calculate_wreward
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import adjacent_positions
from cascadia_ai.game_state import Action, GameState

Cache = defaultdict[str, dict]


features_shapes = [[13], [37, 11], [34, 5]]


def pad_list(data: list[list], length: int):
    while len(data) < length:
        data.append([0] * len(data[0]))
    return data


def get_features(s: GameState, a: Action | None = None, cache: Cache | None = None):
    if a is None:
        turns_remaining = s.turns_remaining
        nature_tokens = s.nature_tokens
        bsizes = Counter(len(g) for g in s.env.wildlife_groups(Wildlife.BEAR))
        wcounter = Counter(s.env.wildlife.values())
        areas = s.env.areas
        unoccupied_tiles = set(s.env.unoccupied_tiles())
        adjacent_empty = s.env.all_adjacent_empty()

    else:
        placed_tile = s.tile_display[a.tile_index].rotate(a.tile_rotation)
        if a.wildlife_position is None:
            placed_wildlife = None
            wildlife_target = None
        else:
            placed_wildlife = s.wildlife_display[a.wildlife_index]
            wildlife_target = (
                placed_tile
                if a.wildlife_position == a.tile_position
                else s.env.tiles[a.wildlife_position]
            )

        turns_remaining = s.turns_remaining - 1

        nature_tokens = s.nature_tokens
        if a.nt_spent:
            nature_tokens -= 1
        if wildlife_target is not None and wildlife_target.nature_token_reward:
            nature_tokens += 1

        # TODO
        bsizes = Counter(len(g) for g in s.env.wildlife_groups(Wildlife.BEAR))

        wcounter = Counter(s.env.wildlife.values())
        if placed_wildlife is not None:
            wcounter[placed_wildlife] += 1

        areas = {
            h: area.place_tile(a.tile_position, placed_tile)
            for h, area in s.env.areas.items()
        }

        unoccupied_tiles = set(s.env.unoccupied_tiles())
        if a.tile_position != a.wildlife_position:
            unoccupied_tiles.add((a.tile_position, placed_tile))
            if a.wildlife_position is not None:
                unoccupied_tiles.remove(
                    (a.wildlife_position, s.env.tiles[a.wildlife_position])
                )

        adjacent_empty = s.env.all_adjacent_empty()
        adjacent_empty.remove(a.tile_position)
        adjacent_empty.update(
            apos
            for apos in adjacent_positions(a.tile_position)
            if apos not in s.env.tiles
        )

    global_features = [
        turns_remaining,
        nature_tokens,
        bsizes[2],
        bsizes[1],
        wcounter[Wildlife.ELK],
        wcounter[Wildlife.SALMON],
        wcounter[Wildlife.HAWK],
        wcounter[Wildlife.FOX],
        *(areas[h].largest_area for h in Habitat),
    ]

    wildlife_rewards = []
    tile_rewards = []

    for pos, tile in unoccupied_tiles:
        wildlife_rewards.append(
            [
                int(tile.nature_token_reward),
                *(int(w in tile.wildlife_slots) for w in Wildlife),
                *(
                    (
                        0
                        if w not in tile.wildlife_slots
                        # TODO
                        else (calculate_wreward(s, w, pos) or 0)
                    )
                    for w in Wildlife
                ),
            ]
        )

    for pos in adjacent_empty:
        wildlife_rewards.append(
            [
                0,
                *(0 for _ in Wildlife),
                # TODO
                *((calculate_wreward(s, w, pos) or 0) for w in Wildlife),
            ]
        )

        tile_rewards.append([areas[h].get_best_reward(pos) for h in Habitat])

    return (
        np.array(global_features, dtype=np.float32),
        np.array(pad_list(wildlife_rewards, 37), dtype=np.float32),
        np.array(pad_list(tile_rewards, 34), dtype=np.float32),
    )


def get_next_features(s: GameState, actions: list[Action], cache: Cache | None = None):
    features_list = [get_features(s, a, cache) for a in actions]
    return [
        np.stack([f[i] for f in features_list]) for i in range(len(features_list[0]))
    ]
