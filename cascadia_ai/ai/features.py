import numpy as np
from collections import Counter
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.game_state import Action, GameState
from cascadia_ai.positions import HexPosition, adjacent_positions
from cascadia_ai.wildlife_layer import FoxLayer, WildlifeLayer


features_shapes = [[13], [37, 11], [34, 5]]


def pad_list(data: list[list], length: int):
    while len(data) < length:
        data.append([0] * len(data[0]))
    return data


def get_wreward(
    wlayers: dict[Wildlife, WildlifeLayer | FoxLayer],
    pos: HexPosition,
    wildlife: Wildlife,
):
    reward = wlayers[wildlife].get_reward(pos, wildlife)

    if reward is None:
        return 0

    if wildlife == Wildlife.FOX:
        return reward or 0

    fox_reward = wlayers[Wildlife.FOX].get_reward(pos, wildlife) or 0

    return (reward or 0) + fox_reward


def get_features(s: GameState, a: Action | None = None):
    if a is None:
        turns_remaining = s.turns_remaining
        nature_tokens = s.env.nature_tokens
        hlayers = s.env.hlayers
        wlayers = s.env.wlayers
        unoccupied_tiles = set(s.env.unoccupied_tiles())
        adjacent_empty = s.env.all_adjacent_empty()

    else:
        placed_tile = s.tile_display[a.tile_index].rotate(a.tile_rotation)
        if a.wildlife_position is None:
            wildlife_target = None
        else:
            wildlife_target = (
                placed_tile
                if a.wildlife_position == a.tile_position
                else s.env.tiles[a.wildlife_position]
            )

        turns_remaining = s.turns_remaining - 1

        nature_tokens = s.env.nature_tokens
        if a.nt_spent:
            nature_tokens -= 1
        if wildlife_target is not None and wildlife_target.nature_token_reward:
            nature_tokens += 1

        hlayers = {
            h: area.place_tile(a.tile_position, placed_tile)
            for h, area in s.env.hlayers.items()
        }

        wlayers = (
            s.env.wlayers
            if a.wildlife_position is None
            else {
                w: layer.place_wildlife(
                    a.wildlife_position, s.wildlife_display[a.wildlife_index]
                )
                for w, layer in s.env.wlayers.items()
            }
        )

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

    bsizes = Counter(len(g) for g in wlayers[Wildlife.BEAR]._groups)

    global_features = [
        turns_remaining,
        nature_tokens,
        bsizes[2],
        bsizes[1],
        wlayers[Wildlife.ELK].count,
        wlayers[Wildlife.SALMON].count,
        wlayers[Wildlife.HAWK].count,
        wlayers[Wildlife.FOX].count,
        *(hlayers[h].largest_area for h in Habitat),
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
                        # else sum(wlayers[w2].get_reward(pos, w) or 0 for w2 in Wildlife)
                        else get_wreward(wlayers, pos, w)
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
                *(
                    # sum(wlayers[w2].get_reward(pos, w) or 0 for w2 in Wildlife)
                    get_wreward(wlayers, pos, w)
                    for w in Wildlife
                ),
            ]
        )

        tile_rewards.append([hlayers[h].get_best_reward(pos) for h in Habitat])

    return (
        np.array(global_features, dtype=np.float32),
        np.array(pad_list(wildlife_rewards, 37), dtype=np.float32),
        np.array(pad_list(tile_rewards, 34), dtype=np.float32),
    )


def get_next_features(s: GameState, actions: list[Action]):
    features_list = [get_features(s, a) for a in actions]
    return [
        np.stack([f[i] for f in features_list]) for i in range(len(features_list[0]))
    ]
