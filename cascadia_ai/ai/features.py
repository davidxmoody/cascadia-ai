from cascadia_ai.enums import Wildlife
from cascadia_ai.environments import Environment
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_wildlife_score
from itertools import product


def flatten_features(
    features: dict[str, float | int | bool | dict[str, float | int | bool]]
) -> dict[str, float]:
    new_features: dict[str, float] = {}

    for key, value in features.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                new_features[f"{key}_{subkey}"] = float(subvalue)
        else:
            new_features[key] = float(value)

    return new_features


def habitat_group_sizes(env: Environment):
    features: dict[str, int] = {}
    for habitat, groups in env.habitat_groups().items():
        group_sizes = sorted((len(g) for g in groups), reverse=True)
        group_sizes += [0, 0, 0]
        features[f"hgroup_0_{habitat.value}"] = group_sizes[0]
        features[f"hgroup_1_{habitat.value}"] = group_sizes[1]
        features[f"hgroup_2_{habitat.value}"] = group_sizes[2]
        features[f"hgroup_bonus_{habitat.value}"] = 2 if group_sizes[0] >= 7 else 0
    return features


def unoccupied_potential(env: Environment, wscore: dict[Wildlife, int]):
    unoccupied = list(env.unoccupied_tiles())

    combinations = product(*(t.wildlife_slots for _, t in unoccupied))

    wdiffs: list[dict[str, int]] = []

    for combination in combinations:
        new_env = env.copy()
        for (pos, _), w in zip(unoccupied, combination):
            new_env.place_wildlife(pos, w)
        comb_wscore = calculate_wildlife_score(new_env)
        wdiffs.append({w.value: comb_wscore[w] - wscore[w] for w in comb_wscore})

    return {
        "potential_wdiff_max": {
            w.value: max(wdiff[w.value] for wdiff in wdiffs) for w in Wildlife
        },
        "potential_wdiff_best": max(wdiffs, key=lambda wdiff: sum(wdiff.values())),
    }


def get_features(gs: GameState) -> dict[str, float]:
    wscore = calculate_wildlife_score(gs.env)

    features = {
        "nature_tokens": gs.nature_tokens,
        "turns_remaining": gs.turns_remaining,
        "open_tiles": gs.env.num_tiles_placed - gs.env.num_wildlife_placed,
        "wscore": {k.value: v for k, v in wscore.items()},
        **unoccupied_potential(gs.env, wscore),
        **habitat_group_sizes(gs.env),
    }

    # TODO add info about remaining wildlife counts in bag and habitat counts in supply

    # TODO add in features for what would remain, but this also requires
    # calculating it based on the old gs and action because the overpopulation
    # check may change it in the new gs
    # TODO could maybe just do tiles for now and then do wildlife after some refactoring
    # leftover_tile_indexes = list(range(4))
    # features["leftover_habitats"] = gs

    # TODO info about how easy it could be to connect them or extend habitat groups

    # TODO single bears (that could potentially have a bear next to them)
    # TODO number of bear pairs that could be filled in by placing on an existing slot

    # TODO open hawk slots not next to another hawk
    # TODO fox features
    # TODO salmon group sizes
    # TODO could salmon be connected (accounting for triangles)
    # TODO max elk score if filled in elk spots with elk

    return flatten_features(features)


# print_gs(gs)
# get_gs_features(gs)
