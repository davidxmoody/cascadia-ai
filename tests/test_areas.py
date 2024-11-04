import pickle
import cProfile
import pstats
from typing import Any
from cascadia_ai.areas import HabitatAreas
from cascadia_ai.enums import Habitat
from cascadia_ai.game_state import GameState


def test_areas():
    with open("data/test_state.pkl", "rb") as f:
        test_state: GameState = pickle.load(f)

    profiler = cProfile.Profile()

    ha = HabitatAreas(test_state.env.tiles)

    all_pos = test_state.env.all_adjacent_empty()
    first_pos = next(iter(all_pos))

    profiler.enable()
    ha.get_simple_tile_reward(first_pos, Habitat.WETLANDS)
    # for pos in all_pos:
    #     for h in Habitat:
    #         ha.get_simple_tile_reward(pos, h)
    profiler.disable()

    hgroups = test_state.env.habitat_groups()

    for h in Habitat:
        assert ha.largest_area(h) == len(hgroups[h][0])

    ps: Any = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.TIME)
    for func in ps.stats:
        cc, nc, tt, ct, callers = ps.stats[func]
        ps.stats[func] = (cc, nc, tt * 1000000, ct * 1000000, callers)
    ps.print_stats()
