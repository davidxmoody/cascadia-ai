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

    ha = HabitatAreas(Habitat.WETLANDS, test_state.env.tiles)

    all_pos = test_state.env.all_adjacent_empty()

    for pos in all_pos:
        ha.get_best_reward(pos)

    profiler.enable()

    for pos in all_pos:
        ha.get_best_reward(pos)

    profiler.disable()

    #     hgroups = test_state.env.habitat_groups()
    #     assert ha.largest_area == len(hgroups[Habitat.WETLANDS][0])

    ps: Any = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.TIME)
    for func in ps.stats:
        cc, nc, tt, ct, callers = ps.stats[func]
        ps.stats[func] = (cc, nc, tt * 1000000, ct * 1000000, callers)
    ps.print_stats()


# def step(pos: HexPosition, rot: Rotation) -> tuple[HexPosition, Rotation]:
#     match rot:
#         case 0:
#             return ((pos[0], pos[1] + 1), 3)
#         case 1:
#             return ((pos[0] + 1, pos[1]), 4)
#         case 2:
#             return ((pos[0] + 1, pos[1] - 1), 5)
#         case 3:
#             return ((pos[0], pos[1] - 1), 0)
#         case 4:
#             return ((pos[0] - 1, pos[1]), 1)
#         case 5:
#             return ((pos[0] - 1, pos[1] + 1), 2)
#         case _:
#             raise Exception("Invalid rotation")
