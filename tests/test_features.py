import pickle
import cProfile
import pstats
from typing import Any
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.game_state import GameState
from cascadia_ai.ai.features2 import StateFeatures


def test_features():
    with open("data/test_state.pkl", "rb") as f:
        test_state: GameState = pickle.load(f)

    profiler = cProfile.Profile()

    profiler.enable()

    sf = StateFeatures(test_state)
    actions, _ = get_actions_and_rewards(test_state)
    sf.get_next_features(actions)

    profiler.disable()

    ps: Any = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.TIME)
    for func in ps.stats:
        cc, nc, tt, ct, callers = ps.stats[func]
        ps.stats[func] = (cc, nc, tt * 1000, ct * 1000, callers)
    ps.print_stats()
