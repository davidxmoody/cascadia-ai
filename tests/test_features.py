import cProfile
import pstats
from typing import Any
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.game_state import GameState
from cascadia_ai.ai.features import get_next_features
from cascadia_ai.tui import print_state


def test_features():
    state = GameState()

    while state.turns_remaining > 5:
        actions, rewards = get_actions_and_rewards(state)
        action = actions[rewards.index(max(rewards))]
        state.take_action(action)

    print_state(state)

    actions, rewards = get_actions_and_rewards(state)

    profiler = cProfile.Profile()
    profiler.enable()

    get_next_features(state, actions)

    profiler.disable()

    ps: Any = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.TIME)
    for func in ps.stats:
        cc, nc, tt, ct, callers = ps.stats[func]
        ps.stats[func] = (cc, nc, tt * 1000, ct * 1000, callers)
    ps.print_stats()
