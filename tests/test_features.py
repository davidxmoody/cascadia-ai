import pickle
import cProfile
import pstats
from typing import Any
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.areas import HabitatAreas
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.game_state import GameState
from cascadia_ai.ai.features import get_next_features
from cascadia_ai.groups import WildlifeGroups


def test_features():
    with open("data/test_state.pkl", "rb") as f:
        test_state: GameState = pickle.load(f)
        # TODO fix this
        test_state.env.hareas = {
            h: HabitatAreas(h, test_state.env.tiles) for h in Habitat
        }
        test_state.env.wgroups = {
            w: WildlifeGroups(w, test_state.env.wildlife) for w in Wildlife
        }

    profiler = cProfile.Profile()

    actions, _ = get_actions_and_rewards(test_state)

    profiler.enable()

    get_next_features(test_state, actions)

    profiler.disable()

    ps: Any = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.TIME)
    for func in ps.stats:
        cc, nc, tt, ct, callers = ps.stats[func]
        ps.stats[func] = (cc, nc, tt * 1000, ct * 1000, callers)
    ps.print_stats()
