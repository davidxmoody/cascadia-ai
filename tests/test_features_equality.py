from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.game_state import GameState
from cascadia_ai.ai.features import get_features
from cascadia_ai.tui import print_state
import numpy as np


def test_features():
    state = GameState()

    while state.turns_remaining > 0:
        print_state(state)

        actions, rewards = get_actions_and_rewards(state)
        for action in actions:
            features = get_features(state, action)
            state_copy = state.copy()
            state_copy.take_action(action)
            features_slow = get_features(state_copy)

            for i in range(len(features)):
                fsorted = np.sort(features[i], axis=0)
                fslowsorted = np.sort(features_slow[i], axis=0)
                equal = np.array_equal(fsorted, fslowsorted)
                if not equal:
                    print(fsorted)
                    print(fslowsorted)
                    print(fsorted != fslowsorted)
                    print(action)
                    print_state(state)
                    print_state(state_copy)
                assert equal

        action = actions[rewards.index(max(rewards))]
        state.take_action(action)
