from random import choice
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score


def test_get_actions():
    state = GameState()

    while state.turns_remaining > 0:
        score = calculate_score(state)
        actions, rewards = get_actions_and_rewards(state)

        for action, reward in zip(actions, rewards):
            new_state = state.copy()
            new_state.take_action(action)
            new_score = calculate_score(new_state)

            assert reward == new_score.total - score.total

        state.take_action(choice(actions))
