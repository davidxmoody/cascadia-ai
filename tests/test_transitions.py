import pytest
from random import choice
from cascadia_ai.game_state import GameState
from cascadia_ai.ai.transitions import get_transitions
from cascadia_ai.score import calculate_score
from cascadia_ai.tui import print_gs


seeds = list(range(10))


@pytest.mark.parametrize("seed", seeds)
def test_get_transitions(seed: int):
    state = GameState(seed)

    while state.turns_remaining > 0:
        score = calculate_score(state)
        actions, rewards, _ = get_transitions(state)

        for action, reward in zip(actions, rewards):
            new_state = state.take_action(action)
            new_score = calculate_score(new_state)

            # TODO only check a fraction of the returned ones to make the test faster

            if reward != new_score.total - score.total:
                print_gs(state)
                print_gs(new_state)
                print(action, reward)
                print(score)
                print(new_score)

            assert reward == new_score.total - score.total

        state = state.take_action(choice(actions))
