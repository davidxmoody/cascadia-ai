from random import choice
from cascadia_ai.ai.transitions import get_transitions
from cascadia_ai.game_state import GameState


def play_game(state: GameState, until_turns_remaining=0):
    while state.turns_remaining > until_turns_remaining:
        actions, rewards, _ = get_transitions(state)
        max_reward = max(rewards)
        max_actions = [
            action for action, reward in zip(actions, rewards) if reward == max_reward
        ]
        action = choice(max_actions)
        state.take_action(action)
    return state
