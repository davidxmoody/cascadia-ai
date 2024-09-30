from random import choice
from cascadia_ai.ai.actions import get_actions
from cascadia_ai.game_state import GameState


def play_game(state: GameState, until_turns_remaining=0):
    while state.turns_remaining > until_turns_remaining:
        actions_and_rewards = list(get_actions(state))

        max_reward = max(r for _, r in actions_and_rewards)
        max_actions = [a for a, r in actions_and_rewards if r == max_reward]
        max_actions_no_nt = [a for a in max_actions if a.tile_index == a.wildlife_index]

        action = choice(max_actions_no_nt if len(max_actions_no_nt) else max_actions)
        state.take_action(action)
    return state
