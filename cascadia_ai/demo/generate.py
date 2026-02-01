import pickle
import torch

from cascadia_ai.ai.model import DQNLightning, load_model
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.ai.features import get_next_features
from cascadia_ai.game_state import GameState
from cascadia_ai.tui import print_state


def play_game_with_history(model: DQNLightning, gamma: float = 0.9) -> list[GameState]:
    states: list[GameState] = []
    state = GameState()
    states.append(state.copy())

    while state.turns_remaining > 0:
        actions, rewards = get_actions_and_rewards(state)

        if state.turns_remaining == 1:
            i = rewards.index(max(rewards))
        else:
            next_features = [
                torch.from_numpy(nparray)
                for nparray in get_next_features(state, actions)
            ]

            with torch.no_grad():
                q_values = model(*next_features).squeeze()
            expected_rewards = q_values * gamma + torch.tensor(rewards)
            i = expected_rewards.argmax().item()

        state.take_action(actions[i])
        states.append(state.copy())

    return states


if __name__ == "__main__":
    print("Loading model...")
    model = load_model()

    print("Playing game...")
    states = play_game_with_history(model)

    print(f"Game complete! {len(states)} states captured.")

    print_state(states[-1])

    output_path = "cascadia_ai/demo/game.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(states, f)
    print(f"Saved to {output_path}")
