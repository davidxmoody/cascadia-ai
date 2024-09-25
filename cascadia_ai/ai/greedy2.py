from random import choice
from time import time
from cascadia_ai.ai.transitions import get_transitions
from cascadia_ai.game_state import GameState
from cascadia_ai.score import Score, calculate_score
from cascadia_ai.tui import print_state
import pandas as pd


def scores_to_df(scores: list[Score]):
    return pd.DataFrame(
        {
            **({k.value: v for k, v in score.wildlife.items()}),
            **({k.value: v for k, v in score.habitat.items()}),
            "nature": score.nature_tokens,
            "total": score.total,
        }
        for score in scores
    )


def main(iterations=100):
    start_time = time()
    results: list[Score] = []

    for seed in range(iterations):
        state = GameState(seed)

        while state.turns_remaining > 0:
            actions, rewards, _ = get_transitions(state)
            max_reward = max(rewards)
            max_actions = [
                action
                for action, reward in zip(actions, rewards)
                if reward == max_reward
            ]
            action = choice(max_actions)
            state = state.take_action(action)

        results.append(calculate_score(state))
        print_state(state)
        print("--------------------------------------------------------------\n")

    df = scores_to_df(results)
    print(f"Time taken: {time() - start_time:0.1f}s (for {iterations} iterations)")
    print("\nMean scores:")
    print(df.mean())


if __name__ == "__main__":
    main()
