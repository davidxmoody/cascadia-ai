from random import choice
from statistics import mean
from time import time
from cascadia_ai.ai.transitions import get_transitions
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score
from cascadia_ai.tui import print_state


def main(iterations=100):
    start_time = time()
    results = []

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

        score = calculate_score(state)
        results.append(score.total)
        print(
            ", ".join(
                str(num).rjust(3)
                for num in [
                    *score.wildlife.values(),
                    *score.habitat.values(),
                    score.nature_tokens,
                    score.total,
                ]
            )
        )
        # TODO add score to state printing
        # print_state(state)

    print(f"\nMean score after {iterations} iterations: {mean(results):0.1f}")
    print(f"Time taken: {time() - start_time:0.1f}s")


if __name__ == "__main__":
    main()
