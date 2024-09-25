from statistics import mean
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score


def main(iterations=100):
    results = []

    for seed in range(iterations):
        state = GameState(seed)

        while state.turns_remaining > 0:
            state = state.take_action(state.get_random_action())

        results.append(calculate_score(state))

    print(f"\nMean score after {iterations} iterations: {mean(results)}")


if __name__ == "__main__":
    main()
