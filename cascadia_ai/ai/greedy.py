from statistics import mean
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score


def main(iterations=100):
    results = []

    for seed in range(iterations):
        gs = GameState(seed)

        while gs.turns_remaining > 0:
            action = max(
                list(gs.available_actions()),
                key=lambda a: calculate_score(gs.take_action(a)).total,
            )
            gs = gs.take_action(action)

        score = calculate_score(gs)
        print(score)
        results.append(score)

    print(f"\nMean score after {iterations} iterations: {mean(results)}")


if __name__ == "__main__":
    main()
