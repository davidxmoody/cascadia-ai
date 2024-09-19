from statistics import mean
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score


def main(iterations=100):
    results = []

    for seed in range(iterations):
        gs = GameState(seed)

        while gs.turns_remaining > 0:
            move = max(
                list(gs.available_moves()),
                key=lambda m: calculate_score(gs.make_move(m)).total,
            )
            gs = gs.make_move(move)

        score = calculate_score(gs)
        print(score)
        results.append(score)

    print(f"\nMean score after {iterations} iterations: {mean(results)}")


if __name__ == "__main__":
    main()
