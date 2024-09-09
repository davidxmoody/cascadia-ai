from statistics import mean
from random import Random
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score


def main(iterations=100):
    results = []

    for seed in range(iterations):
        rand = Random(seed)
        gs = GameState(seed)

        while gs.turns_remaining > 0:
            move = rand.choice(list(gs.available_moves()))
            gs = gs.make_move(move)

        score = calculate_score(gs)
        print(score)
        results.append(score)

    print(f"\nMean score after {iterations} iterations: {mean(results)}")


if __name__ == "__main__":
    main()
