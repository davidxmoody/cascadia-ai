import pandas as pd
from tqdm import tqdm
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score
from cascadia_ai.ai.greedy import play_game


iterations = 100

results: list[dict] = []

for _ in tqdm(range(iterations), desc="Playing games"):
    final_state = play_game(GameState())
    score = calculate_score(final_state)

    results.append(
        {
            **{k.value: v for k, v in score.wildlife.items()},
            **{k.value: v for k, v in score.habitat.items()},
            "nt": score.nature_tokens,
            "total": score.total,
        }
    )

df = pd.DataFrame(results)
df.to_csv("results/greedy.tsv", sep="\t", index=False)
