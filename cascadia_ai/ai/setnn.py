from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from tqdm import tqdm
import lightning as L
import numpy as np
import pandas as pd
from statistics import mean

from cascadia_ai.ai.features import get_features, get_next_features
from cascadia_ai.ai.training_data import (
    get_greedy_played_games,
    get_model_played_games,
    get_model_realistic_states,
)
from cascadia_ai.ai.model import DQNLightning
from cascadia_ai.tui import print_state
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.game_state import GameState

# %%
greedy_played_games = get_greedy_played_games()


# %%
features_list = [
    get_features(state)
    for state, _ in tqdm(greedy_played_games, desc="Calculating features")
]

features_tensors = [
    torch.from_numpy(np.stack([f[i] for f in features_list]))
    for i in range(len(features_list[0]))
]

labels = torch.tensor(
    [
        mean(score.total for score in scores) - state.env.score.total
        for state, scores in tqdm(greedy_played_games, desc="Calculating labels")
    ]
)


# %%
dataset = TensorDataset(*features_tensors, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_ds,
    batch_size=100,
    shuffle=True,
    num_workers=7,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=100,
    num_workers=7,
    persistent_workers=True,
)

logger = TensorBoardLogger("tb_logs", name="setnn")

trainer = L.Trainer(max_epochs=100, logger=logger)

model = DQNLightning()

trainer.fit(model, train_loader, val_loader)


# %%
torch.save(model.state_dict(), "data/model.pth")


# %%
# model = DQNLightning()
# model.load_state_dict(torch.load("data/model.pth", weights_only=True))


# %%
get_model_realistic_states(model, load=False)


# %%
get_model_played_games(model, load=False)


# %%
def play_test_game(
    model: DQNLightning,
    state: GameState | None = None,
    gamma=0.9,
    until_turns_remaining=0,
):
    if state is None:
        state = GameState()

    while state.turns_remaining > until_turns_remaining:
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

    return state


results = []
for _ in tqdm(range(400), desc="Playing test games"):
    end_state = play_test_game(model)
    score = end_state.env.score
    print_state(end_state)
    results.append(
        {
            **{k.value: v for k, v in score.wildlife.items()},
            "wtotal": sum(score.wildlife.values()),
            **{k.value: v for k, v in score.habitat.items()},
            "htotal": sum(score.habitat.values()),
            "nt": score.nature_tokens,
            "total": score.total,
        }
    )

df = pd.DataFrame(results)
print(df.describe().T[["mean", "min", "max"]])
