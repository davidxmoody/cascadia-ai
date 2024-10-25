from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.adam import Adam
import torch
import torch.nn as nn
import lightning as L
from tqdm import tqdm
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.ai.features3 import StateFeatures, feature_names
from cascadia_ai.ai.training_data import get_greedy_played_games
from cascadia_ai.enums import Wildlife
from cascadia_ai.environments import HexPosition
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score
import numpy as np
import pandas as pd
from statistics import mean


# %%
greedy_played_games = get_greedy_played_games()


# %%
features_list = [
    StateFeatures(state).get_features()
    for state, _ in tqdm(greedy_played_games, desc="Calculating features")
]

set_features = torch.from_numpy(np.stack([a for a, _ in features_list]))
extra_features = torch.from_numpy(np.vstack([b for _, b in features_list]))

labels = torch.tensor(
    [
        mean(score.total for score in scores) - calculate_score(state).total
        for state, scores in tqdm(greedy_played_games, desc="Calculating labels")
    ]
)


# %%
class DQNLightning(L.LightningModule):
    def __init__(self, set_feature_dim: int, extra_feature_dim: int, hidden_dim: int):
        super().__init__()

        self.element_mlp = nn.Sequential(
            nn.Linear(set_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fc1 = nn.Linear(2 * hidden_dim + extra_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.loss_fn = nn.MSELoss()

    def forward(self, set_features, extra_features):  # type: ignore
        # print("forward", set_features.shape, extra_features.shape)
        set_result = self.element_mlp(set_features)
        # print("set_result", set_result.shape)
        set_sum = set_result.sum(dim=1)
        # print("set_sum", set_sum.shape)
        set_max, _ = set_result.max(dim=1)
        # print("set_max", set_max.shape)
        combined = torch.cat([set_sum, set_max, extra_features], dim=1)
        # print("combined", combined.shape)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch):  # type: ignore
        setf, extraf, y = batch
        y_pred = self(setf, extraf)
        loss = self.loss_fn(y_pred, y.reshape((-1, 1)))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):  # type: ignore
        setf, extraf, y = batch
        y_pred = self(setf, extraf)
        loss = self.loss_fn(y_pred, y.reshape((-1, 1)))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


# %%
num_set_features = 15
num_extra_features = len(feature_names)
num_hidden = 20

dataset = TensorDataset(set_features, extra_features, labels)

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

model = DQNLightning(num_set_features, num_extra_features, num_hidden)

trainer.fit(model, train_loader, val_loader)


# %%
def play_test_game(model: DQNLightning, state: GameState, gamma: float = 0.9):
    while state.turns_remaining > 0:
        actions, rewards = get_actions_and_rewards(state)

        if state.turns_remaining == 1:
            i = rewards.index(max(rewards))

        else:
            # next_features = torch.from_numpy(
            #     StateFeatures(state).get_next_features(actions)
            # )
            nfs = [
                StateFeatures(state.copy().take_action(a)).get_features()
                for a in actions
            ]
            next_set_features = torch.from_numpy(np.stack([a for a, _ in nfs]))
            next_extra_features = torch.from_numpy(np.stack([b for _, b in nfs]))

            with torch.no_grad():
                q_values = model(next_set_features, next_extra_features).squeeze()
            expected_rewards = q_values * gamma + torch.tensor(rewards)
            i = expected_rewards.argmax().item()

        state.take_action(actions[i])

    return state


results = []
for _ in tqdm(range(100), desc="Playing test games"):
    end_state = play_test_game(model, GameState())
    # print_state(end_state)
    score = calculate_score(end_state)
    print(score)
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
