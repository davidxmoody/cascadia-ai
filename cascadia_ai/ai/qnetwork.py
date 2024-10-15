from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.adam import Adam
import torch
import torch.nn as nn
import lightning as L
from tqdm import tqdm
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.ai.features import StateFeatures, feature_names
from cascadia_ai.ai.training_data import get_greedy_played_games
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score
from cascadia_ai.tui import print_state
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.inspection import permutation_importance
from statistics import mean


# %%
greedy_played_games = get_greedy_played_games()


# %%
features = torch.from_numpy(
    np.vstack(
        [
            StateFeatures(state)._data
            for state, _ in tqdm(greedy_played_games, desc="Calculating features")
        ]
    )
)

labels = torch.tensor(
    [
        mean(score.total for score in scores)
        for _, scores in tqdm(greedy_played_games, desc="Calculating labels")
    ]
)


# %%
class DQNLightning(L.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.loss_fn = nn.MSELoss()

    def forward(self, x):  # type: ignore
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch):  # type: ignore
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y.reshape((-1, 1)))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):  # type: ignore
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y.reshape((-1, 1)))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


# %%
num_features = len(features[0])

dataset = TensorDataset(features, labels)

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

logger = TensorBoardLogger("tb_logs", name="qnetwork")

trainer = L.Trainer(max_epochs=100, logger=logger)

model = DQNLightning(num_features, 20)

trainer.fit(model, train_loader, val_loader)


# %%
def play_test_game(model: DQNLightning, state: GameState, gamma: float = 0.9):
    while state.turns_remaining > 0:
        actions, rewards = get_actions_and_rewards(state)

        if state.turns_remaining == 1:
            i = rewards.index(max(rewards))

        else:
            next_features = torch.from_numpy(
                StateFeatures(state).get_next_features(actions)
            )
            with torch.no_grad():
                q_values = model(next_features).squeeze()
            expected_rewards = q_values * gamma + torch.tensor(rewards)
            i = expected_rewards.argmax().item()

        state.take_action(actions[i])

    return state


results = []
for _ in tqdm(range(100), desc="Playing test games"):
    end_state = play_test_game(model, GameState())
    print_state(end_state)
    score = calculate_score(end_state)
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


# %%
class SklearnWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self):
        pass

    def predict(self, X):
        with torch.no_grad():
            return self.model(torch.FloatTensor(X)).numpy()


wrapper_model = SklearnWrapper(model)

results = permutation_importance(
    wrapper_model, features, labels, n_repeats=30, scoring="neg_mean_squared_error"
)

feature_importance_df = pd.DataFrame(
    {"feature_name": feature_names, "importance": results["importances_mean"]}
)

feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
)

fig = px.bar(
    feature_importance_df,
    x="importance",
    y="feature_name",
    orientation="h",
    title="Permutation Importance of Features",
    labels={"importance": "Permutation Importance", "feature_name": "Features"},
    text="importance",
)

fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig.show()
