import pickle
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.adam import Adam
import torch
import torch.nn as nn
import lightning as L
from random import choice, random
from tqdm import tqdm
from cascadia_ai.ai.actions import get_actions
from cascadia_ai.ai.features import StateFeatures, feature_names
from cascadia_ai.enums import Wildlife
from cascadia_ai.game_state import Action, GameState
from cascadia_ai.score import calculate_score
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.inspection import permutation_importance


# %%
single_bear_bonus = 3


def adjust_reward(state: GameState, action: Action, reward: int):
    wildlife = state.wildlife_display[action.wildlife_index]
    placed_single_bear = (
        wildlife == Wildlife.BEAR
        and action.wildlife_position is not None
        and not state.env.has_adjacent_wildlife(action.wildlife_position, Wildlife.BEAR)
    )
    return reward + single_bear_bonus if placed_single_bear else reward


def play_game_greedy_epsilon_biased(epsilon: float, until_turns_remaining: int = 0):
    state = GameState()
    while state.turns_remaining > until_turns_remaining:
        actions_and_rewards = list(get_actions(state))

        if random() < epsilon:
            random_action = choice(actions_and_rewards)[0]
            state.take_action(random_action)

        else:
            actions_and_rewards = [
                (a, adjust_reward(state, a, r)) for a, r in actions_and_rewards
            ]

            max_reward = max(r for _, r in actions_and_rewards)
            max_actions = [a for a, r in actions_and_rewards if r == max_reward]
            max_actions_no_nt = [a for a in max_actions if not a.nt_spent]

            action = choice(
                max_actions_no_nt if len(max_actions_no_nt) else max_actions
            )
            state.take_action(action)
    return state


load_realistic_states = True

if load_realistic_states:
    with open("data/realistic_states.pkl", "rb") as f:
        realistic_states = pickle.load(f)

else:
    realistic_states = [
        play_game_greedy_epsilon_biased(0.1, i % 19 + 1)
        for i in tqdm(range(100000), desc=f"Generating realistic states")
    ]
    with open("data/realistic_states.pkl", "wb") as f:
        pickle.dump(realistic_states, f)


# %%
def play_game_greedy(state: GameState):
    while state.turns_remaining > 0:
        actions_and_rewards = list(get_actions(state))

        max_reward = max(r for _, r in actions_and_rewards)
        max_actions = [a for a, r in actions_and_rewards if r == max_reward]
        max_actions_no_nt = [a for a in max_actions if not a.nt_spent]

        action = choice(max_actions_no_nt if len(max_actions_no_nt) else max_actions)
        state.take_action(action)
    return state


load_greedy_played_games = True

if load_greedy_played_games:
    with open("data/greedy_played_games.pkl", "rb") as f:
        greedy_played_games = pickle.load(f)

else:
    greedy_played_games = [
        (state, play_game_greedy(state.copy()))
        for state in tqdm(realistic_states, desc="Greedy playing games")
    ]
    with open("data/greedy_played_games.pkl", "wb") as f:
        pickle.dump(greedy_played_games, f)


# %%
features = torch.from_numpy(
    np.vstack(
        [
            StateFeatures(s1)._data
            for s1, _ in tqdm(greedy_played_games, desc="Getting features")
        ]
    )
)

labels = torch.tensor(
    [
        float(calculate_score(s2).total - calculate_score(s1).total)
        for s1, s2 in tqdm(greedy_played_games, desc="Calculating labels")
    ]
)


# %%
class DQNLightning(L.LightningModule):
    def __init__(self, input_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

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
        self.log("val_loss", loss, prog_bar=False)

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

model = DQNLightning(num_features)

trainer.fit(model, train_loader, val_loader)


# %%
def play_test_game(model: DQNLightning, state: GameState, gamma: float = 0.9):
    state = GameState()

    while state.turns_remaining > 0:
        # TODO consider refactoring get_actions to return separate lists
        actions, rewards = zip(*get_actions(state))

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


# %%
model_played_games = []
for state in tqdm(realistic_states[:1000], desc="Model playing games"):
    played_state = play_test_game(model, state.copy())
    model_played_games.append((state, played_state))


with open("data/model_played_games.pkl", "wb") as f:
    pickle.dump(model_played_games, f)


# %%
with open("data/model_played_games.pkl", "rb") as f:
    model_played_games = pickle.load(f)


# %%
results = []
for _ in tqdm(range(100), desc="Playing test games"):
    score = calculate_score(play_test_game(model, GameState()))
    results.append(
        {
            **{k.value: v for k, v in score.wildlife.items()},
            **{k.value: v for k, v in score.habitat.items()},
            "nt": score.nature_tokens,
            "total": score.total,
        }
    )

df = pd.DataFrame(results)
print(df.mean().T)


# %%
class SklearnWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
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
