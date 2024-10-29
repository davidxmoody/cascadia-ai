from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.adam import Adam
import torch
import torch.nn as nn
import lightning as L
from tqdm import tqdm
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.ai.features import get_features
from cascadia_ai.ai.training_data import get_greedy_played_games
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score
import numpy as np
import pandas as pd
from statistics import mean


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
        mean(score.total for score in scores) - calculate_score(state).total
        for state, scores in tqdm(greedy_played_games, desc="Calculating labels")
    ]
)


# %%
class DQNLightning(L.LightningModule):
    def __init__(self, main_dim: int, set1_dim: int, set2_dim: int, hidden_dim: int):
        super().__init__()

        self.set1_network = nn.Sequential(
            nn.Linear(set1_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.set2_network = nn.Sequential(
            nn.Linear(set2_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.final_network = nn.Sequential(
            nn.Linear(main_dim + 2 * hidden_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, main_features, set1_features, set2_features):  # type: ignore
        set1_out = self.set1_network(set1_features)
        set2_out = self.set2_network(set2_features)

        set1_max, set1_sum = set1_out.max(dim=1)[0], set1_out.sum(dim=1)
        set2_max, set2_sum = set2_out.max(dim=1)[0], set2_out.sum(dim=1)

        combined_features = torch.cat(
            [main_features, set1_max, set1_sum, set2_max, set2_sum], dim=1
        )

        return self.final_network(combined_features)

    def training_step(self, batch):  # type: ignore
        mainf, set1f, set2f, y = batch
        y_pred = self(mainf, set1f, set2f)
        loss = self.loss_fn(y_pred, y.reshape((-1, 1)))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):  # type: ignore
        mainf, set1f, set2f, y = batch
        y_pred = self(mainf, set1f, set2f)
        loss = self.loss_fn(y_pred, y.reshape((-1, 1)))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


# %%
num_hidden = 20
num_main_features = features_tensors[0].shape[1]
num_set1_features = features_tensors[1].shape[2]
num_set2_features = features_tensors[2].shape[2]

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

trainer = L.Trainer(max_epochs=50, logger=logger)

model = DQNLightning(
    num_main_features, num_set1_features, num_set2_features, num_hidden
)

trainer.fit(model, train_loader, val_loader)


# %%
def play_test_game(
    model: DQNLightning, state: GameState | None = None, gamma: float = 0.9
):
    if state is None:
        state = GameState()

    while state.turns_remaining > 0:
        actions, rewards = get_actions_and_rewards(state)

        if state.turns_remaining == 1:
            i = rewards.index(max(rewards))

        else:
            next_features = [get_features(state.copy().take_action(a)) for a in actions]
            features_tensors = [
                torch.from_numpy(np.stack([f[i] for f in next_features]))
                for i in range(len(next_features[0]))
            ]

            with torch.no_grad():
                q_values = model(*features_tensors).squeeze()
            expected_rewards = q_values * gamma + torch.tensor(rewards)
            i = expected_rewards.argmax().item()

        state.take_action(actions[i])

    return state


results = []
for _ in tqdm(range(100), desc="Playing test games"):
    end_state = play_test_game(model)
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
