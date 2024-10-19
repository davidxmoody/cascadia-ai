from collections import Counter
from statistics import mean
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
import lightning as L
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.ai.training_data import get_greedy_played_games
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import HexPosition, adjacent_positions
from cascadia_ai.game_state import GameState
from cascadia_ai.score import Score, calculate_score
import pandas as pd


# %%
greedy_played_games = get_greedy_played_games()


# %%
def encodew(*wildlife: Wildlife | None):
    counter = Counter(wildlife)
    return [counter[w] for w in Wildlife]


def encodeh(*habitats: Habitat | None):
    counter = Counter(habitats)
    return [counter[h] for h in Habitat]


def get_node_features(state: GameState, pos: HexPosition):
    tile = state.env.tiles.get(pos)
    wildlife = state.env.wildlife.get(pos)

    return [
        tile is not None,
        *encodeh(*(tile.habitats if tile is not None else [])),
        *encodew(wildlife),
        *encodew(
            *(tile.wildlife_slots if tile is not None and wildlife is None else [])
        ),
        tile is not None and wildlife is None and tile.nature_token_reward,
    ]


def get_global_features(state: GameState):
    return [
        state.turns_remaining,
        state.nature_tokens,
        # TODO add things in display
    ]


def get_data(state: GameState, label: float = 0):
    positions = list(state.env.tiles.keys()) + list(state.env.all_adjacent_empty())

    x = torch.tensor(
        [get_node_features(state, pos) for pos in positions], dtype=torch.float32
    )
    edge_index_tuples = [
        (i, positions.index(apos))
        for i, pos in enumerate(positions)
        for apos in adjacent_positions(pos)
        if apos in positions
    ]
    edge_index = torch.tensor(edge_index_tuples).t().contiguous()

    graph_data = Data(x=x, edge_index=edge_index)
    non_graph_features = torch.tensor(get_global_features(state), dtype=torch.float32)
    return graph_data, non_graph_features, torch.tensor(label, dtype=torch.float32)


# %%
class GameDataset(Dataset):
    def __init__(self, played_games: list[tuple[GameState, list[Score]]]):
        self.played_games = played_games

    def __len__(self):
        return len(self.played_games)

    def __getitem__(self, idx: int):  # type: ignore
        state, scores = self.played_games[idx]
        return get_data(
            state, mean(s.total for s in scores) - calculate_score(state).total
        )


dataset = GameDataset(greedy_played_games[:10000])


# %%
class GNNModel(L.LightningModule):
    def __init__(self, num_node_features, num_additional_features):
        super().__init__()
        num_hidden = 16
        self.conv1 = GCNConv(num_node_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        self.fc1 = nn.Linear(2 * num_hidden + num_additional_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)

    def forward(self, graph_data, additional_features):  # type: ignore
        x = self.conv1(graph_data.x, graph_data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, graph_data.edge_index)
        x = torch.relu(x)

        x_sum = global_mean_pool(x, graph_data.batch)
        x_max = global_max_pool(x, graph_data.batch)

        x = torch.cat([x_sum, x_max, additional_features], dim=1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch):  # type: ignore
        graph_data, additional_features, target = batch
        output = self.forward(graph_data, additional_features)
        loss = nn.MSELoss()(output.squeeze(), target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):  # type: ignore
        graph_data, additional_features, target = batch
        output = self.forward(graph_data, additional_features)
        loss = nn.MSELoss()(output.squeeze(), target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


# %%
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100)

logger = TensorBoardLogger("tb_logs", name="gnn")

model = GNNModel(17, 2)

trainer = L.Trainer(max_epochs=100, logger=logger)

trainer.fit(model, train_loader, val_loader)

# %%
model = GNNModel(17, 2)
optimizer = Adam(model.parameters(), lr=0.01)

criterion = nn.MSELoss()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for graph_data, additional_features, target in dataloader:
        optimizer.zero_grad()
        out = model(graph_data, additional_features)
        loss = criterion(out.squeeze(), target.float())
        print(loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


# %%
def play_test_game(model: GNNModel, state: GameState, gamma: float = 0.9):
    while state.turns_remaining > 0:
        actions, rewards = get_actions_and_rewards(state)

        if state.turns_remaining == 1:
            i = rewards.index(max(rewards))

        else:
            next_states = [state.copy().take_action(a) for a in actions]
            graph_data = []
            additional_features = []
            for ns in next_states:
                gd, af, _ = get_data(ns)
                graph_data.append(gd)
                additional_features.append(af)
            graph_data_batched = Batch.from_data_list(graph_data)
            additional_features_batched = torch.stack(additional_features)

            with torch.no_grad():
                q_values = model(
                    graph_data_batched, additional_features_batched
                ).squeeze()
            expected_rewards = q_values * gamma + torch.tensor(rewards)
            i = expected_rewards.argmax().item()

        state.take_action(actions[i])

    print(calculate_score(state))
    return state


results = []
for _ in tqdm(range(100), desc="Playing test games"):
    end_state = play_test_game(model, GameState())
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
