from random import randint, sample
import lightning as L
import torch
from torch import nn, tensor
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch.optim.adam import Adam
from cascadia_ai.enums import Habitat, Wildlife
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from cascadia_ai.game_state import GameState
from cascadia_ai.hex_grid import HexPosition
from cascadia_ai.score import calculate_score


# %%
def get_node_features(gs: GameState, p: HexPosition) -> list[float]:
    rtile = gs.env.tiles[p]
    if rtile is None:
        raise Exception("Missing tile")
    wildlife = gs.env.wildlife[p]

    wfilled = [wildlife == w for w in Wildlife]
    wpotential = [
        wildlife == w or (wildlife is None and w in rtile.tile.wildlife_slots)
        for w in Wildlife
    ]

    sides = [rtile.tile.habitats[0]] * 3 + [rtile.tile.habitats[1]] * 3
    for _ in range(rtile.rotation):
        sides = [sides[-1]] + sides[:-1]

    sides_exploded = [h1 == h2 for h1 in sides for h2 in Habitat]

    bools = [rtile.tile.nature_token_reward, *wfilled, *wpotential, *sides_exploded]
    return [float(b) for b in bools]


def gs_to_data(gs: GameState, final_score: int):
    positions = list(gs.env.tiles.keys())
    nodes = list[list[float]]()
    adjacency = list[tuple[int, int]]()

    for pi, p in enumerate(positions):
        nodes.append(get_node_features(gs, p))

        for a, _ in gs.env.tiles.adjacent(p):
            ai = positions.index(a)
            adjacency.append((pi, ai))

    x = tensor(nodes)
    edge_index = tensor(adjacency).t().contiguous()
    y = tensor([float(final_score)])

    return Data(x=x, edge_index=edge_index, y=y)


def generate_training_data(iterations=100):
    for i in range(iterations):
        print(i)
        seed = randint(0, 10000000)
        gamestates = [GameState(seed)]

        while (gs := gamestates[-1]).turns_remaining > 0:
            all_moves = list(gs.available_moves())
            moves = sample(all_moves, min(20, len(all_moves)))
            chosen_move = max(
                moves, key=lambda m: calculate_score(gs.make_move(m)).total
            )
            gamestates.append(gs.make_move(chosen_move))

        final_score = calculate_score(gamestates[-1]).total

        for gs in gamestates:
            yield gs_to_data(gs, final_score)


# %%
training_data = list(generate_training_data(100))


# %%
num_node_features = 41
batch_size = 100


class QNetwork(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):  # type: ignore
        # print("forward", x, edge_index, batch)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # print("forward 2", x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # print("forward 3", x)
        x = global_mean_pool(x, batch)

        # print("forward 4", x)
        x = self.fc(x)
        # print("forward 5", x)

        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def training_step(self, batch):  # type: ignore
        out = self(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(out, batch.y.reshape(-1, 1))
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss


# %%
loader = DataLoader(training_data, batch_size=100, shuffle=True)

model = QNetwork().to("mps")

trainer = L.Trainer(max_epochs=10)
trainer.fit(model, loader)


# %%
test_data = list(generate_training_data(100))


# %%
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model.eval()

total_loss = 0.0
num_batches = len(test_loader)
all_predictions = []
all_targets = []

loss_fn = torch.nn.MSELoss()

results = []

with torch.no_grad():
    for batch in test_loader:
        predictions = model(batch.x, batch.edge_index, batch.batch)

        loss = loss_fn(predictions, batch.y.reshape(-1, 1))
        total_loss += loss.item()

        for y, pred in zip(batch.y, predictions):
            results.append((y.item(), pred.item()))

average_loss = total_loss / num_batches

print(f"Average test loss: {average_loss:.4f}")

print(results)
