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
from cascadia_ai.environments import HexPosition
from cascadia_ai.score import calculate_score


# %%
def get_node_features(state: GameState, p: HexPosition) -> list[float]:
    tile = state.env.tiles[p]
    if tile is None:
        raise Exception("Missing tile")
    wildlife = state.env.wildlife[p]

    wfilled = [wildlife == w for w in Wildlife]
    wpotential = [
        wildlife == w or (wildlife is None and w in tile.wildlife_slots)
        for w in Wildlife
    ]

    sides = [tile.habitats[0]] * 3 + [tile.habitats[1]] * 3
    for _ in range(tile.rotation):
        sides = [sides[-1]] + sides[:-1]

    sides_exploded = [h1 == h2 for h1 in sides for h2 in Habitat]

    bools = [tile.nature_token_reward, *wfilled, *wpotential, *sides_exploded]
    return [float(b) for b in bools]


# %%


def state_to_data(state: GameState, final_score: int):
    positions = list(state.env.tiles.keys())
    nodes = list[list[float]]()
    adjacency = list[tuple[int, int]]()

    for pi, p in enumerate(positions):
        nodes.append(get_node_features_v2(state, p))

        for a, _ in state.env.adjacent_tiles(p):
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

        while (state := gamestates[-1]).turns_remaining > 0:
            all_actions = list(state.available_actions())
            actions = sample(all_actions, min(20, len(all_actions)))
            chosen_action = max(
                actions, key=lambda a: calculate_score(state.take_action(a)).total
            )
            gamestates.append(state.take_action(chosen_action))

        final_score = calculate_score(gamestates[-1]).total

        for state in gamestates:
            yield state_to_data(state, final_score)


# %%
training_data = list(generate_training_data(500))


# %%
num_node_features = 23
batch_size = 100


class QNetwork(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):  # type: ignore
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc(x)

        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.005)

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
loader = DataLoader(training_data, batch_size=100, shuffle=True, num_workers=7)

model = QNetwork().to("mps")

trainer = L.Trainer(max_epochs=50)
trainer.fit(model, loader)


# %%
test_data = list(generate_training_data(200))


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


# %%
def get_node_features_v2(state: GameState, p: HexPosition) -> list[float]:
    tile = state.env.tiles[p]
    if tile is None:
        raise Exception("Missing tile")

    wildlife = state.env.wildlife[p]

    num_matching_edges = 0
    for ap, art in state.env.adjacent_tiles(p):
        q1, r1 = p
        q2, r2 = ap
        edge1 = tile.get_edge((q2 - q1, r2 - r1))
        edge2 = art.get_edge((q1 - q2, r1 - r2))
        if edge1 == edge2:
            num_matching_edges += 1

    features: list[float] = [
        float(num_matching_edges),
        float(tile.single_habitat),
        float(tile.nature_token_reward and wildlife is None),
    ]

    for w in [Wildlife.BEAR, Wildlife.ELK, Wildlife.SALMON, Wildlife.HAWK]:
        features.extend(
            [
                float(wildlife == w),
                float(wildlife is None and w in tile.wildlife_slots),
                float(sum(aw == w for _, aw in state.env.adjacent_wildlife(p))),
                float(
                    sum(
                        state.env.wildlife[apos] is None and w in atile.wildlife_slots
                        for apos, atile in state.env.adjacent_tiles(p)
                    )
                ),
            ]
        )

    adjacent_unique = {w for _, w in state.env.adjacent_wildlife(p)}
    adjacent_empty_slots = {
        w
        for apos, atile in state.env.adjacent_tiles(p)
        for w in atile.wildlife_slots
        if state.env.wildlife[apos] is None
    }
    features.extend(
        [
            float(wildlife == Wildlife.FOX),
            float(wildlife is None and Wildlife.FOX in tile.wildlife_slots),
            float(len(adjacent_unique)),
            float(len(adjacent_empty_slots)),
        ]
    )

    return features
