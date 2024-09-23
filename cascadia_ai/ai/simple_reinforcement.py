from torch.utils.data import DataLoader, TensorDataset
from torch.optim.adam import Adam
import torch
import torch.nn as nn
import lightning as L
from random import random
from time import time
from cascadia_ai.ai.features2 import get_move_features, get_state_features

from cascadia_ai.game_state import GameState, Move
from cascadia_ai.score import calculate_score

device = "mps"
batch_size = 32

num_state_features = len(get_state_features(GameState()))
num_move_features = len(get_move_features(GameState(), Move(0, (0, 0), 0, 0, (0, 0))))


class DQNLightning(L.LightningModule):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc = nn.Linear(2 * hidden_dim, 1)

        self.loss_fn = nn.MSELoss()

    def forward(self, states, actions):  # type: ignore
        state_embedded = self.state_embedding(states)
        action_embedded = self.action_embedding(actions)

        combined = torch.cat((state_embedded, action_embedded), dim=-1)

        q_value = self.fc(combined).squeeze(-1)

        return q_value

    def training_step(self, batch):  # type: ignore
        states, actions, y = batch
        y_pred = self(states, actions)
        # loss = self.loss_fn(y_pred, y.reshape((-1, 1)))
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


def play_game(model: DQNLightning, epsilon: float, seed: int | None = None):
    # print("Play game start")
    pgstart = time()

    state = GameState(seed)
    score = calculate_score(state).total

    state_features: list[list[float]] = []
    action_features: list[list[float]] = []
    rewards: list[int] = []

    # TODO rename "moves" to "actions" everywhere

    while state.turns_remaining > 0:
        current_state_features = get_state_features(state)

        if random() < epsilon:
            action = state.get_random_move()

        else:
            # print("Available actions start")
            # aastart = time()
            available_actions = state.available_moves()
            available_action_features = [
                get_move_features(state, action) for action in available_actions
            ]
            # print("Available actions end:", time() - aastart)

            # print("Model call start")
            # mcstart = time()
            with torch.no_grad():
                # TODO make the call able to take a single state maybe?
                q_values = model(
                    torch.tensor([current_state_features] * len(available_actions)),
                    torch.tensor(available_action_features),
                )
            # print("Model call end:", time() - mcstart)

            max_index = int(torch.argmax(q_values).item())
            action = available_actions[max_index]

        state_features.append(current_state_features)
        action_features.append(get_move_features(state, action))

        state = state.make_move(action)
        new_score = calculate_score(state).total
        rewards.append(new_score - score)
        score = new_score

    # print("Play game end:", time() - pgstart)

    q_values = [float(sum(rewards[i:])) for i in range(len(rewards))]

    print(
        f"Play game score: {calculate_score(state).total:>3} in {time() - pgstart:.3f} with epsilon {epsilon:.2f}"
    )

    return (state_features, action_features, q_values)


def generate_dataset(model: DQNLightning, epsilon: float, num_games: int):
    state_featuress: list[list[float]] = []
    action_features: list[list[float]] = []
    q_values: list[float] = []

    for i in range(num_games):
        gstart = time()
        (sf, af, q) = play_game(model, epsilon)
        state_featuress.extend(sf)
        action_features.extend(af)
        q_values.extend(q)
        print(f"Generated game {i:>3} in {time() - gstart}s")

    return TensorDataset(
        torch.tensor(state_featuress),
        torch.tensor(action_features),
        torch.tensor(q_values),
    )


model = DQNLightning(num_state_features, num_move_features, 32)

epsilon = 1.0

for generation in range(1000):
    print(f"Generation {generation}, epsilon {epsilon}")

    dataset = generate_dataset(model, epsilon, 100)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=7,
        # persistent_workers=True,
    )

    trainer = L.Trainer(max_epochs=50)

    trainer.fit(model, dataloader)

    with torch.no_grad():
        (_, _, q) = play_game(model, 0, seed=0)
    print("Test score:", q[0])

    epsilon = max(epsilon - 0.1, 0.1)
