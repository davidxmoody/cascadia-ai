from collections import deque
from numpy._typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.adam import Adam
import torch
import torch.nn as nn
import lightning as L
from random import choice, randint, random, sample
from tqdm import tqdm
from cascadia_ai.ai.transitions import get_transitions, feature_names
from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score
import numpy as np
from statistics import mean

device = "mps"
batch_size = 100


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
        # loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


# %%
def play_game(model: DQNLightning | None, epsilon: float, state: GameState):
    while state.turns_remaining > 0:
        actions, rewards, features = get_transitions(state)

        if random() < epsilon:
            i = randint(0, len(actions) - 1)

        elif model is None or state.turns_remaining == 1:
            max_reward = max(rewards)
            i = choice([i for i, r in enumerate(rewards) if r == max_reward])

        else:
            with torch.no_grad():
                turns_remaining = state.turns_remaining - 1
                expected_mean_rewards = model(torch.tensor(features)).squeeze()
                expected_total_rewards = (
                    expected_mean_rewards * turns_remaining + torch.tensor(rewards)
                )
                i = expected_total_rewards.argmax().item()

        state = state.take_action(actions[i])
        yield (actions[i], rewards[i], features[i], state)


# %%
def play_test_games(model: DQNLightning | None, epsilon: float, num_games: int):
    states: list[tuple[GameState, NDArray[np.float32]]] = []
    final_scores: list[int] = []

    for _ in tqdm(range(num_games), desc="Playing test games"):
        game_steps = list(play_game(model, epsilon, GameState()))
        *middle_states, final_state = [(s, f) for _, _, f, s in game_steps]
        states.extend(middle_states)
        final_scores.append(calculate_score(final_state[0]).total)

    print(f"Mean final score: {mean(final_scores)}")

    return states


# %%
seen_states = deque[tuple[GameState, NDArray[np.float32]]](maxlen=100000)
seen_states.extend(play_test_games(None, 0.2, 1000))


# %%
def generate_dataset(
    seen_states: deque[tuple[GameState, NDArray[np.float32]]],
    model: DQNLightning | None,
    epsilon: float,
    num_games: int,
):
    sampled = sample(seen_states, num_games)

    features_list = []
    labels = []

    for state, features in tqdm(sampled, desc="Generating dataset"):
        rewards = [r for _, r, _, _ in play_game(model, epsilon, state.reset_rand())]

        features_list.append(features)
        labels.append(mean(rewards))

    return TensorDataset(torch.tensor(features_list), torch.tensor(labels))


# %%
model = DQNLightning(len(feature_names))
epsilon = 1.0

for gen in range(1000):
    print(f"Generation {gen}, epsilon {epsilon}, num seen states: {len(seen_states)}")

    dataloader = DataLoader(
        # TODO i dont think this makes sense to train against expected values generated with the epsilon
        generate_dataset(seen_states, model, epsilon, 200),
        batch_size=batch_size,
        shuffle=True,
        num_workers=7,
        persistent_workers=True,
    )

    trainer = L.Trainer(max_epochs=100)

    trainer.fit(model, dataloader)

    # with torch.no_grad():
    #     final_state = list(play_game(model, 0, GameState(0)))[-1][3]
    #     test_score = calculate_score(final_state).total
    #     print("Test score:", test_score)

    epsilon = max(epsilon - 0.1, 0.1)
    seen_states.extend(play_test_games(model, 0.0, 200))


# %%
# s = GameState(0)
# while s.turns_remaining > 0:
#     print_state(s)
#     print("\n-----------------------------------------------------\n")

#     actions, rewards, features = get_transitions(s)
#     with torch.no_grad():
#         q_values = model(torch.tensor(features)).squeeze()
#         expected_rewards = q_values + torch.tensor(rewards)
#         i = expected_rewards.argmax().item()
#         print(i, rewards[:50], q_values[:50])
#     print()
#     s = s.take_action(actions[i])
# print_state(s)
