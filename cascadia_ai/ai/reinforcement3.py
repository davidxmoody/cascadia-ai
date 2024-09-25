from torch.utils.data import DataLoader, TensorDataset
from torch.optim.adam import Adam
import torch
import torch.nn as nn
import lightning as L
from random import randint, random
from cascadia_ai.ai.transitions import get_transitions
from cascadia_ai.game_state import GameState
from cascadia_ai.score import Score, calculate_score
from cascadia_ai.tui import print_state
import pandas as pd

device = "mps"
batch_size = 100
num_features = len(get_transitions(GameState())[2][0])


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
        # loss = self.loss_fn(y_pred, y.reshape((-1, 1)))
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


def play_game(model: DQNLightning, epsilon: float, state: GameState):
    while state.turns_remaining > 0:
        actions, rewards, features = get_transitions(state)

        if random() < epsilon:
            i = randint(0, len(actions) - 1)

        elif state.turns_remaining == 1:
            i = rewards.index(max(rewards))

        else:
            with torch.no_grad():
                q_values = model(torch.tensor(features)).squeeze()
                expected_rewards = q_values + torch.tensor(rewards)
                i = expected_rewards.argmax().item()

        state = state.take_action(actions[i])
        yield (actions[i], rewards[i], features[i], state)


def generate_dataset(model: DQNLightning, epsilon: float, num_games: int):
    features_list: list[list[float]] = []
    q_values: list[float] = []

    for game_index in range(num_games):
        print(f"Generating game {game_index}")
        game_results = list(play_game(model, epsilon, GameState()))
        for i, step in enumerate(game_results[:-1]):
            q_values.append(float(sum(s[1] for s in game_results[i + 1 :])))
            features_list.append(step[2])

    return TensorDataset(torch.tensor(features_list), torch.tensor(q_values))


model = DQNLightning(num_features)
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
        final_state = list(play_game(model, 0, GameState(0)))[-1][3]
        test_score = calculate_score(final_state).total
        print("Test score:", test_score)

    epsilon = max(epsilon - 0.1, 0.1)


# %%
s = GameState(0)
while s.turns_remaining > 0:
    print_state(s)
    print("\n-----------------------------------------------------\n")

    actions, rewards, features = get_transitions(s)
    with torch.no_grad():
        q_values = model(torch.tensor(features)).squeeze()
        expected_rewards = q_values + torch.tensor(rewards)
        i = expected_rewards.argmax().item()
        print(i, rewards[:50], q_values[:50])
    print()
    s = s.take_action(actions[i])
print_state(s)


# %%
def scores_to_df(scores: list[Score]):
    return pd.DataFrame(
        {
            **({k.value: v for k, v in score.wildlife.items()}),
            **({k.value: v for k, v in score.habitat.items()}),
            "nature": score.nature_tokens,
            "total": score.total,
        }
        for score in scores
    )


final_states = []

for seed in range(1000):
    state = GameState(seed)

    while state.turns_remaining > 0:
        actions, rewards, features = get_transitions(state)
        with torch.no_grad():
            q_values = model(torch.tensor(features)).squeeze()
            expected_rewards = q_values + torch.tensor(rewards)
            i = expected_rewards.argmax().item()
            state = state.take_action(actions[i])

    final_states.append(state)
    print_state(state)
    print("--------------------------------------------------------------\n")

scores = scores_to_df([calculate_score(s) for s in final_states])
