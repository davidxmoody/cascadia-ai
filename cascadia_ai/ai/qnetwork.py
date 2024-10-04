import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.adam import Adam
import torch
import torch.nn as nn
import lightning as L
from random import choice, random
from tqdm import tqdm
from cascadia_ai.ai.actions import get_actions
from cascadia_ai.ai.features import StateFeatures
from cascadia_ai.enums import Wildlife
from cascadia_ai.game_state import Action, GameState
from cascadia_ai.score import calculate_score
import numpy as np
import pandas as pd


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


# %%
def generate_realistic_states(epsilon: float, num: int):
    for i in tqdm(range(num), desc=f"Generating realistic states (epsilon {epsilon})"):
        yield play_game_greedy_epsilon_biased(epsilon, i % 19 + 1)


realistic_states = list(generate_realistic_states(0.1, 100000))

# with open("data/realistic_states.pkl", "wb") as f:
#     pickle.dump(realistic_states, f)


# %%
with open("data/realistic_states.pkl", "rb") as f:
    realistic_states = pickle.load(f)


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


greedy_played_games = [
    (state, play_game_greedy(state.copy()))
    for state in tqdm(realistic_states, desc="Greedy playing games")
]


# with open("data/greedy_played_games.pkl", "wb") as f:
#     pickle.dump(greedy_played_games, f)


# %%
with open("data/greedy_played_games.pkl", "rb") as f:
    greedy_played_games = pickle.load(f)


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

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


# %%
num_features = len(features[0])

model = DQNLightning(num_features)

dataset = TensorDataset(features, labels)

dataloader = DataLoader(
    dataset,
    batch_size=100,
    shuffle=True,
    num_workers=7,
    persistent_workers=True,
)

trainer = L.Trainer(max_epochs=200)

trainer.fit(model, dataloader)


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

    return calculate_score(state)


# %%
def float_range(start, stop, step, repeat=1):
    while start <= stop:
        for _ in range(repeat):
            yield int(start * 100) / 100
        start += step


results = []
for gamma in tqdm(float_range(0.7, 1.2, 0.02, 50), desc="Playing test games"):
    score = play_test_game(model, GameState(), gamma)
    results.append(
        {
            "gamma": gamma,
            **{k.value: v for k, v in score.wildlife.items()},
            **{k.value: v for k, v in score.habitat.items()},
            "nt": score.nature_tokens,
            "total": score.total,
        }
    )

df = pd.DataFrame(results)
# df.describe().T
# px.bar(df.groupby("gamma").mean().reset_index(), x="gamma", y="total").show()
