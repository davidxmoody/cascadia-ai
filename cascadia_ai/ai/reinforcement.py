from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from cascadia_ai.ai.features import get_features
from torch.optim.adam import Adam
import torch
import torch.nn as nn
import lightning as L
from random import random, sample
from collections import deque

from cascadia_ai.game_state import GameState
from cascadia_ai.score import calculate_score


device = "mps"


class RLDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        # TODO try defining this with sequentially and include the relu bits in there
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNLightning(L.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
    ):
        super(DQNLightning, self).__init__()
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque[GameState](maxlen=10000)
        self.batch_size = 32
        self.update_target_every = 10

    def forward(self, x):  # type: ignore
        return self.model(x)

    def training_step(self, batch):  # type: ignore
        # if len(self.replay_buffer) < self.batch_size:
        #     return None

        ids = batch.tolist()

        print("training step", ids)

        # TODO consider re-randomising the seed when plucking states from replay buffer
        # states = sample(self.replay_buffer, self.batch_size)

        states = [self.replay_buffer[id] for id in ids]
        features = torch.tensor([list(get_features(s).values()) for s in states]).to(
            device
        )
        predicted_q_values = self(features)

        target_q_values = []

        print("predicted", predicted_q_values)

        for state in states:
            print("states", state)

            score = calculate_score(state).total
            moves = sample(state.available_moves(), 20)
            next_states = [state.make_move(move) for move in moves]
            rewards = [calculate_score(ns).total - score for ns in next_states]

            if next_states[0].turns_remaining == 0:
                target_q_values.append(max(rewards))

            else:
                print("before next features")
                f = [list(get_features(ns).values()) for ns in next_states]
                print("after calc f")
                next_features = torch.tensor(f).to(device)
                print("before target")
                with torch.no_grad():
                    next_q_values: torch.Tensor = self.target_model(next_features)
                max_q_value, max_index = torch.max(next_q_values, dim=0)
                reward = rewards[int(max_index.item())]
                target_q_values.append(reward + self.gamma * max_q_value.item())

        target_q_values = torch.tensor(target_q_values).reshape((-1, 1)).to(device)
        print("target", target_q_values)

        loss = self.loss_fn(predicted_q_values, target_q_values)
        self.log("train_loss", loss)

        print("train loss", loss)

        if self.global_step % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def epsilon_greedy_next_state(self, state: GameState):
        if random() < self.epsilon:
            return state.get_random_next_state()
        else:
            # TODO consider caching next states somewhere
            next_states = list(state.get_all_next_states())
            next_features = [get_features(state) for state in next_states]
            with torch.no_grad():
                q_values = self.model(next_features)
            return next_states[torch.argmax(q_values)]

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_end, self.epsilon * (1 - 1 / self.epsilon_decay)
        )

    def store_state(self, state: GameState):
        self.replay_buffer.append(state)

    def train_dataloader(self):
        return DataLoader(
            RLDataset(len(self.replay_buffer)),  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
        )


num_features = len(get_features(GameState()))

model = DQNLightning(num_features, 1)

trainer = L.Trainer(max_epochs=500)

for episode in range(10000):
    print("episode start", episode)

    state = GameState()

    while state.turns_remaining > 1:
        state = model.epsilon_greedy_next_state(state)
        model.store_state(state)

    model.decay_epsilon()

    print("train start", episode)

    trainer.fit(model)

    print("episode end", episode)
