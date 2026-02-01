import torch
import torch.nn as nn
import lightning as L
from torch.optim.adam import Adam

from cascadia_ai.ai.features import features_shapes


class DQNLightning(L.LightningModule):
    def __init__(self, shapes: list[list[int]] = features_shapes, hidden_dim: int = 20):
        super().__init__()

        main_dim = shapes[0][0]
        set1_dim = shapes[1][1]
        set2_dim = shapes[2][1]

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


def load_model(path: str = "data/model.pth") -> DQNLightning:
    model = DQNLightning()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
