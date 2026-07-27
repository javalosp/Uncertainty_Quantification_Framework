import torch
import torch.nn as nn
import torch.nn.functional as F


class NBeatsBlock(nn.Module):
    def __init__(self, input_dim: int, theta_dim: int, hidden_dim: int = 128):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        # theta_b reconstructs backcast (input_dim), theta_f projects forecast (theta_dim)
        self.theta_b = nn.Linear(hidden_dim, input_dim)
        self.theta_f = nn.Linear(hidden_dim, theta_dim)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))

        backcast = self.theta_b(h)
        forecast = self.theta_f(h)
        return backcast, forecast


class ProbabilisticNBeats(nn.Module):
    """N-BEATS architecture outputs Mean and Variance via Gaussian NLL optimization."""

    def __init__(self, backcast_length: int, forecast_length: int, num_blocks: int = 3, hidden_dim: int = 128):
        super(ProbabilisticNBeats, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_dim=backcast_length,
                theta_dim=forecast_length,
                hidden_dim=hidden_dim,
            )
            for _ in range(num_blocks)
        ])

        # Dual output heads for Mean and Standard Deviation
        self.head_mean = nn.Linear(forecast_length, forecast_length)
        self.head_std = nn.Linear(forecast_length, forecast_length)

    def forward(self, x: torch.Tensor):
        residuals = x
        forecast_accumulator = 0

        for block in self.blocks:
            backcast, forecast = block(residuals)
            residuals = residuals - backcast
            forecast_accumulator = forecast_accumulator + forecast

        mean = self.head_mean(forecast_accumulator)
        # Softplus guarantees positive standard deviation
        std = F.softplus(self.head_std(forecast_accumulator)) + 1e-6

        return mean, std


def gaussian_nll_loss(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor):
    """Loss function for probabilistic training."""
    var = std ** 2
    loss = 0.5 * torch.log(var) + ((target - mean) ** 2) / (2 * var)
    return torch.mean(loss)