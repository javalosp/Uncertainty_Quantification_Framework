import torch
import torch.nn as nn
import numpy as np


class VariationalLSTM(nn.Module):
    """LSTM model with persistent Monte Carlo Dropout during evaluation[cite: 7]."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout_rate: float = 0.2):
        super(VariationalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_mean = nn.Linear(hidden_size, 1)
        self.fc_logvar = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        # Apply dropout manual pass for single-layer edge cases[cite: 7]
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        last_step = self.dropout(last_step)

        mean = self.fc_mean(last_step)
        log_var = self.fc_logvar(last_step)
        return mean, log_var

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Runs n_samples forward passes with activated dropout to estimate uncertainty[cite: 7]."""
        self.train()  # Keep dropout enabled during inference[cite: 7]

        means = []
        log_vars = []

        with torch.no_grad():
            for _ in range(n_samples):
                m, lv = self.forward(x)
                means.append(m.unsqueeze(0))
                log_vars.append(lv.unsqueeze(0))

        # Stack predictions: shape [n_samples, batch_size, 1][cite: 7]
        means = torch.cat(means, dim=0)
        vars_aleatoric = torch.exp(torch.cat(log_vars, dim=0))

        # Epistemic uncertainty = Variance of Monte Carlo means[cite: 7]
        epistemic_var = torch.var(means, dim=0)
        # Aleatoric uncertainty = Mean of predicted variances[cite: 7]
        aleatoric_var = torch.mean(vars_aleatoric, dim=0)

        total_var = epistemic_var + aleatoric_var
        overall_mean = torch.mean(means, dim=0)

        # Output all 4 components: mean, total_std, aleatoric_std, epistemic_std[cite: 7]
        return (
            overall_mean.cpu().numpy(),
            torch.sqrt(total_var).cpu().numpy(),
            torch.sqrt(aleatoric_var).cpu().numpy(),
            torch.sqrt(epistemic_var).cpu().numpy(),
        )