import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
from .metrics import ProbabilisticMetrics

try:
    from src.neural_uq import VariationalLSTM, ProbabilisticNBeats, gaussian_nll_loss
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False


class NeuralForecasterWrapper:
    """Universal scikit-learn compatible wrapper for PyTorch neural UQ models."""
    def __init__(self, model_type='lstm', seq_len=5, epochs=120, lr=0.01, hidden_dim=32):
        self.model_type = model_type.lower()
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.model = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.last_window = None

    def fit(self, X, y):
        y_arr = np.array(y, dtype=float).ravel()
        self.y_mean = np.mean(y_arr)
        self.y_std = np.std(y_arr) if np.std(y_arr) > 0 else 1.0
        norm_y = (y_arr - self.y_mean) / self.y_std
        
        if len(norm_y) <= self.seq_len:
            raise ValueError(f"Time series length {len(norm_y)} must exceed sequence length {self.seq_len}.")
            
        self.last_window = norm_y[-self.seq_len:]
        
        X_seq, Y_seq = [], []
        for i in range(len(norm_y) - self.seq_len):
            X_seq.append(norm_y[i : i + self.seq_len])
            Y_seq.append(norm_y[i + self.seq_len])
            
        X_t = torch.tensor(np.array(X_seq), dtype=torch.float32)
        Y_t = torch.tensor(np.array(Y_seq), dtype=torch.float32)
        
        if self.model_type == 'lstm':
            X_t = X_t.unsqueeze(-1)
            Y_t = Y_t.unsqueeze(-1)
            self.model = VariationalLSTM(input_size=1, hidden_size=self.hidden_dim, num_layers=2, dropout_rate=0.2)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            for _ in range(self.epochs):
                self.model.train()
                optimizer.zero_grad()
                mean, logvar = self.model(X_t)
                logvar = torch.clamp(logvar, min=-5.0, max=5.0)
                loss = torch.mean(0.5 * torch.exp(-logvar) * (Y_t - mean)**2 + 0.5 * logvar)
                loss.backward()
                optimizer.step()
        else: # nbeats
            Y_t = Y_t.unsqueeze(-1)
            self.model = ProbabilisticNBeats(backcast_length=self.seq_len, forecast_length=1, num_blocks=2, hidden_dim=self.hidden_dim)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            for _ in range(self.epochs):
                self.model.train()
                optimizer.zero_grad()
                mean, std = self.model(X_t)
                loss = gaussian_nll_loss(mean, std, Y_t)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X, return_std=True, n_samples=30):
        steps = len(np.array(X).ravel())
        self.model.train() # Keep dropout active for MC sampling
        
        curr_win = np.tile(self.last_window, (n_samples, 1))
        f_means, f_stds = [], []
        
        for h in range(steps):
            x_in = torch.tensor(curr_win, dtype=torch.float32)
            with torch.no_grad():
                if self.model_type == 'lstm':
                    m_norm, logv_norm = self.model(x_in.unsqueeze(-1))
                    m_norm = m_norm.squeeze(-1)
                    v_norm = torch.exp(torch.clamp(logv_norm.squeeze(-1), min=-5.0, max=5.0))
                else:
                    m_norm, s_norm = self.model(x_in)
                    m_norm = m_norm.squeeze(-1)
                    v_norm = s_norm.squeeze(-1)**2
            
            m_val = m_norm.numpy() * self.y_std + self.y_mean
            v_val = v_norm.numpy() * (self.y_std**2)
            
            step_mean = np.mean(m_val)
            step_epi_var = np.var(m_val)
            step_ale_var = np.mean(v_val)
            step_std = np.sqrt(step_epi_var + step_ale_var)
            
            f_means.append(step_mean)
            f_stds.append(step_std)
            
            next_step = m_norm.numpy()
            curr_win = np.hstack([curr_win[:, 1:], next_step[:, np.newaxis]])
            
        if return_std:
            return np.array(f_means), np.array(f_stds)
        return np.array(f_means)


class ModelBenchmarkOrchestrator:
    """
    Orchestrates backtesting, probabilistic evaluation (CRPS, Coverage, Sharpness),
    and visual ranking across Step 1 (Classical) and Step 2 (Neural/Non-Parametric) UQ models.
    """
    def __init__(self, X_train, y_train, X_test, y_test, nominal_level=0.95):
        self.X_train = np.array(X_train).ravel()
        self.y_train = np.array(y_train, dtype=float).ravel()
        self.X_test = np.array(X_test).ravel()
        self.y_test = np.array(y_test, dtype=float).ravel()
        self.nominal_level = nominal_level
        self.models = {}
        self.results_df = None
        self.predictions_cache = {}

    def register_model(self, name: str, model_instance=None, step: str = "Step 1", model_type: str = "gaussian", **kwargs):
        """Registers a forecasting/UQ model for benchmarking."""
        if model_instance is None and model_type in ['lstm', 'nbeats']:
            model_instance = NeuralForecasterWrapper(model_type=model_type, **kwargs)
        elif model_type in ['lstm', 'nbeats'] and not hasattr(model_instance, 'fit'):
            model_instance = NeuralForecasterWrapper(model_type=model_type, **kwargs)
            
        self.models[name] = {
            'instance': model_instance,
            'step': step,
            'type': model_type
        }
        return self

    def run_benchmark(self) -> pd.DataFrame:
        """Executes training and evaluation across all registered models."""
        if not self.models:
            raise RuntimeError("[Orchestrator] No models registered for benchmarking.")
            
        print("="*80)
        print(f" EXECUTING PROBABILISTIC UQ BENCHMARK ({len(self.models)} MODELS REGISTERED)")
        print("="*80)
        
        records = []
        for name, meta in self.models.items():
            model = meta['instance']
            step = meta['step']
            print(f"[*] Benchmarking {name} ({step})...")
            
            try:
                model.fit(self.X_train, self.y_train)
                mean_pred, std_pred = model.predict(self.X_test, return_std=True)
                
                mean_pred = np.array(mean_pred, dtype=float).ravel()
                std_pred = np.array(std_pred, dtype=float).ravel()
                
                metrics = ProbabilisticMetrics.evaluate_all(mean_pred, std_pred, self.y_test, nominal_level=self.nominal_level)
                
                self.predictions_cache[name] = {
                    'mean': mean_pred,
                    'std': std_pred,
                    'step': step
                }
                
                row = {'Model_Name': name, 'Pipeline_Step': step}
                row.update(metrics)
                records.append(row)
                print(f"    -> CRPS: {metrics['CRPS']:.4f} | Coverage ({int(self.nominal_level*100)}%): {metrics[f'PICP_{int(self.nominal_level*100)}%']*100:.1f}% | RMSE: {metrics['RMSE']:.2f}")
            except Exception as e:
                print(f"    [Error] Benchmarking failed for {name}: {e}")
                
        self.results_df = pd.DataFrame(records).sort_values(by='CRPS', ascending=True).reset_index(drop=True)
        print("\n--- Benchmark Execution Complete ---")
        return self.results_df

    def plot_benchmark(self, save_path="benchmark_comparison.png"):
        """Generates a comparative visual plot of predictions, uncertainty bands, and ground truth."""
        if not self.predictions_cache:
            raise RuntimeError("[Orchestrator] Must run run_benchmark() before plotting.")
            
        n_models = len(self.predictions_cache)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 3.5 * n_models), sharex=True)
        if n_models == 1:
            axes = [axes]
            
        z_score = norm.ppf(1.0 - (1.0 - self.nominal_level) / 2.0)
        
        for ax, (name, pred_data) in zip(axes, self.predictions_cache.items()):
            mean_p = pred_data['mean']
            std_p = pred_data['std']
            step_label = pred_data['step']
            
            ax.plot(self.X_train, self.y_train, label="Training Observations", color='black', marker='o', markersize=4, linestyle='-')
            ax.plot(self.X_test, self.y_test, label="Ground Truth (Test)", color='darkgreen', marker='s', markersize=5, linestyle='--')
            
            ax.plot(self.X_test, mean_p, label=f"{name} Mean Forecast", color='blue', linewidth=2)
            ax.fill_between(self.X_test, mean_p - z_score * std_p, mean_p + z_score * std_p, color='blue', alpha=0.2, label=f"{int(self.nominal_level*100)}% Prediction Interval")
            
            ax.set_title(f"[{step_label}] {name} - Probabilistic Forecast Comparison", fontsize=12, fontweight='bold')
            ax.set_ylabel("Material Flow / Stock")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper left', fontsize=9)
            
        axes[-1].set_xlabel("Time (Horizon)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[*] Benchmark visual comparison saved to: {save_path}")