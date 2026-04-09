"""
lstm_service.py
---------------
LSTM (Long Short-Term Memory) model for Multi-X → Multi-Y regression.
Treats each row as a single-timestep sequence (seq_len = 1).
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os, pickle, json, threading

_stop_events: dict[int, threading.Event] = {}


def _safe(val):
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    except Exception:
        return None


class LSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.bn   = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, 1, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]        # last timestep → (batch, hidden_dim)
        out = self.bn(out)
        return self.head(out)      # (batch, output_dim)


class LSTMService:
    def __init__(self, model_dir: str = "./uploads/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def request_stop(self, run_id: int):
        if run_id in _stop_events:
            _stop_events[run_id].set()

    def train(
        self,
        df: pd.DataFrame,
        x_cols: list,
        y_cols: list,
        epochs: int = 100,
        hidden_dim: int = 64,
        num_layers: int = 2,
        test_size: float = 0.2,
        model_run_id: int = 0,
        progress_callback=None,
        **kwargs,
    ) -> dict:
        stop_event = threading.Event()
        _stop_events[model_run_id] = stop_event

        try:
            X = df[x_cols].values.astype(np.float32)
            Y = df[y_cols].values.astype(np.float32)

            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_x.fit_transform(X)
            Y_scaled = scaler_y.fit_transform(Y)

            X_tr, X_val, Y_tr, Y_val = train_test_split(
                X_scaled, Y_scaled, test_size=test_size, random_state=42
            )
            n_train, n_test = len(X_tr), len(X_val)

            # Add seq_len=1 dimension for LSTM
            X_tr_t  = torch.tensor(X_tr[:, None, :]).to(self.device)
            Y_tr_t  = torch.tensor(Y_tr).to(self.device)
            X_val_t = torch.tensor(X_val[:, None, :]).to(self.device)
            Y_val_t = torch.tensor(Y_val).to(self.device)

            loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=64, shuffle=True)

            model = LSTMNet(len(x_cols), len(y_cols), hidden_dim, num_layers).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            train_losses, val_losses = [], []
            stopped_early = False

            for epoch in range(epochs):
                if stop_event.is_set():
                    stopped_early = True
                    break

                model.train()
                epoch_loss, batches = 0.0, 0
                for xb, yb in loader:
                    if stop_event.is_set():
                        stopped_early = True
                        break
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batches += 1

                if stopped_early:
                    break

                avg_loss = epoch_loss / max(batches, 1)
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val_t), Y_val_t).item()

                train_losses.append(round(avg_loss, 6))
                val_losses.append(round(val_loss, 6))

                if progress_callback and (epoch + 1) % 5 == 0:
                    progress_callback(epoch + 1, epochs, avg_loss, val_loss)

            # Save
            save_path = os.path.join(self.model_dir, f"model_{model_run_id}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            with open(os.path.join(save_path, "scalers.pkl"), "wb") as f:
                pickle.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, f)
            with open(os.path.join(save_path, "config.json"), "w") as f:
                json.dump({
                    "x_cols": x_cols, "y_cols": y_cols,
                    "input_dim": len(x_cols), "output_dim": len(y_cols),
                    "hidden_dim": hidden_dim, "num_layers": num_layers,
                    "model_type": "lstm",
                }, f)

            epochs_done = len(train_losses)
            if epochs_done == 0:
                return {
                    "stopped_early": True, "epochs_completed": 0,
                    "train_loss_final": None, "val_loss_final": None,
                    "r2_score": None, "per_y_r2": {}, "per_y_mae": {},
                    "train_loss_history": [], "val_loss_history": [],
                    "n_train_rows": n_train, "n_test_rows": n_test,
                    "model_path": save_path,
                }

            model.eval()
            with torch.no_grad():
                y_pred = model(X_val_t).cpu().numpy()
                y_true = Y_val_t.cpu().numpy()

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-8))

            Y_val_orig  = scaler_y.inverse_transform(y_true)
            Y_pred_orig = scaler_y.inverse_transform(y_pred)
            per_y_r2, per_y_mae = {}, {}
            for i, col in enumerate(y_cols):
                a = Y_val_orig[:, i]
                p = Y_pred_orig[:, i]
                ss_r = np.sum((a - p) ** 2)
                ss_t = np.sum((a - a.mean()) ** 2)
                per_y_r2[col]  = _safe(round(float(1 - ss_r / (ss_t + 1e-8)), 4))
                per_y_mae[col] = _safe(round(float(np.mean(np.abs(a - p))), 4))

            return {
                "stopped_early":      stopped_early,
                "epochs_completed":   epochs_done,
                "train_loss_final":   _safe(train_losses[-1]),
                "val_loss_final":     _safe(val_losses[-1]),
                "r2_score":           _safe(round(r2, 4)),
                "per_y_r2":           per_y_r2,
                "per_y_mae":          per_y_mae,
                "train_loss_history": [_safe(v) for v in train_losses],
                "val_loss_history":   [_safe(v) for v in val_losses],
                "n_train_rows":       n_train,
                "n_test_rows":        n_test,
                "model_path":         save_path,
            }
        finally:
            _stop_events.pop(model_run_id, None)

    # ── internal loader ────────────────────────────────────────────────────────

    def _load(self, model_run_id: int):
        save_path = os.path.join(self.model_dir, f"model_{model_run_id}")
        with open(os.path.join(save_path, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(save_path, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)
        model = LSTMNet(
            config["input_dim"], config["output_dim"],
            config["hidden_dim"], config.get("num_layers", 2),
        ).to(self.device)
        model.load_state_dict(
            torch.load(os.path.join(save_path, "model.pt"), map_location=self.device)
        )
        model.eval()
        return model, scalers["scaler_x"], scalers["scaler_y"], config

    def predict(self, model_run_id: int, input_dict: dict) -> dict:
        model, scaler_x, scaler_y, config = self._load(model_run_id)
        x_cols = config["x_cols"]
        df_input = pd.DataFrame(input_dict)
        X = df_input[x_cols].values.astype(np.float32)
        X_scaled = scaler_x.transform(X)
        X_t = torch.tensor(X_scaled[:, None, :]).to(self.device)
        with torch.no_grad():
            Y_scaled = model(X_t).cpu().numpy()
        Y_pred = scaler_y.inverse_transform(Y_scaled)
        return {col: Y_pred[:, i].tolist() for i, col in enumerate(config["y_cols"])}

    def predict_on_test_split(
        self,
        model_run_id: int,
        df: pd.DataFrame,
        x_cols: list,
        y_cols: list,
        test_size: float = 0.2,
        sample_size: int = 300,
    ) -> dict:
        model, scaler_x, scaler_y, config = self._load(model_run_id)
        x_cols = config["x_cols"]
        y_cols = config["y_cols"]

        X_all = df[x_cols].values.astype(np.float32)
        Y_all = df[y_cols].values.astype(np.float32)
        X_scaled = scaler_x.transform(X_all)
        Y_scaled = scaler_y.transform(Y_all)

        _, X_test, _, Y_test_s = train_test_split(X_scaled, Y_scaled, test_size=test_size, random_state=42)
        _, _, _, Y_test_orig   = train_test_split(X_all, Y_all, test_size=test_size, random_state=42)
        n_test = len(X_test)

        X_t = torch.tensor(X_test[:, None, :]).to(self.device)
        with torch.no_grad():
            Y_pred_s = model(X_t).cpu().numpy()
        Y_pred = scaler_y.inverse_transform(Y_pred_s)

        indices = np.linspace(0, n_test - 1, min(sample_size, n_test), dtype=int)
        per_column = {}
        for i, col in enumerate(y_cols):
            actual    = Y_test_orig[:, i]
            predicted = Y_pred[:, i]
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            per_column[col] = {
                "r2":        _safe(round(float(1 - ss_res / (ss_tot + 1e-8)), 4)),
                "mae":       _safe(round(float(np.mean(np.abs(actual - predicted))), 4)),
                "actual":    [_safe(v) for v in actual[indices].tolist()],
                "predicted": [_safe(v) for v in predicted[indices].tolist()],
            }
        return {
            "n_test_rows": n_test, "sample_size": len(indices),
            "x_cols_used": x_cols, "y_cols_used": y_cols, "columns": per_column,
        }

    def compute_feature_weights(self, model_run_id: int, baseline_x: dict = None) -> dict:
        model, scaler_x, scaler_y, config = self._load(model_run_id)
        x_cols = config["x_cols"]
        y_cols = config["y_cols"]

        if baseline_x and all(c in baseline_x for c in x_cols):
            x_orig   = np.array([[float(baseline_x[c]) for c in x_cols]], dtype=np.float32)
            x_scaled = scaler_x.transform(x_orig)
        else:
            x_scaled = np.zeros((1, len(x_cols)), dtype=np.float32)

        # Gradient-based Jacobian via autograd
        x_t = torch.tensor(x_scaled[:, None, :], requires_grad=True).to(self.device)

        jacobian_rows = []
        for j in range(len(y_cols)):
            if x_t.grad is not None:
                x_t.grad.zero_()
            y_pred = model(x_t)
            y_pred[0, j].backward(retain_graph=(j < len(y_cols) - 1))
            grad = x_t.grad.detach().cpu().numpy()[0, 0, :].copy()  # (n_x,)
            jacobian_rows.append(grad)

        jacobian_scaled = np.array(jacobian_rows)   # (n_y, n_x)
        sigma_x = scaler_x.scale_
        sigma_y = scaler_y.scale_
        jacobian_orig = jacobian_scaled * sigma_y[:, None] / sigma_x[None, :]

        weights, std_weights, importance = {}, {}, {}
        for xc in x_cols:
            weights[xc] = {}; std_weights[xc] = {}; importance[xc] = {}

        for j, yc in enumerate(y_cols):
            col_orig = jacobian_orig[j, :]
            total    = np.sum(np.abs(col_orig)) + 1e-8
            for i, xc in enumerate(x_cols):
                weights[xc][yc]     = _safe(float(col_orig[i]))
                std_weights[xc][yc] = _safe(float(jacobian_scaled[j, i]))
                importance[xc][yc]  = _safe(float(np.abs(col_orig[i]) / total * 100))

        return {
            "x_cols": x_cols, "y_cols": y_cols,
            "weights": weights, "std_weights": std_weights, "importance": importance,
        }
