"""
autoencoder.py
--------------
The core ML logic: Denoising AutoEncoder (DAE) for Multi-X → Multi-Y mapping.

How a Denoising AutoEncoder works:
  1. Take clean input data (X features)
  2. Add random noise to it  →  corrupted input
  3. Pass through Encoder  →  compressed representation (bottleneck)
  4. Pass through Decoder  →  reconstructed / predicted output (Y targets)
  5. Train to minimize the difference between predicted Y and actual Y
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
import json
import threading


def _safe(val):
    """Convert nan/inf floats to None so JSON serialization never fails."""
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    except Exception:
        return None

# Module-level stop flags — one threading.Event per active training run.
# request_stop() sets the event; the training loop checks it each epoch.
_stop_events: dict[int, threading.Event] = {}


# ─── Model Architecture ───────────────────────────────────────────────────────

class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        bottleneck = max(hidden_dim // 2, 8)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─── Training Service ─────────────────────────────────────────────────────────

class DAEService:
    def __init__(self, model_dir: str = "./uploads/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def request_stop(self, run_id: int):
        """Signal the training loop for this run to stop after the current epoch."""
        if run_id in _stop_events:
            _stop_events[run_id].set()

    def _add_noise(self, x: torch.Tensor, noise_factor: float) -> torch.Tensor:
        return x + torch.randn_like(x) * noise_factor

    def train(
        self,
        df: pd.DataFrame,
        x_cols: list,
        y_cols: list,
        noise_factor: float = 0.1,
        epochs: int = 100,
        hidden_dim: int = 64,
        test_size: float = 0.2,
        model_run_id: int = 0,
        progress_callback=None,   # called every 5 epochs: callback(epoch, total, train_loss, val_loss)
    ) -> dict:
        """
        Train the DAE. Checks the stop flag after each epoch.
        If stopped early, saves the model at the current state and returns.
        """
        # Register a stop event for this run
        stop_event = threading.Event()
        _stop_events[model_run_id] = stop_event

        try:
            # 1. Prepare data
            X = df[x_cols].values.astype(np.float32)
            Y = df[y_cols].values.astype(np.float32)

            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_x.fit_transform(X)
            Y_scaled = scaler_y.fit_transform(Y)

            X_train, X_val, Y_train, Y_val = train_test_split(
                X_scaled, Y_scaled, test_size=test_size, random_state=42
            )
            n_train, n_test = len(X_train), len(X_val)

            X_train_t = torch.tensor(X_train).to(self.device)
            Y_train_t = torch.tensor(Y_train).to(self.device)
            X_val_t   = torch.tensor(X_val).to(self.device)
            Y_val_t   = torch.tensor(Y_val).to(self.device)

            loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=32, shuffle=True)

            model = DenoisingAutoEncoder(
                input_dim=len(x_cols),
                output_dim=len(y_cols),
                hidden_dim=hidden_dim,
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            train_losses, val_losses = [], []
            stopped_early = False

            # 2. Training loop — checks stop flag every batch for instant stop
            for epoch in range(epochs):
                if stop_event.is_set():
                    stopped_early = True
                    break

                model.train()
                epoch_loss = 0.0
                batches_done = 0

                for x_batch, y_batch in loader:
                    # Check flag every single batch — stops within milliseconds
                    if stop_event.is_set():
                        stopped_early = True
                        break

                    x_noisy = self._add_noise(x_batch, noise_factor)
                    optimizer.zero_grad()
                    loss = criterion(model(x_noisy), y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batches_done += 1

                if stopped_early:
                    break

                avg_train_loss = epoch_loss / max(batches_done, 1)

                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val_t), Y_val_t).item()

                train_losses.append(round(avg_train_loss, 6))
                val_losses.append(round(val_loss, 6))

                # Report progress every 5 epochs
                if progress_callback and (epoch + 1) % 5 == 0:
                    progress_callback(epoch + 1, epochs, avg_train_loss, val_loss)

            # 3. Save model (even if stopped early — partial model is still usable)
            save_path = os.path.join(self.model_dir, f"model_{model_run_id}")
            os.makedirs(save_path, exist_ok=True)

            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            with open(os.path.join(save_path, "scalers.pkl"), "wb") as f:
                pickle.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, f)
            with open(os.path.join(save_path, "config.json"), "w") as f:
                json.dump({
                    "x_cols": x_cols, "y_cols": y_cols,
                    "hidden_dim": hidden_dim,
                    "input_dim": len(x_cols), "output_dim": len(y_cols),
                }, f)

            # 4. Compute R² on validation set
            epochs_done = len(train_losses)
            if epochs_done == 0:
                return {
                    "stopped_early": True, "epochs_completed": 0,
                    "train_loss_final": None, "val_loss_final": None,
                    "r2_score": None, "train_loss_history": [],
                    "val_loss_history": [], "model_path": save_path,
                }

            model.eval()
            with torch.no_grad():
                y_pred_np = model(X_val_t).cpu().numpy()
                y_true_np = Y_val_t.cpu().numpy()

            ss_res = np.sum((y_true_np - y_pred_np) ** 2)
            ss_tot = np.sum((y_true_np - y_true_np.mean(axis=0)) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-8))

            # Per-Y R² and MAE on original (unscaled) values for interpretability
            Y_val_orig  = scaler_y.inverse_transform(y_true_np)
            Y_pred_orig = scaler_y.inverse_transform(y_pred_np)
            per_y_r2, per_y_mae = {}, {}
            for i, col in enumerate(y_cols):
                a = Y_val_orig[:, i]
                p = Y_pred_orig[:, i]
                ss_r = np.sum((a - p) ** 2)
                ss_t = np.sum((a - a.mean()) ** 2)
                per_y_r2[col]  = _safe(round(float(1 - ss_r / (ss_t + 1e-8)), 4))
                per_y_mae[col] = _safe(round(float(np.mean(np.abs(a - p))), 4))

            return {
                "stopped_early":       stopped_early,
                "epochs_completed":    epochs_done,
                "train_loss_final":    _safe(train_losses[-1]),
                "val_loss_final":      _safe(val_losses[-1]),
                "r2_score":            _safe(round(r2, 4)),
                "per_y_r2":            per_y_r2,
                "per_y_mae":           per_y_mae,
                "train_loss_history":  [_safe(v) for v in train_losses],
                "val_loss_history":    [_safe(v) for v in val_losses],
                "n_train_rows":        n_train,
                "n_test_rows":         n_test,
                "model_path":          save_path,
            }

        finally:
            # Always clean up the stop flag
            _stop_events.pop(model_run_id, None)

    def predict(self, model_run_id: int, input_dict: dict) -> dict:
        save_path = os.path.join(self.model_dir, f"model_{model_run_id}")

        with open(os.path.join(save_path, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(save_path, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)

        scaler_x = scalers["scaler_x"]
        scaler_y = scalers["scaler_y"]

        model = DenoisingAutoEncoder(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
        ).to(self.device)
        model.load_state_dict(
            torch.load(os.path.join(save_path, "model.pt"), map_location=self.device)
        )
        model.eval()

        x_cols = config["x_cols"]
        df_input = pd.DataFrame(input_dict)
        X = df_input[x_cols].values.astype(np.float32)
        X_scaled = scaler_x.transform(X)
        X_tensor = torch.tensor(X_scaled).to(self.device)

        with torch.no_grad():
            Y_scaled_pred = model(X_tensor).cpu().numpy()

        Y_pred = scaler_y.inverse_transform(Y_scaled_pred)
        y_cols = config["y_cols"]
        return {col: Y_pred[:, i].tolist() for i, col in enumerate(y_cols)}

    def compute_feature_weights(self, model_run_id: int, baseline_x: dict = None) -> dict:
        """
        Compute the Jacobian matrix ∂Y/∂X at the dataset mean (or a supplied baseline).
        Uses PyTorch autograd — one backward pass per Y output.

        Returns:
          weights     — original-scale sensitivity: dY_orig/dX_orig
          std_weights — standardised sensitivity:   dY_std/dX_std  (unitless, comparable across columns)
          importance  — per-Y normalised |weight| as a percentage (sums to 100 per Y column)
        """
        save_path = os.path.join(self.model_dir, f"model_{model_run_id}")
        with open(os.path.join(save_path, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(save_path, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)

        scaler_x = scalers["scaler_x"]
        scaler_y = scalers["scaler_y"]
        x_cols   = config["x_cols"]
        y_cols   = config["y_cols"]

        model = DenoisingAutoEncoder(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
        ).to(self.device)
        model.load_state_dict(
            torch.load(os.path.join(save_path, "model.pt"), map_location=self.device)
        )
        model.eval()

        # Evaluation point: provided baseline → scale it; otherwise use mean (0 in scaled space)
        if baseline_x and all(c in baseline_x for c in x_cols):
            x_orig   = np.array([[float(baseline_x[c]) for c in x_cols]], dtype=np.float32)
            x_scaled = scaler_x.transform(x_orig)
        else:
            x_scaled = np.zeros((1, len(x_cols)), dtype=np.float32)

        x_tensor = torch.tensor(x_scaled, requires_grad=True).to(self.device)

        # One backward pass per Y output to build the Jacobian
        jacobian_rows = []
        for j in range(len(y_cols)):
            if x_tensor.grad is not None:
                x_tensor.grad.zero_()
            y_pred = model(x_tensor)
            y_pred[0, j].backward(retain_graph=(j < len(y_cols) - 1))
            jacobian_rows.append(x_tensor.grad.detach().cpu().numpy()[0].copy())

        jacobian_scaled = np.array(jacobian_rows)           # (n_y, n_x) — standardised units

        # Chain-rule conversion to original units: dY_orig/dX_orig = J_scaled × (σ_y / σ_x)
        sigma_x = scaler_x.scale_                           # (n_x,)
        sigma_y = scaler_y.scale_                           # (n_y,)
        jacobian_orig = jacobian_scaled * sigma_y[:, None] / sigma_x[None, :]  # (n_y, n_x)

        # Build output dicts keyed [x_col][y_col]
        weights, std_weights, importance = {}, {}, {}
        for xc in x_cols:
            weights[xc]     = {}
            std_weights[xc] = {}
            importance[xc]  = {}

        for j, yc in enumerate(y_cols):
            col_orig = jacobian_orig[j, :]
            total    = np.sum(np.abs(col_orig)) + 1e-8
            for i, xc in enumerate(x_cols):
                weights[xc][yc]     = _safe(float(col_orig[i]))
                std_weights[xc][yc] = _safe(float(jacobian_scaled[j, i]))
                importance[xc][yc]  = _safe(float(np.abs(col_orig[i]) / total * 100))

        return {
            "x_cols":      x_cols,
            "y_cols":      y_cols,
            "weights":     weights,
            "std_weights": std_weights,
            "importance":  importance,
        }

    def predict_on_test_split(
        self,
        model_run_id: int,
        df: pd.DataFrame,
        x_cols: list,
        y_cols: list,
        test_size: float = 0.2,
        sample_size: int = 300,
    ) -> dict:
        """
        Reproduce the exact same train/test split used during training,
        run predictions on the test rows, and return predicted vs actual
        for every Y column.

        Uses config.json as the source of truth for x_cols/y_cols/architecture —
        this guarantees the columns and scaler exactly match what was trained,
        regardless of what is stored in the DB.

        Returns up to `sample_size` rows for chart rendering.
        """
        save_path = os.path.join(self.model_dir, f"model_{model_run_id}")

        model_pt = os.path.join(save_path, "model.pt")
        if not os.path.exists(model_pt):
            raise FileNotFoundError(
                "No saved model weights found for this run. "
                "The model was likely stopped before any epoch completed. "
                "Please train again with more epochs."
            )

        with open(os.path.join(save_path, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(save_path, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)

        # ── Use config.json as the ground truth ───────────────────────────────
        # The DB (run.x_columns / run.y_columns) may diverge from the actual
        # saved model if, for example, a cleaned dataset had inherited wrong
        # column lists. config.json is written by the training loop itself and
        # is always in sync with the scalers and model weights.
        x_cols    = config["x_cols"]
        y_cols    = config["y_cols"]

        # Verify all required columns exist in the loaded dataframe
        missing_x = [c for c in x_cols if c not in df.columns]
        missing_y = [c for c in y_cols if c not in df.columns]
        if missing_x or missing_y:
            raise ValueError(
                f"Dataset is missing columns needed by the model — "
                f"X: {missing_x or 'ok'}, Y: {missing_y or 'ok'}. "
                "Make sure you are predicting on the same dataset used for training."
            )

        scaler_x = scalers["scaler_x"]
        scaler_y = scalers["scaler_y"]

        model = DenoisingAutoEncoder(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
        ).to(self.device)
        model.load_state_dict(
            torch.load(os.path.join(save_path, "model.pt"), map_location=self.device)
        )
        model.eval()

        # Reproduce the exact same split (same random_state=42 as training)
        X_all = df[x_cols].values.astype(np.float32)
        Y_all = df[y_cols].values.astype(np.float32)

        X_scaled = scaler_x.transform(X_all)
        Y_scaled = scaler_y.transform(Y_all)

        _, X_test_scaled, _, Y_test_scaled = train_test_split(
            X_scaled, Y_scaled, test_size=test_size, random_state=42
        )
        _, _, _, Y_test_actual = train_test_split(
            X_all, Y_all, test_size=test_size, random_state=42
        )

        n_test = len(X_test_scaled)

        # Run predictions on all test rows
        X_tensor = torch.tensor(X_test_scaled).to(self.device)
        with torch.no_grad():
            Y_pred_scaled = model(X_tensor).cpu().numpy()
        Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

        # Sample evenly for chart rendering
        indices = np.linspace(0, n_test - 1, min(sample_size, n_test), dtype=int)

        per_column = {}
        for i, col in enumerate(y_cols):
            actual_all    = Y_test_actual[:, i]
            predicted_all = Y_pred[:, i]

            # R² on full test set
            ss_res = np.sum((actual_all - predicted_all) ** 2)
            ss_tot = np.sum((actual_all - actual_all.mean()) ** 2)
            r2  = float(1 - ss_res / (ss_tot + 1e-8))
            mae = float(np.mean(np.abs(actual_all - predicted_all)))

            per_column[col] = {
                "r2":        _safe(round(r2, 4)),
                "mae":       _safe(round(mae, 4)),
                "actual":    [_safe(v) for v in actual_all[indices].tolist()],
                "predicted": [_safe(v) for v in predicted_all[indices].tolist()],
            }

        return {
            "n_test_rows":  n_test,
            "sample_size":  len(indices),
            "x_cols_used":  x_cols,   # ground-truth columns from config.json
            "y_cols_used":  y_cols,   # ground-truth columns from config.json
            "columns":      per_column,
        }
