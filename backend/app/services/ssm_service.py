"""
ssm_service.py
--------------
State Space Model (SSM) — Kalman Filter for Dynamic Linear Regression.

Model equations (Dynamic Linear Model / DLM):
  Observation: y_t  = h_t @ theta_t + v_t,   v_t  ~ N(0, R)
  State evo:   theta_t = theta_{t-1} + w_t,   w_t  ~ N(0, Q)

  h_t   = [x_t_1, ..., x_t_d, 1.0]    (features + bias)
  theta = [w_1, …, w_d, bias]           (time-varying regression coefficients)
  Q = q_scale * I                        (process noise — how fast coefficients drift)
  R = r_scale                            (observation noise variance)

One KF is fitted independently per Y output. The final theta estimate (theta_T)
is used as a fixed linear model for prediction on new data.

Feature weights = the final regression coefficients (exact, not approximated).
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os, pickle, json


def _safe(val):
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    except Exception:
        return None


def _kalman_filter(X_scaled: np.ndarray, y_col: np.ndarray,
                   q_scale: float, r_scale: float) -> np.ndarray:
    """
    Run a Kalman filter over (X_scaled, y_col) and return the final
    state estimate theta_T — a vector of shape (n_features + 1,).
    """
    n, d    = X_scaled.shape
    n_state = d + 1            # d features + 1 bias term

    theta = np.zeros(n_state)
    P     = np.eye(n_state) * 10.0    # large initial uncertainty
    Q     = np.eye(n_state) * q_scale
    R     = float(r_scale)

    for t in range(n):
        # ── Predict ──────────────────────────────────────────────────────
        P_pred = P + Q                       # (n_state, n_state)

        # ── Update ───────────────────────────────────────────────────────
        h          = np.append(X_scaled[t], 1.0)          # (n_state,)
        S          = float(h @ P_pred @ h) + R             # scalar
        K          = (P_pred @ h) / S                      # Kalman gain (n_state,)
        innovation = float(y_col[t]) - float(h @ theta)
        theta      = theta + K * innovation
        P          = (np.eye(n_state) - np.outer(K, h)) @ P_pred

    return theta   # (n_state,) = [w_1..w_d, bias]


class SSMService:
    def __init__(self, model_dir: str = "./uploads/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def request_stop(self, run_id: int):
        pass   # not interruptible (fast sequential fitting)

    def train(
        self,
        df: pd.DataFrame,
        x_cols: list,
        y_cols: list,
        ssm_q_scale: float = 0.01,
        ssm_r_scale: float = 0.1,
        test_size: float = 0.2,
        model_run_id: int = 0,
        progress_callback=None,
        **kwargs,
    ) -> dict:
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

        # Fit one Kalman filter per Y output on training data
        Theta = np.zeros((len(y_cols), len(x_cols) + 1))   # (n_y, n_x + 1)
        for j in range(len(y_cols)):
            Theta[j] = _kalman_filter(X_tr, Y_tr[:, j], ssm_q_scale, ssm_r_scale)

        # Save
        save_path = os.path.join(self.model_dir, f"model_{model_run_id}")
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "model.pkl"), "wb") as f:
            pickle.dump({"Theta": Theta}, f)
        with open(os.path.join(save_path, "scalers.pkl"), "wb") as f:
            pickle.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, f)
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump({
                "x_cols": x_cols, "y_cols": y_cols,
                "input_dim": len(x_cols), "output_dim": len(y_cols),
                "ssm_q_scale": ssm_q_scale, "ssm_r_scale": ssm_r_scale,
                "model_type": "ssm",
            }, f)

        # Validation metrics
        X_val_aug  = np.hstack([X_val, np.ones((n_test, 1))])  # (n_test, n_x+1)
        Y_pred_s   = X_val_aug @ Theta.T                        # (n_test, n_y)

        Y_val_orig  = scaler_y.inverse_transform(Y_val)
        Y_pred_orig = scaler_y.inverse_transform(Y_pred_s)

        ss_res = np.sum((Y_val - Y_pred_s) ** 2)
        ss_tot = np.sum((Y_val - Y_val.mean(axis=0)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        per_y_r2, per_y_mae = {}, {}
        for i, col in enumerate(y_cols):
            a = Y_val_orig[:, i]
            p = Y_pred_orig[:, i]
            ss_r = np.sum((a - p) ** 2)
            ss_t = np.sum((a - a.mean()) ** 2)
            per_y_r2[col]  = _safe(round(float(1 - ss_r / (ss_t + 1e-8)), 4))
            per_y_mae[col] = _safe(round(float(np.mean(np.abs(a - p))), 4))

        return {
            "stopped_early":      False,
            "epochs_completed":   0,
            "train_loss_final":   None,
            "val_loss_final":     None,
            "r2_score":           _safe(round(r2, 4)),
            "per_y_r2":           per_y_r2,
            "per_y_mae":          per_y_mae,
            "train_loss_history": [],
            "val_loss_history":   [],
            "n_train_rows":       n_train,
            "n_test_rows":        n_test,
            "model_path":         save_path,
        }

    def _load(self, model_run_id: int):
        save_path = os.path.join(self.model_dir, f"model_{model_run_id}")
        with open(os.path.join(save_path, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(save_path, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)
        with open(os.path.join(save_path, "model.pkl"), "rb") as f:
            data = pickle.load(f)
        return data["Theta"], scalers["scaler_x"], scalers["scaler_y"], config

    def predict(self, model_run_id: int, input_dict: dict) -> dict:
        Theta, scaler_x, scaler_y, config = self._load(model_run_id)
        x_cols = config["x_cols"]
        y_cols = config["y_cols"]
        df_input = pd.DataFrame(input_dict)
        X = df_input[x_cols].values.astype(np.float32)
        X_scaled = scaler_x.transform(X)
        X_aug    = np.hstack([X_scaled, np.ones((len(X_scaled), 1))])
        Y_scaled = X_aug @ Theta.T
        Y_pred   = scaler_y.inverse_transform(Y_scaled)
        return {col: Y_pred[:, i].tolist() for i, col in enumerate(y_cols)}

    def predict_on_test_split(
        self,
        model_run_id: int,
        df: pd.DataFrame,
        x_cols: list,
        y_cols: list,
        test_size: float = 0.2,
        sample_size: int = 300,
    ) -> dict:
        Theta, scaler_x, scaler_y, config = self._load(model_run_id)
        x_cols = config["x_cols"]
        y_cols = config["y_cols"]

        X_all = df[x_cols].values.astype(np.float32)
        Y_all = df[y_cols].values.astype(np.float32)
        X_scaled = scaler_x.transform(X_all)
        Y_scaled = scaler_y.transform(Y_all)

        _, X_test, _, Y_test_s = train_test_split(X_scaled, Y_scaled, test_size=test_size, random_state=42)
        _, _, _, Y_test_orig   = train_test_split(X_all, Y_all, test_size=test_size, random_state=42)
        n_test = len(X_test)

        X_aug    = np.hstack([X_test, np.ones((n_test, 1))])
        Y_pred_s = X_aug @ Theta.T
        Y_pred   = scaler_y.inverse_transform(Y_pred_s)
        indices  = np.linspace(0, n_test - 1, min(sample_size, n_test), dtype=int)

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
        """
        For SSM, the Kalman-filter coefficients ARE the feature weights.
        Theta[j, i] = dY_j_scaled / dX_i_scaled  (linear model coefficient)
        Convert to original scale: weight_orig = Theta[j,i] * sigma_y[j] / sigma_x[i]
        """
        Theta, scaler_x, scaler_y, config = self._load(model_run_id)
        x_cols  = config["x_cols"]
        y_cols  = config["y_cols"]
        sigma_x = scaler_x.scale_   # (n_x,)
        sigma_y = scaler_y.scale_   # (n_y,)

        # Theta shape: (n_y, n_x + 1) — last column is bias, ignore for weights
        W_scaled = Theta[:, :len(x_cols)]           # (n_y, n_x)
        W_orig   = W_scaled * sigma_y[:, None] / sigma_x[None, :]  # (n_y, n_x)

        weights, std_weights, importance = {}, {}, {}
        for xc in x_cols:
            weights[xc] = {}; std_weights[xc] = {}; importance[xc] = {}

        for j, yc in enumerate(y_cols):
            col_orig = W_orig[j, :]
            total    = np.sum(np.abs(col_orig)) + 1e-8
            for i, xc in enumerate(x_cols):
                weights[xc][yc]     = _safe(float(col_orig[i]))
                std_weights[xc][yc] = _safe(float(W_scaled[j, i]))
                importance[xc][yc]  = _safe(float(np.abs(col_orig[i]) / total * 100))

        return {
            "x_cols": x_cols, "y_cols": y_cols,
            "weights": weights, "std_weights": std_weights, "importance": importance,
        }
