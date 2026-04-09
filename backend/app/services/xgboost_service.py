"""
xgboost_service.py
------------------
XGBoost multi-output regressor for Multi-X → Multi-Y regression.
Uses sklearn MultiOutputRegressor wrapping XGBRegressor.
Feature importance is sourced from XGBoost's native gain-based scores.
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import os, pickle, json

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def _safe(val):
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    except Exception:
        return None


class XGBoostService:
    def __init__(self, model_dir: str = "./uploads/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def request_stop(self, run_id: int):
        pass   # XGBoost training is not interruptible

    def train(
        self,
        df: pd.DataFrame,
        x_cols: list,
        y_cols: list,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        test_size: float = 0.2,
        model_run_id: int = 0,
        progress_callback=None,
        **kwargs,
    ) -> dict:
        if not XGBOOST_AVAILABLE:
            raise RuntimeError(
                "xgboost is not installed. Run: pip install xgboost"
            )

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

        base = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model = MultiOutputRegressor(base)
        model.fit(X_tr, Y_tr)

        # Save
        save_path = os.path.join(self.model_dir, f"model_{model_run_id}")
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(save_path, "scalers.pkl"), "wb") as f:
            pickle.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, f)
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump({
                "x_cols": x_cols, "y_cols": y_cols,
                "input_dim": len(x_cols), "output_dim": len(y_cols),
                "n_estimators": n_estimators, "max_depth": max_depth,
                "learning_rate": learning_rate,
                "model_type": "xgboost",
            }, f)

        # Metrics on validation set (original scale)
        Y_pred_s = model.predict(X_val)
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
            "epochs_completed":   n_estimators,   # trees = "epochs" for XGBoost
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

    # ── internal loader ────────────────────────────────────────────────────────

    def _load(self, model_run_id: int):
        save_path = os.path.join(self.model_dir, f"model_{model_run_id}")
        with open(os.path.join(save_path, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(save_path, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)
        with open(os.path.join(save_path, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        return model, scalers["scaler_x"], scalers["scaler_y"], config

    def predict(self, model_run_id: int, input_dict: dict) -> dict:
        model, scaler_x, scaler_y, config = self._load(model_run_id)
        x_cols = config["x_cols"]
        df_input = pd.DataFrame(input_dict)
        X = df_input[x_cols].values.astype(np.float32)
        X_scaled = scaler_x.transform(X)
        Y_scaled = model.predict(X_scaled)
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

        Y_pred_s    = model.predict(X_test)
        Y_pred      = scaler_y.inverse_transform(Y_pred_s)
        indices     = np.linspace(0, n_test - 1, min(sample_size, n_test), dtype=int)

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

        sigma_x = scaler_x.scale_
        sigma_y = scaler_y.scale_

        # Baseline in scaled space (zeros = mean)
        if baseline_x and all(c in baseline_x for c in x_cols):
            x_orig   = np.array([[float(baseline_x[c]) for c in x_cols]], dtype=np.float32)
            x_base   = scaler_x.transform(x_orig)[0]
        else:
            x_base = np.zeros(len(x_cols), dtype=np.float32)

        y_base_s = model.predict(x_base.reshape(1, -1))[0]  # (n_y,)

        weights, std_weights, importance = {}, {}, {}
        for xc in x_cols:
            weights[xc] = {}; std_weights[xc] = {}; importance[xc] = {}

        # Native XGBoost feature importance per Y estimator (gain-based)
        xgb_imp = {}  # xgb_imp[yi][xi]
        for j, yc in enumerate(y_cols):
            est = model.estimators_[j]
            fi  = est.feature_importances_    # shape (n_x,), sums to 1
            xgb_imp[j] = fi

        # Perturbation for directional weights
        for i, xc in enumerate(x_cols):
            x_pert = x_base.copy()
            x_pert[i] = x_base[i] + 1.0    # +1 std in scaled space
            y_pert_s = model.predict(x_pert.reshape(1, -1))[0]  # (n_y,)
            delta_s  = y_pert_s - y_base_s  # (n_y,) in scaled space
            delta_o  = delta_s * sigma_y    # convert to original scale

            for j, yc in enumerate(y_cols):
                std_weights[xc][yc] = _safe(float(delta_s[j]))
                weights[xc][yc]     = _safe(float(delta_o[j]))

        # Importance from XGBoost native (per Y, normalise to %)
        for j, yc in enumerate(y_cols):
            fi  = xgb_imp[j]
            tot = fi.sum() + 1e-8
            for i, xc in enumerate(x_cols):
                importance[xc][yc] = _safe(float(fi[i] / tot * 100))

        return {
            "x_cols": x_cols, "y_cols": y_cols,
            "weights": weights, "std_weights": std_weights, "importance": importance,
        }
