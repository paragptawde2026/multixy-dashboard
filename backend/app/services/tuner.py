"""
tuner.py
--------
Hyperparameter tuning for the DAE model.
Supports Random Search and Grid Search over a pre-defined parameter space.

Speed design:
  • Data is sampled, scaled, split, and converted to tensors ONCE before the
    trial loop — trials only differ in model architecture and training params.
  • A small row subsample (TUNE_MAX_ROWS) and tight epoch cap (TUNE_EPOCHS_CAP)
    keep each trial to a few seconds while preserving relative R² ranking.

Each trial reports:
  - Overall R² across all Y columns
  - Per-Y column R² scores
  - Focus score (mean R² of selected Y cols when focus_y_cols is set)

After all trials:
  - Best global hyperparameters (by focus score when focus_y_cols given)
  - Best per-Y hyperparameters for columns below the per-Y threshold
  - List of Y columns still below threshold even with best params
"""

import math
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from app.services.autoencoder import DenoisingAutoEncoder


# ─── Search Space ─────────────────────────────────────────────────────────────

PARAM_SPACE = {
    "noise_factor": [0.05, 0.10, 0.15, 0.20, 0.30],
    "hidden_dim":   [32, 64, 128, 256],
    "epochs":       [50, 100, 150, 200],
}

# Each trial is capped at this many epochs — relative ranking is stable at 20
TUNE_EPOCHS_CAP = 20

# Large batch = fewer steps per epoch = fast
TUNE_BATCH_SIZE = 512

# Subsample size — ranking stays consistent; full 44k rows is wasteful for search
TUNE_MAX_ROWS = 5_000


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe(val):
    try:
        if val is None or math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    except Exception:
        return None


def _r2_per_column(y_true: np.ndarray, y_pred: np.ndarray, y_cols: list) -> dict:
    """Compute per-column R² on unscaled arrays."""
    out = {}
    for i, col in enumerate(y_cols):
        a = y_true[:, i]
        p = y_pred[:, i]
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        out[col] = _safe(round(float(1 - ss_res / (ss_tot + 1e-8)), 4))
    return out


# ─── Single Trial (receives pre-built tensors — no data prep overhead) ─────────

def _train_trial(
    *,
    X_tr_t:    torch.Tensor,   # training X on device
    Y_tr_t:    torch.Tensor,   # training Y (scaled) on device
    X_val_t:   torch.Tensor,   # validation X on device
    Y_val_t:   torch.Tensor,   # validation Y (scaled) on device  — for val loss
    Y_val_raw: np.ndarray,     # validation Y (original scale)    — for R²
    scaler_y:  StandardScaler,
    x_dim:     int,
    y_cols:    list,
    params:    dict,
    device:    torch.device,
):
    """
    Train one mini-model. All tensors are pre-built; this function only
    handles model construction, the training loop, and metric computation.
    """
    noise_factor = params["noise_factor"]
    hidden_dim   = params["hidden_dim"]
    epochs       = min(params["epochs"], TUNE_EPOCHS_CAP)

    model = DenoisingAutoEncoder(
        input_dim=x_dim,
        output_dim=len(y_cols),
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader    = DataLoader(
        TensorDataset(X_tr_t, Y_tr_t),
        batch_size=TUNE_BATCH_SIZE,
        shuffle=True,
    )

    train_loss = val_loss = None
    for _ in range(epochs):
        model.train()
        ep_loss, n = 0.0, 0
        for xb, yb in loader:
            xn = xb + torch.randn_like(xb) * noise_factor
            optimizer.zero_grad()
            loss = criterion(model(xn), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n += 1
        train_loss = ep_loss / max(n, 1)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), Y_val_t).item()

    # Metrics on original scale
    model.eval()
    with torch.no_grad():
        Y_pred_sc = model(X_val_t).cpu().numpy()
    Y_pred = scaler_y.inverse_transform(Y_pred_sc)

    per_y_r2 = _r2_per_column(Y_val_raw, Y_pred, y_cols)
    r2_vals  = [v for v in per_y_r2.values() if v is not None]
    overall  = _safe(round(float(np.mean(r2_vals)), 4)) if r2_vals else None

    return {
        "params":      params,
        "overall_r2":  overall,
        "per_y_r2":    per_y_r2,
        "train_loss":  _safe(round(train_loss, 6)) if train_loss is not None else None,
        "val_loss":    _safe(round(val_loss,   6)) if val_loss   is not None else None,
        "epochs_used": epochs,
    }


# ─── Main Tuning Entry Point ───────────────────────────────────────────────────

def run_tuning(
    df,
    x_cols:           list,
    y_cols:           list,
    strategy:         str   = "random",
    n_iterations:     int   = 10,
    test_size:        float = 0.20,
    r2_threshold:     float = 0.85,
    per_y_threshold:  float = 0.80,
    focus_y_cols:     list  = None,  # score trials on these Y cols only (None = all)
    progress_callback       = None,  # fn(trial_idx, total, trial_result)
) -> dict:
    """
    Run N trials of hyperparameter search.

    Data is prepared once before the loop (sample → scale → split → tensors).
    Each trial only rebuilds the model and runs the training loop.

    Returns:
      trials            – all trial results
      best_params       – params of the highest-scoring trial
      best_overall_r2   – score of that trial (focus R² when focus_y_cols set)
      per_y_best_params – {col: params} that maximised each column's R²
      per_y_best_r2     – {col: best R²} per column
      poor_y_columns    – Y cols still below per_y_threshold at best params
      meets_threshold   – whether best score >= r2_threshold
      focus_y_cols      – which Y columns drove the scoring
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve scoring columns
    scored_y = [c for c in (focus_y_cols or []) if c in y_cols] or y_cols

    # ── Prepare data ONCE ──────────────────────────────────────────────────────
    if len(df) > TUNE_MAX_ROWS:
        df = df.sample(n=TUNE_MAX_ROWS, random_state=42).reset_index(drop=True)

    X = df[x_cols].values.astype(np.float32)
    Y = df[y_cols].values.astype(np.float32)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_sc = scaler_x.fit_transform(X)
    Y_sc = scaler_y.fit_transform(Y)

    X_tr, X_val, Y_tr_sc, Y_val_sc = train_test_split(
        X_sc, Y_sc, test_size=test_size, random_state=42
    )
    _, _, _, Y_val_raw = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    # Move to device once — all trials share these tensors (read-only)
    X_tr_t  = torch.tensor(X_tr).to(device)
    Y_tr_t  = torch.tensor(Y_tr_sc).to(device)
    X_val_t = torch.tensor(X_val).to(device)
    Y_val_t = torch.tensor(Y_val_sc).to(device)

    # ── Build candidate list ───────────────────────────────────────────────────
    all_combos = [
        {"noise_factor": nf, "hidden_dim": hd, "epochs": ep}
        for nf, hd, ep in itertools.product(
            PARAM_SPACE["noise_factor"],
            PARAM_SPACE["hidden_dim"],
            PARAM_SPACE["epochs"],
        )
    ]
    if strategy == "random":
        candidates = random.sample(all_combos, min(n_iterations, len(all_combos)))
    else:
        candidates = all_combos[:n_iterations]

    # ── Trial loop ─────────────────────────────────────────────────────────────
    trials       = []
    best_overall = None
    per_y_best   = {col: None for col in y_cols}

    for i, params in enumerate(candidates):
        try:
            result = _train_trial(
                X_tr_t=X_tr_t,
                Y_tr_t=Y_tr_t,
                X_val_t=X_val_t,
                Y_val_t=Y_val_t,
                Y_val_raw=Y_val_raw,
                scaler_y=scaler_y,
                x_dim=len(x_cols),
                y_cols=y_cols,
                params=params,
                device=device,
            )
            result["trial"] = i + 1

            # Focus score — mean R² of scored_y columns only
            focus_vals = [
                result["per_y_r2"].get(c)
                for c in scored_y
                if result["per_y_r2"].get(c) is not None
            ]
            focus_score          = _safe(round(float(np.mean(focus_vals)), 4)) if focus_vals else None
            result["focus_score"] = focus_score

            trials.append(result)

            # Best trial by focus score
            if focus_score is not None:
                if best_overall is None or focus_score > (best_overall.get("focus_score") or -999):
                    best_overall = result

            # Best trial per Y column
            for col in y_cols:
                col_r2 = result["per_y_r2"].get(col)
                if col_r2 is not None:
                    prev = per_y_best[col]
                    if prev is None or col_r2 > (prev["per_y_r2"].get(col) or -999):
                        per_y_best[col] = result

        except Exception as e:
            trials.append({
                "trial": i + 1, "params": params,
                "error": str(e), "overall_r2": None,
                "per_y_r2": {}, "focus_score": None,
            })

        if progress_callback:
            progress_callback(i + 1, len(candidates), trials[-1])

    # ── Aggregate results ──────────────────────────────────────────────────────
    poor_y = []
    if best_overall:
        poor_y = [
            col for col in y_cols
            if (best_overall["per_y_r2"].get(col) or 0) < per_y_threshold
        ]

    return {
        "strategy":          strategy,
        "n_trials":          len(trials),
        "trials":            trials,
        "focus_y_cols":      focus_y_cols,
        "scored_y_cols":     scored_y,
        "best_params":       best_overall["params"]      if best_overall else None,
        "best_overall_r2":   best_overall.get("focus_score") if best_overall else None,
        "per_y_best_params": {
            col: (per_y_best[col]["params"] if per_y_best[col] else None)
            for col in y_cols
        },
        "per_y_best_r2": {
            col: (per_y_best[col]["per_y_r2"].get(col) if per_y_best[col] else None)
            for col in y_cols
        },
        "poor_y_columns":  poor_y,
        "meets_threshold": (best_overall.get("focus_score") or 0) >= r2_threshold
                           if best_overall else False,
        "r2_threshold":    r2_threshold,
        "per_y_threshold": per_y_threshold,
    }
