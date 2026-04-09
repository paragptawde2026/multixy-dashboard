"""
routers/model.py
----------------
API endpoints for training models and running predictions.
Supports: DAE, LSTM, XGBoost, GPR, SSM.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
import pandas as pd
import os

from app.database import get_db
from app.models.db_models import Dataset, ModelRun, Prediction, TuningRun
from app.schemas.schemas import (
    TrainRequest, ModelRunResponse,
    PredictRequest, PredictionResponse,
    TuneRequest, TuningRunResponse,
)
from app.services.autoencoder import DAEService
from app.services.lstm_service import LSTMService
from app.services.xgboost_service import XGBoostService
from app.services.gpr_service import GPRService
from app.services.ssm_service import SSMService
from app.services.tuner import run_tuning

router = APIRouter(prefix="/model", tags=["Model"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
_MODEL_DIR  = os.path.join(UPLOAD_DIR, "models")

# Service singletons
dae_service     = DAEService(model_dir=_MODEL_DIR)
lstm_service    = LSTMService(model_dir=_MODEL_DIR)
xgboost_service = XGBoostService(model_dir=_MODEL_DIR)
gpr_service     = GPRService(model_dir=_MODEL_DIR)
ssm_service     = SSMService(model_dir=_MODEL_DIR)

ACTIVE_STATUSES = ("pending", "training")


def _get_service(model_type: str):
    """Return the correct service object for a given model_type string."""
    return {
        "lstm":    lstm_service,
        "xgboost": xgboost_service,
        "gpr":     gpr_service,
        "ssm":     ssm_service,
    }.get(model_type, dae_service)   # default = DAE


def _run_training(model_run_id: int, dataset_id: int):
    """
    Background task: trains the DAE model.
    Re-fetches all DB objects with its own session to avoid detached-instance errors.
    Updates progress in DB every 5 epochs.
    """
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
        run.status = "training"
        db.commit()

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        file_path = os.path.join(UPLOAD_DIR, dataset.filename)

        if dataset.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif dataset.original_name == "Data_DAE.xlsx":
            df = pd.read_excel(file_path, sheet_name="Model Data")
        else:
            df = pd.read_excel(file_path)

        def progress_callback(epoch, total, train_loss, val_loss):
            """Write live epoch progress into the DB so the frontend can poll it."""
            import math
            def safe(v):
                return None if (v is None or math.isnan(v) or math.isinf(v)) else round(v, 6)
            try:
                r = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
                existing = r.metrics or {}
                existing.update({
                    "current_epoch":   epoch,
                    "total_epochs":    total,
                    "live_train_loss": safe(train_loss),
                    "live_val_loss":   safe(val_loss),
                })
                r.metrics = dict(existing)
                db.commit()
            except Exception:
                pass

        service = _get_service(run.model_type or "dae")
        extra   = run.model_params or {}

        metrics = service.train(
            df=df,
            x_cols=run.x_columns,
            y_cols=run.y_columns,
            noise_factor=run.noise_factor,
            epochs=run.epochs,
            hidden_dim=run.hidden_dim,
            test_size=run.test_size,
            model_run_id=model_run_id,
            progress_callback=progress_callback,
            **extra,          # model-specific params (num_layers, n_estimators, etc.)
        )

        stopped_early = metrics.pop("stopped_early", False)
        run.model_path = metrics.pop("model_path", None)
        run.status     = "stopped" if stopped_early else "done"
        run.metrics    = metrics
        db.commit()

    except Exception as e:
        run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
        run.status  = "error"
        run.metrics = {"error": str(e)}
        db.commit()
    finally:
        db.close()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/train", response_model=ModelRunResponse)
def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Start training. Blocked if another training run is already active.
    Returns immediately (status='pending'); poll /model/runs/{id} for progress.
    """
    # Block concurrent training
    active = (
        db.query(ModelRun)
        .filter(ModelRun.status.in_(ACTIVE_STATUSES))
        .first()
    )
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Training run #{active.id} is already in progress (status: {active.status}). "
                   f"Stop it first or wait for it to finish.",
        )

    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    for col in request.x_columns + request.y_columns:
        if col not in dataset.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not in dataset.")

    if not request.x_columns:
        raise HTTPException(status_code=400, detail="Select at least one X column.")
    if not request.y_columns:
        raise HTTPException(status_code=400, detail="Select at least one Y column.")

    # Collect model-specific params into model_params JSON
    model_params = {}
    mt = request.model_type
    if mt == "lstm":
        model_params = {"num_layers": request.num_layers}
    elif mt == "xgboost":
        model_params = {
            "n_estimators":  request.n_estimators,
            "max_depth":     request.max_depth,
            "learning_rate": request.learning_rate,
        }
    elif mt == "gpr":
        model_params = {"gpr_kernel": request.gpr_kernel}
    elif mt == "ssm":
        model_params = {
            "ssm_q_scale": request.ssm_q_scale,
            "ssm_r_scale": request.ssm_r_scale,
        }

    run = ModelRun(
        dataset_id=request.dataset_id,
        model_type=mt,
        x_columns=request.x_columns,
        y_columns=request.y_columns,
        noise_factor=request.noise_factor,
        epochs=request.epochs,
        hidden_dim=request.hidden_dim,
        test_size=request.test_size,
        model_params=model_params if model_params else None,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(_run_training, run.id, dataset.id)
    return run


@router.post("/runs/{run_id}/stop", response_model=ModelRunResponse)
def stop_training(run_id: int, db: Session = Depends(get_db)):
    """
    Signal the training loop to stop. The loop checks the flag every batch
    so it exits within milliseconds once the data-loading phase is done.
    If the run is still in 'pending' (data loading), it is force-marked stopped immediately.
    """
    run = db.query(ModelRun).filter(ModelRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")
    if run.status not in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Run #{run_id} is not active (status: {run.status})."
        )

    # Signal the training thread to stop
    dae_service.request_stop(run_id)

    # If still pending (stuck in data loading), force-stop it immediately in DB
    if run.status == "pending":
        run.status = "stopped"
        run.metrics = {"error": "Stopped before training began (during data loading)."}
        db.commit()
        db.refresh(run)

    return run


@router.post("/kill-all")
def kill_all_training(db: Session = Depends(get_db)):
    """
    Emergency endpoint: immediately stops ALL active training runs.
    Marks them as 'stopped' in the database and sets all stop flags.
    Use this if the Stop button is unresponsive.
    """
    active_runs = db.query(ModelRun).filter(ModelRun.status.in_(ACTIVE_STATUSES)).all()

    count = 0
    for run in active_runs:
        dae_service.request_stop(run.id)
        run.status = "stopped"
        existing = run.metrics or {}
        existing["error"] = "Force-killed via kill-all endpoint."
        run.metrics = dict(existing)
        count += 1

    db.commit()
    return {"message": f"Killed {count} active training run(s).", "count": count}


def _sanitize_metrics(obj):
    """Recursively replace any nan/inf float values with None so JSON serialization never fails."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_metrics(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@router.get("/active")
def get_active_run(db: Session = Depends(get_db)):
    """Return the currently running/pending training run, or null if none."""
    active = (
        db.query(ModelRun)
        .filter(ModelRun.status.in_(ACTIVE_STATUSES))
        .order_by(ModelRun.created_at.desc())
        .first()
    )
    if not active:
        return {"active": None}
    return {
        "active": {
            "id":      active.id,
            "status":  active.status,
            "metrics": _sanitize_metrics(active.metrics or {}),
        }
    }


@router.get("/runs", response_model=list[ModelRunResponse])
def list_runs(db: Session = Depends(get_db)):
    return db.query(ModelRun).order_by(ModelRun.created_at.desc()).all()


@router.get("/runs/{run_id}", response_model=ModelRunResponse)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.query(ModelRun).filter(ModelRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")
    return run


@router.delete("/runs/{run_id}")
def delete_run(run_id: int, db: Session = Depends(get_db)):
    """
    Delete a model run and all associated data (predictions, model files on disk).
    Refuses to delete a currently active (pending/training) run.
    """
    import shutil as _shutil
    run = db.query(ModelRun).filter(ModelRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    if run.status in ("pending", "training"):
        raise HTTPException(
            status_code=409,
            detail="Cannot delete an active training run. Stop it first.",
        )

    # Remove linked predictions
    db.query(Prediction).filter(Prediction.model_run_id == run_id).delete()

    # Remove model files from disk
    model_path = os.path.join(UPLOAD_DIR, "models", f"model_{run_id}")
    if os.path.exists(model_path):
        _shutil.rmtree(model_path, ignore_errors=True)

    db.delete(run)
    db.commit()
    return {"deleted": True, "id": run_id}


@router.delete("/runs")
def bulk_delete_runs(keep_last: int = 0, status: str = None, db: Session = Depends(get_db)):
    """
    Bulk-delete runs.
    - keep_last > 0  → delete all except the N most-recent successful runs
    - status         → delete all runs with that status (error / stopped)
    Both filters can be combined.
    """
    import shutil as _shutil
    q = db.query(ModelRun).filter(ModelRun.status.notin_(["pending", "training"]))

    if status:
        q = q.filter(ModelRun.status == status)

    runs = q.order_by(ModelRun.created_at.desc()).all()

    if keep_last > 0 and not status:
        # Keep the N most-recent 'done' runs, delete the rest
        done_runs = [r for r in runs if r.status == "done"]
        keep_ids  = {r.id for r in done_runs[:keep_last]}
        runs      = [r for r in runs if r.id not in keep_ids]

    deleted_ids = []
    for run in runs:
        db.query(Prediction).filter(Prediction.model_run_id == run.id).delete()
        model_path = os.path.join(UPLOAD_DIR, "models", f"model_{run.id}")
        if os.path.exists(model_path):
            _shutil.rmtree(model_path, ignore_errors=True)
        db.delete(run)
        deleted_ids.append(run.id)

    db.commit()
    return {"deleted_count": len(deleted_ids), "deleted_ids": deleted_ids}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    run = db.query(ModelRun).filter(ModelRun.id == request.model_run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")
    if run.status not in ("done", "stopped"):
        raise HTTPException(
            status_code=400,
            detail=f"Model is not ready (status: {run.status}). "
                   "Only 'done' or 'stopped' models can predict."
        )

    service = _get_service(run.model_type or "dae")
    try:
        predictions = service.predict(run.id, request.input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    pred = Prediction(
        model_run_id=run.id,
        input_data=request.input_data,
        predicted_values=predictions,
        actual_values=None,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


@router.post("/whatif")
def whatif_predict(request: PredictRequest, db: Session = Depends(get_db)):
    run = db.query(ModelRun).filter(ModelRun.id == request.model_run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")
    if run.status not in ("done", "stopped"):
        raise HTTPException(status_code=400, detail=f"Model is not ready (status: {run.status}).")
    n_rows = max((len(v) for v in request.input_data.values()), default=0)
    if n_rows > 500:
        raise HTTPException(status_code=400, detail="Max 500 input rows per What-If call.")
    service = _get_service(run.model_type or "dae")
    try:
        predictions = service.predict(run.id, request.input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    return {"model_run_id": run.id, "predicted_values": predictions, "n_rows": n_rows}


@router.get("/runs/{run_id}/feature_weights")
def get_feature_weights(run_id: int, db: Session = Depends(get_db)):
    run = db.query(ModelRun).filter(ModelRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")
    if run.status not in ("done", "stopped"):
        raise HTTPException(status_code=400, detail=f"Model is not ready (status: {run.status}).")
    service = _get_service(run.model_type or "dae")
    try:
        result = service.compute_feature_weights(run.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature weight computation failed: {str(e)}")
    return result


@router.get("/comparison")
def model_comparison(db: Session = Depends(get_db)):
    """
    Return all completed runs with their metrics for model comparison.
    Identifies the best model overall and per-Y output.
    """
    runs = (
        db.query(ModelRun)
        .filter(ModelRun.status.in_(["done", "stopped"]))
        .order_by(ModelRun.created_at.desc())
        .all()
    )

    run_summaries = []
    for r in runs:
        m = r.metrics or {}
        run_summaries.append({
            "id":          r.id,
            "model_type":  r.model_type or "dae",
            "dataset_id":  r.dataset_id,
            "x_columns":   r.x_columns or [],
            "y_columns":   r.y_columns or [],
            "status":      r.status,
            "r2_score":    m.get("r2_score"),
            "per_y_r2":    m.get("per_y_r2", {}),
            "per_y_mae":   m.get("per_y_mae", {}),
            "n_train_rows": m.get("n_train_rows"),
            "n_test_rows":  m.get("n_test_rows"),
            "epochs_completed": m.get("epochs_completed"),
            "model_params": r.model_params,
            "created_at":  r.created_at.isoformat() if r.created_at else None,
        })

    # Best overall (highest R²)
    scored = [r for r in run_summaries if r["r2_score"] is not None]
    best_overall = max(scored, key=lambda r: r["r2_score"], default=None)

    # Best per Y output
    all_y = set()
    for r in run_summaries:
        all_y.update((r["per_y_r2"] or {}).keys())

    best_per_y = {}
    for yc in all_y:
        candidates = [
            r for r in run_summaries
            if r["per_y_r2"] and r["per_y_r2"].get(yc) is not None
        ]
        if candidates:
            best = max(candidates, key=lambda r: r["per_y_r2"][yc])
            best_per_y[yc] = {
                "run_id":     best["id"],
                "model_type": best["model_type"],
                "r2":         best["per_y_r2"][yc],
            }

    return {
        "runs":         run_summaries,
        "best_overall": {
            "run_id":     best_overall["id"],
            "model_type": best_overall["model_type"],
            "r2_score":   best_overall["r2_score"],
        } if best_overall else None,
        "best_per_y": best_per_y,
    }


@router.get("/runs/{run_id}/predict-test")
def predict_on_test_data(run_id: int, db: Session = Depends(get_db)):
    """
    Load the dataset used for this run, reproduce the exact train/test split,
    and return predicted vs actual Y values for the test rows.
    No user input required — uses the held-out test data automatically.
    """
    run = db.query(ModelRun).filter(ModelRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")
    if run.status not in ("done", "stopped"):
        raise HTTPException(
            status_code=400,
            detail=f"Model is not ready (status: {run.status})."
        )

    dataset = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    file_path = os.path.join(UPLOAD_DIR, dataset.filename)
    if dataset.filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif dataset.original_name == "Data_DAE.xlsx":
        df = pd.read_excel(file_path, sheet_name="Model Data")
    else:
        df = pd.read_excel(file_path)

    service = _get_service(run.model_type or "dae")
    try:
        result = service.predict_on_test_split(
            model_run_id=run.id,
            df=df,
            x_cols=run.x_columns,
            y_cols=run.y_columns,
            test_size=run.test_size,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # ── Sync DB if config.json columns differ from what is stored ──────────
    # This self-heals the run record so subsequent calls are consistent.
    actual_x = result.get("x_cols_used", run.x_columns)
    actual_y = result.get("y_cols_used", run.y_columns)
    if actual_x != run.x_columns or actual_y != run.y_columns:
        run.x_columns = actual_x
        run.y_columns = actual_y
        db.commit()

    return result


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────

def _run_tuning_task(tuning_run_id: int, dataset_id: int):
    """Background task: runs N search trials and saves results to DB."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        tr = db.query(TuningRun).filter(TuningRun.id == tuning_run_id).first()
        tr.status = "running"
        db.commit()

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        file_path = os.path.join(UPLOAD_DIR, dataset.filename)
        if dataset.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif dataset.original_name == "Data_DAE.xlsx":
            df = pd.read_excel(file_path, sheet_name="Model Data")
        else:
            df = pd.read_excel(file_path)

        completed_trials = []

        def progress_cb(trial_idx, total, trial_result):
            """Persist each completed trial so the UI can poll live progress."""
            completed_trials.append(trial_result)
            try:
                r = db.query(TuningRun).filter(TuningRun.id == tuning_run_id).first()
                r.current_trial = trial_idx
                r.results = {"trials": completed_trials, "status": "running"}
                db.commit()
            except Exception:
                pass

        results = run_tuning(
            df=df,
            x_cols=tr.x_columns,
            y_cols=tr.y_columns,
            strategy=tr.strategy,
            n_iterations=tr.n_iterations,
            test_size=tr.test_size,
            r2_threshold=tr.r2_threshold,
            per_y_threshold=tr.per_y_threshold,
            focus_y_cols=tr.focus_y_cols,
            progress_callback=progress_cb,
        )

        tr = db.query(TuningRun).filter(TuningRun.id == tuning_run_id).first()
        tr.status        = "done"
        tr.current_trial = results["n_trials"]
        tr.results       = results
        db.commit()

    except Exception as e:
        tr = db.query(TuningRun).filter(TuningRun.id == tuning_run_id).first()
        if tr:
            tr.status  = "error"
            tr.results = {"error": str(e)}
            db.commit()
    finally:
        db.close()


@router.post("/tune", response_model=TuningRunResponse)
def start_tuning(
    request: TuneRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Start a hyperparameter search (grid or random).
    Returns immediately — poll /model/tuning/{id} for progress.
    Only one tuning run is allowed at a time.
    """
    active = db.query(TuningRun).filter(TuningRun.status.in_(["pending", "running"])).first()
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Tuning run #{active.id} is already in progress. Wait for it to finish.",
        )

    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    tr = TuningRun(
        dataset_id      = request.dataset_id,
        x_columns       = request.x_columns,
        y_columns       = request.y_columns,
        strategy        = request.strategy,
        n_iterations    = request.n_iterations,
        r2_threshold    = request.r2_threshold,
        per_y_threshold = request.per_y_threshold,
        test_size       = request.test_size,
        focus_y_cols    = request.focus_y_cols or None,
        status          = "pending",
        current_trial   = 0,
    )
    db.add(tr)
    db.commit()
    db.refresh(tr)

    background_tasks.add_task(_run_tuning_task, tr.id, dataset.id)
    return tr


@router.get("/tuning/active")
def get_active_tuning(db: Session = Depends(get_db)):
    """Return the currently running tuning run, or null if none."""
    active = (
        db.query(TuningRun)
        .filter(TuningRun.status.in_(["pending", "running"]))
        .order_by(TuningRun.created_at.desc())
        .first()
    )
    return {"active": active}


@router.get("/tuning/{tuning_id}", response_model=TuningRunResponse)
def get_tuning_run(tuning_id: int, db: Session = Depends(get_db)):
    tr = db.query(TuningRun).filter(TuningRun.id == tuning_id).first()
    if not tr:
        raise HTTPException(status_code=404, detail="Tuning run not found.")
    return tr


@router.get("/tuning", response_model=list[TuningRunResponse])
def list_tuning_runs(db: Session = Depends(get_db)):
    return db.query(TuningRun).order_by(TuningRun.created_at.desc()).limit(20).all()


@router.get("/predictions", response_model=list[PredictionResponse])
def list_predictions(db: Session = Depends(get_db)):
    return db.query(Prediction).order_by(Prediction.created_at.desc()).all()
