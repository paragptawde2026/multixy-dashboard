"""
schemas.py
----------
Pydantic schemas define what data the API accepts and returns.
Think of them as "contracts" between frontend and backend.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


# --- Dataset schemas ---

class DatasetResponse(BaseModel):
    id: int
    filename: str
    original_name: str
    columns: List[str]
    x_columns: Optional[List[str]]
    y_columns: Optional[List[str]]
    row_count: int
    created_at: datetime

    class Config:
        from_attributes = True


# --- Model training schemas ---

class TrainRequest(BaseModel):
    dataset_id:   int
    x_columns:    List[str]
    y_columns:    List[str]
    model_type:   str   = "dae"    # dae | lstm | xgboost | gpr | ssm
    # DAE / LSTM shared
    noise_factor: float = 0.1
    epochs:       int   = 100
    hidden_dim:   int   = 64
    test_size:    float = 0.2
    # LSTM
    num_layers:   int   = 2
    # XGBoost
    n_estimators: int   = 100
    max_depth:    int   = 6
    learning_rate: float = 0.1
    # GPR
    gpr_kernel:   str   = "rbf"    # rbf | matern52
    # SSM
    ssm_q_scale:  float = 0.01
    ssm_r_scale:  float = 0.1


class ModelRunResponse(BaseModel):
    id:           int
    dataset_id:   int
    model_type:   Optional[str]
    x_columns:    List[str]
    y_columns:    List[str]
    noise_factor: float
    epochs:       int
    hidden_dim:   int
    test_size:    float
    model_params: Optional[Dict[str, Any]]
    status:       str
    metrics:      Optional[Dict[str, Any]]
    created_at:   datetime

    class Config:
        from_attributes = True


# --- Preprocessing schemas ---

class PreprocessRequest(BaseModel):
    dataset_name: Optional[str] = None             # Custom output dataset name (auto-generated if omitted)
    columns_to_drop: Optional[List[str]] = None    # Step 0 — remove duplicate/unwanted columns
    remove_nan: bool = True                        # Step 1 — drop rows with any NaN
    outlier_method: str = "iqr"                    # Step 2 — "none", "iqr", "zscore"
    outlier_threshold: float = 1.5                 # IQR multiplier or Z-score cutoff
    columns_to_check: Optional[List[str]] = None   # columns for outlier detection (None = all)


# --- Hyperparameter tuning schemas ---

class TuneRequest(BaseModel):
    dataset_id:       int
    x_columns:        List[str]
    y_columns:        List[str]
    strategy:         str            = "random"  # "grid" | "random"
    n_iterations:     int            = 10
    r2_threshold:     float          = 0.85      # flag if overall R² below this
    per_y_threshold:  float          = 0.80      # flag per-Y column below this
    test_size:        float          = 0.20
    focus_y_cols:     Optional[List[str]] = None  # Y cols to optimise; None = all


class TuningRunResponse(BaseModel):
    id:              int
    dataset_id:      int
    x_columns:       List[str]
    y_columns:       List[str]
    strategy:        str
    n_iterations:    int
    r2_threshold:    float
    per_y_threshold: float
    test_size:       float
    focus_y_cols:    Optional[List[str]]
    status:          str
    current_trial:   int
    results:         Optional[Dict[str, Any]]
    created_at:      datetime

    class Config:
        from_attributes = True


# --- Prediction schemas ---

class PredictRequest(BaseModel):
    model_run_id: int
    input_data: Dict[str, List[float]]   # column_name -> list of values


class PredictionResponse(BaseModel):
    id: int
    model_run_id: int
    input_data: Dict[str, Any]
    predicted_values: Dict[str, Any]
    actual_values: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True
