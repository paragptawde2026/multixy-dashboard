"""
db_models.py
------------
Defines the database tables using SQLAlchemy ORM.
Each class = one table in the database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.database import Base


class Dataset(Base):
    """Stores info about each uploaded CSV/Excel file."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    columns = Column(JSON)           # all numeric column names in the file
    x_columns = Column(JSON)         # input columns (from Variables_Type sheet or user selection)
    y_columns = Column(JSON)         # output columns (from Variables_Type sheet or user selection)
    row_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelRun(Base):
    """Stores each training run — supports DAE, LSTM, XGBoost, GPR, SSM."""
    __tablename__ = "model_runs"

    id           = Column(Integer, primary_key=True, index=True)
    dataset_id   = Column(Integer, nullable=False)
    model_type   = Column(String,  default="dae")   # dae | lstm | xgboost | gpr | ssm
    x_columns    = Column(JSON)
    y_columns    = Column(JSON)
    noise_factor = Column(Float,   default=0.1)     # DAE/LSTM
    epochs       = Column(Integer, default=100)     # DAE/LSTM
    hidden_dim   = Column(Integer, default=64)      # DAE/LSTM
    test_size    = Column(Float,   default=0.2)
    model_params = Column(JSON,    nullable=True)   # model-specific extra params
    status       = Column(String,  default="pending")
    metrics      = Column(JSON)
    model_path   = Column(String)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())


class TuningRun(Base):
    """Stores a hyperparameter tuning session (grid/random search)."""
    __tablename__ = "tuning_runs"

    id              = Column(Integer, primary_key=True, index=True)
    dataset_id      = Column(Integer, nullable=False)
    x_columns       = Column(JSON)
    y_columns       = Column(JSON)
    strategy        = Column(String, default="random")   # "grid" | "random"
    n_iterations    = Column(Integer, default=10)
    r2_threshold    = Column(Float,   default=0.85)      # overall trigger threshold
    per_y_threshold = Column(Float,   default=0.80)      # per-Y flag threshold
    test_size       = Column(Float,   default=0.20)
    focus_y_cols    = Column(JSON,    nullable=True)      # Y cols to optimise (None = all)
    status          = Column(String,  default="pending") # pending|running|done|error
    current_trial   = Column(Integer, default=0)
    results         = Column(JSON)                       # full tuner output dict
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    """Stores prediction results for later review."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_run_id = Column(Integer, nullable=False)
    input_data = Column(JSON)        # raw input values
    predicted_values = Column(JSON)  # model output
    actual_values = Column(JSON)     # true Y values (if provided)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
