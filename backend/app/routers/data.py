"""
routers/data.py
---------------
API endpoints for uploading and managing datasets.
Supports both manual CSV/Excel upload and auto-loading from the
project's Data_DAE.xlsx file (reads Variables_Type sheet for X/Y split).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import uuid
import os
import shutil

from app.database import get_db
from app.models.db_models import Dataset
from app.schemas.schemas import DatasetResponse, PreprocessRequest

router = APIRouter(prefix="/data", tags=["Data"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Fixed path to the project data file
DATA_FILE_PATH = os.path.join(
    os.path.dirname(__file__),          # .../backend/app/routers
    "..", "..", "..", "..",             # up to "SNOP Product"
    "Multi X-Y", "Data_DAE.xlsx"
)


def _next_dataset_name(source_original_name: str, db) -> str:
    """
    Generate a professional versioned name for a preprocessed dataset.

    Algorithm:
      1. Strip all leading "cleaned_" prefixes from the source name
      2. Strip the file extension
      3. Strip any existing "_v{N}" suffix to get the true base stem
      4. Count how many versioned datasets derived from that stem already exist
      5. Return  "<stem>_v<next>"

    Examples:
      Data_DAE.xlsx                        →  Data_DAE_v1
      Data_DAE_v1  (if v1 exists)          →  Data_DAE_v2
      cleaned_Data_DAE.xlsx (v1 exists)    →  Data_DAE_v2
      cleaned_cleaned_Data_DAE.xlsx        →  Data_DAE_v2  (same base)
    """
    import re

    name = source_original_name
    # Strip all leading "cleaned_" prefixes
    while name.startswith("cleaned_"):
        name = name[len("cleaned_"):]
    # Strip file extension
    stem = name.rsplit(".", 1)[0] if "." in name else name
    # Strip any existing "_v{N}" suffix
    stem = re.sub(r"_v\d+$", "", stem)

    # Count existing datasets whose name matches "<stem>_v<digits>"
    pattern = f"{stem}_v%"
    existing_count = db.query(Dataset).filter(
        Dataset.original_name.like(pattern)
    ).count()

    return f"{stem}_v{existing_count + 1}"


def _load_df(dataset: Dataset) -> pd.DataFrame:
    """Load a dataset's file as a DataFrame, handling CSV and Excel formats."""
    file_path = os.path.join(UPLOAD_DIR, dataset.filename)
    if dataset.filename.endswith(".csv"):
        return pd.read_csv(file_path)
    elif dataset.original_name == "Data_DAE.xlsx":
        return pd.read_excel(file_path, sheet_name="Model Data")
    else:
        return pd.read_excel(file_path)


def _parse_variables_type(xl_path: str):
    """
    Read the 'Variables_Type' sheet and return (x_cols, y_cols).
    Column layout: col[1] = variable name, col[2] = 'X' or 'Y'
    """
    vt = pd.read_excel(xl_path, sheet_name="Variables_Type", header=None)
    x_cols, y_cols = [], []
    for _, row in vt.iterrows():
        name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        kind = str(row.iloc[2]).strip().upper() if pd.notna(row.iloc[2]) else ""
        if not name or name == "nan":
            continue
        if kind == "X":
            x_cols.append(name)
        elif kind == "Y":
            y_cols.append(name)
    return x_cols, y_cols


# ── Auto-load from project Data_DAE.xlsx ────────────────────────────────────

@router.post("/load-excel", response_model=DatasetResponse)
def load_project_excel(db: Session = Depends(get_db)):
    """
    Load Data_DAE.xlsx from the project folder.
    Reads 'Model Data' sheet for data and 'Variables_Type' sheet for X/Y split.
    Registers it as a dataset in the database.
    """
    abs_path = os.path.abspath(DATA_FILE_PATH)
    if not os.path.exists(abs_path):
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found at: {abs_path}"
        )

    # Check if already loaded (avoid duplicates)
    existing = db.query(Dataset).filter(Dataset.original_name == "Data_DAE.xlsx").first()
    if existing:
        return existing

    # Read Model Data sheet
    df = pd.read_excel(abs_path, sheet_name="Model Data")

    # Drop Timestamp column, keep only numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Parse X/Y from Variables_Type sheet
    x_cols, y_cols = _parse_variables_type(abs_path)

    # Only keep X/Y cols that actually exist in the data
    x_cols = [c for c in x_cols if c in numeric_cols]
    y_cols = [c for c in y_cols if c in numeric_cols]

    # Copy file to uploads folder so the model trainer can access it
    dest_name = f"data_dae_{uuid.uuid4().hex[:8]}.xlsx"
    dest_path = os.path.join(UPLOAD_DIR, dest_name)
    shutil.copy2(abs_path, dest_path)

    dataset = Dataset(
        filename=dest_name,
        original_name="Data_DAE.xlsx",
        columns=numeric_cols,
        x_columns=x_cols,
        y_columns=y_cols,
        row_count=len(df),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


# ── Manual file upload ───────────────────────────────────────────────────────

@router.post("/upload", response_model=DatasetResponse)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a CSV or Excel file manually.
    X/Y columns will not be auto-detected; user selects them in the UI.
    """
    if not (file.filename.endswith(".csv") or file.filename.endswith((".xlsx", ".xls"))):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported.")

    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="File must have at least 2 numeric columns.")

    dataset = Dataset(
        filename=unique_name,
        original_name=file.filename,
        columns=numeric_cols,
        x_columns=None,
        y_columns=None,
        row_count=len(df),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


# ── Query endpoints ──────────────────────────────────────────────────────────

@router.get("/", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, force: bool = False, db: Session = Depends(get_db)):
    """
    Delete a dataset record from the database and remove its file from disk.

    force=False (default): refuses if any model run references this dataset.
    force=True:            also deletes all linked runs, their predictions, and
                           model files from disk before removing the dataset.
    """
    import shutil as _shutil
    from app.models.db_models import ModelRun, Prediction

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    linked_runs = db.query(ModelRun).filter(ModelRun.dataset_id == dataset_id).all()

    # Block deletion of active runs regardless of force flag
    active = [r for r in linked_runs if r.status in ("pending", "training")]
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete: Run #{active[0].id} is currently active. Stop it first.",
        )

    if linked_runs and not force:
        run_ids = ", ".join(f"#{r.id}" for r in linked_runs)
        raise HTTPException(
            status_code=409,
            detail=f"Dataset is used by run(s) {run_ids}. "
                   "Use force=true to also delete those runs, or delete the runs first.",
        )

    # Force cascade: remove linked runs + predictions + model files
    if force and linked_runs:
        for run in linked_runs:
            db.query(Prediction).filter(Prediction.model_run_id == run.id).delete()
            model_path = os.path.join(UPLOAD_DIR, "models", f"model_{run.id}")
            if os.path.exists(model_path):
                _shutil.rmtree(model_path, ignore_errors=True)
            db.delete(run)

    # Remove the dataset file from disk (best-effort)
    file_path = os.path.join(UPLOAD_DIR, dataset.filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    db.delete(dataset)
    db.commit()
    return {"deleted": True, "id": dataset_id, "runs_deleted": len(linked_runs) if force else 0}


@router.get("/{dataset_id}/download")
def download_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Download the dataset as a CSV file."""
    import io
    from fastapi.responses import FileResponse, StreamingResponse

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    file_path = os.path.join(UPLOAD_DIR, dataset.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk.")

    stem = dataset.original_name.rsplit(".", 1)[0] if "." in dataset.original_name else dataset.original_name
    download_name = f"{stem}.csv"

    if dataset.filename.endswith(".csv"):
        return FileResponse(
            path=file_path,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )
    else:
        # Excel → convert to CSV in-memory
        if dataset.original_name == "Data_DAE.xlsx":
            df = pd.read_excel(file_path, sheet_name="Model Data")
        else:
            df = pd.read_excel(file_path)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset


@router.get("/{dataset_id}/sample_rows")
def sample_rows(
    dataset_id: int,
    x_cols: str = None,
    n: int = 150,
    db: Session = Depends(get_db),
):
    """
    Return N evenly-sampled rows from the dataset for the What-If baseline row picker.
    Includes a timestamp column (if one exists) for labelling.
    x_cols: comma-separated list of X column names to include in each row.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    file_path = os.path.join(UPLOAD_DIR, dataset.filename)
    # Load full file — we want ALL columns including non-numeric for timestamp detection
    if dataset.filename.endswith(".csv"):
        df_full = pd.read_csv(file_path)
    elif dataset.original_name == "Data_DAE.xlsx":
        df_full = pd.read_excel(file_path, sheet_name="Model Data")
    else:
        df_full = pd.read_excel(file_path)

    # Detect a timestamp / date column for display labels
    ts_col = None
    for col in df_full.columns:
        if col.lower() in ("timestamp", "time", "date", "datetime", "ts", "date_time"):
            ts_col = col
            break

    # Resolve requested X columns
    requested = (
        [c.strip() for c in x_cols.split(",") if c.strip()]
        if x_cols
        else (dataset.x_columns or [])
    )
    valid_x = [c for c in requested if c in df_full.columns]

    n_rows  = len(df_full)
    indices = np.linspace(0, n_rows - 1, min(n, n_rows), dtype=int).tolist()

    rows = []
    for idx in indices:
        row_vals = {}
        for c in valid_x:
            v = df_full.iloc[idx][c]
            row_vals[c] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 6)

        entry = {"row_index": int(idx), "x_values": row_vals}
        if ts_col:
            ts_val = df_full.iloc[idx][ts_col]
            entry["timestamp"] = str(ts_val) if not (pd.isna(ts_val)) else None
        rows.append(entry)

    return {
        "total_rows":     n_rows,
        "sample_size":    len(rows),
        "has_timestamp":  ts_col is not None,
        "timestamp_col":  ts_col,
        "x_cols":         valid_x,
        "rows":           rows,
    }


@router.get("/{dataset_id}/x_coupling")
def x_coupling(
    dataset_id: int,
    vary_x: str,
    x_cols: str = None,
    db: Session = Depends(get_db),
):
    """
    Compute OLS regression slopes β_i = cov(vary_x, X_i) / var(vary_x)
    for every other X column in x_cols.

    This tells the What-If simulator how much X_i is expected to shift
    (in original units) for each unit shift in vary_x, based on the
    linear co-movement observed in the training dataset.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _load_df(dataset)

    # Resolve columns
    requested = (
        [c.strip() for c in x_cols.split(",") if c.strip()]
        if x_cols
        else (dataset.x_columns or [])
    )
    valid = [c for c in requested if c in df.columns]
    if vary_x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{vary_x}' not found in dataset.")

    # Keep only numeric, drop NaNs
    sub = df[valid].dropna()
    if len(sub) < 10:
        raise HTTPException(status_code=400, detail="Not enough rows to compute coupling.")

    vx = sub[vary_x].values
    var_vx = float(np.var(vx, ddof=1))
    if var_vx == 0:
        return {"vary_x": vary_x, "betas": {}, "correlations": {}}

    betas        = {}
    correlations = {}
    for col in valid:
        if col == vary_x:
            continue
        xi   = sub[col].values
        cov  = float(np.cov(vx, xi, ddof=1)[0, 1])
        beta = cov / var_vx
        std_xi = float(np.std(xi, ddof=1))
        std_vx = float(np.std(vx, ddof=1))
        corr   = cov / (std_vx * std_xi) if (std_vx * std_xi) > 0 else 0.0
        betas[col]        = round(beta, 8)
        correlations[col] = round(corr, 4)

    return {"vary_x": vary_x, "betas": betas, "correlations": correlations}


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: int, rows: int = 10, db: Session = Depends(get_db)):
    """Return first N rows for preview."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    file_path = os.path.join(UPLOAD_DIR, dataset.filename)
    if dataset.original_name.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, sheet_name="Model Data" if dataset.original_name == "Data_DAE.xlsx" else 0)

    df = df[dataset.columns]
    return {
        "columns": dataset.columns,
        "x_columns": dataset.x_columns or [],
        "y_columns": dataset.y_columns or [],
        "rows": df.head(rows).round(4).to_dict(orient="records"),
    }


# ── Preprocessing endpoints ──────────────────────────────────────────────────

@router.get("/{dataset_id}/stats")
def dataset_stats(dataset_id: int, db: Session = Depends(get_db)):
    """
    Return per-column statistics: null counts, min/max/mean/std,
    and estimated outlier counts using IQR (1.5x and 3.0x) and Z-score (3.0).
    Used by the Preprocessing page to let the user decide which steps to apply.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _load_df(dataset)[dataset.columns]

    col_stats = []
    for col in dataset.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        s_valid = s.dropna()
        n_valid = len(s_valid)

        if n_valid == 0:
            col_stats.append({
                "name": col, "null_count": null_count,
                "null_pct": 100.0,
                "mean": None, "std": None, "min": None, "max": None,
                "q1": None, "q3": None,
                "iqr_outliers_15": 0, "iqr_outliers_30": 0, "zscore_outliers_3": 0,
            })
            continue

        mean  = float(s_valid.mean())
        std   = float(s_valid.std()) if n_valid > 1 else 0.0
        q1    = float(s_valid.quantile(0.25))
        q3    = float(s_valid.quantile(0.75))
        iqr   = q3 - q1

        low15, high15 = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        low30, high30 = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        out15 = int(((s_valid < low15) | (s_valid > high15)).sum())
        out30 = int(((s_valid < low30) | (s_valid > high30)).sum())
        zout  = int((((s_valid - mean) / std).abs() > 3.0).sum()) if std > 0 else 0

        col_stats.append({
            "name":              col,
            "null_count":        null_count,
            "null_pct":          round(null_count / len(df) * 100, 2),
            "mean":              round(mean, 4),
            "std":               round(std,  4),
            "min":               round(float(s_valid.min()), 4),
            "max":               round(float(s_valid.max()), 4),
            "q1":                round(q1, 4),
            "q3":                round(q3, 4),
            "iqr_outliers_15":   out15,
            "iqr_outliers_30":   out30,
            "zscore_outliers_3": zout,
        })

    # ── Duplicate column detection (optimized) ─────────────────────────────
    # Old approach: O(n² * rows) pairwise comparison — timed out on 44k-row datasets.
    # New approach: hash each column once, group columns by hash (O(n * rows)).
    # For large datasets (>10k rows), sample 5000 rows to speed up hashing.
    import hashlib

    cols_list = df.columns.tolist()
    sample_df = df.sample(n=min(5000, len(df)), random_state=42) if len(df) > 10000 else df

    hash_groups = {}  # hash -> [col_names]
    for col in cols_list:
        s = sample_df[col]
        # Convert NaN to a sentinel string for consistent hashing
        col_bytes = s.fillna("__NAN__").astype(str).values.tobytes()
        h = hashlib.md5(col_bytes).hexdigest()
        hash_groups.setdefault(h, []).append(col)

    dup_groups = []
    for cols in hash_groups.values():
        if len(cols) > 1:
            # Verify with full comparison on original data for correctness
            verified = [cols[0]]
            for c in cols[1:]:
                if df[cols[0]].equals(df[c]):
                    verified.append(c)
            if len(verified) > 1:
                dup_groups.append({
                    "columns":        verified,
                    "suggested_drop": verified[1:],
                })

    return {
        "total_rows":              len(df),
        "null_rows":               int(df.isna().any(axis=1).sum()),
        "columns":                 col_stats,
        "duplicate_column_groups": dup_groups,
    }


@router.post("/{dataset_id}/preprocess")
def preprocess_dataset(
    dataset_id: int,
    req: PreprocessRequest,
    db: Session = Depends(get_db),
):
    """
    Apply NaN removal and/or outlier filtering to a dataset.
    Saves the cleaned data as a new CSV and registers it as a new Dataset.
    Returns the new dataset ID plus removal counts for each step.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _load_df(dataset)[dataset.columns]
    original_rows  = len(df)
    original_cols  = df.columns.tolist()

    # Carry forward current X/Y assignments; will be pruned after column drop
    new_x_cols = list(dataset.x_columns or [])
    new_y_cols = list(dataset.y_columns or [])

    # ── Step 0: Drop unwanted / duplicate columns ──────────────────────────
    cols_dropped = []
    if req.columns_to_drop:
        valid_drops = [c for c in req.columns_to_drop if c in df.columns]
        if valid_drops:
            df = df.drop(columns=valid_drops)
            cols_dropped = valid_drops
            # Remove dropped cols from X/Y assignments so the new dataset
            # doesn't reference columns that no longer exist
            new_x_cols = [c for c in new_x_cols if c not in cols_dropped]
            new_y_cols = [c for c in new_y_cols if c not in cols_dropped]

    # ── Step 1: Remove rows with NaN / Null values ─────────────────────────
    nan_removed = 0
    if req.remove_nan:
        before = len(df)
        df = df.dropna()
        nan_removed = before - len(df)

    # ── Step 2: Remove outlier rows ────────────────────────────────────────
    outlier_removed = 0
    cols_to_check = req.columns_to_check if req.columns_to_check else dataset.columns
    cols_to_check = [c for c in cols_to_check if c in df.columns]

    if req.outlier_method != "none" and cols_to_check:
        before = len(df)
        keep = pd.Series(True, index=df.index)

        if req.outlier_method == "iqr":
            for col in cols_to_check:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                low  = q1 - req.outlier_threshold * iqr
                high = q3 + req.outlier_threshold * iqr
                keep &= (df[col] >= low) & (df[col] <= high)

        elif req.outlier_method == "zscore":
            for col in cols_to_check:
                mean = df[col].mean()
                std  = df[col].std()
                if std > 0:
                    keep &= ((df[col] - mean) / std).abs() <= req.outlier_threshold

        df = df[keep]
        outlier_removed = before - len(df)

    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="All rows were removed after preprocessing. "
                   "Please use a less aggressive threshold or deselect some columns.",
        )

    # ── Save cleaned data ──────────────────────────────────────────────────
    clean_filename = f"processed_{uuid.uuid4().hex[:8]}.csv"
    clean_path = os.path.join(UPLOAD_DIR, clean_filename)
    df.to_csv(clean_path, index=False)

    # Determine the display name: user-supplied or auto-versioned
    if req.dataset_name and req.dataset_name.strip():
        new_original_name = req.dataset_name.strip()
    else:
        new_original_name = _next_dataset_name(dataset.original_name, db)

    new_ds = Dataset(
        filename=clean_filename,
        original_name=new_original_name,
        columns=df.columns.tolist(),
        x_columns=new_x_cols if new_x_cols else dataset.x_columns,
        y_columns=new_y_cols if new_y_cols else dataset.y_columns,
        row_count=len(df),
    )
    db.add(new_ds)
    db.commit()
    db.refresh(new_ds)

    return {
        "dataset_id":           new_ds.id,
        "original_name":        new_ds.original_name,
        "original_rows":        original_rows,
        "cleaned_rows":         len(df),
        "cols_dropped":         cols_dropped,
        "nan_rows_removed":     nan_removed,
        "outlier_rows_removed": outlier_removed,
        "rows_retained_pct":    round(len(df) / original_rows * 100, 1),
    }
