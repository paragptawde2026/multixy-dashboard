"""
main.py
-------
Entry point for the FastAPI backend.
Registers all routes and starts the database.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.database import Base, engine
from app.routers import data, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
    1. Create all database tables.
    2. Reset any runs stuck in 'pending' or 'training' — they were killed
       when the server last stopped, so mark them 'stopped' so the UI unlocks.
    """
    Base.metadata.create_all(bind=engine)

    # Safe column migrations — idempotent ALTER TABLE statements
    from sqlalchemy import text
    _migrations = [
        ("tuning_runs",  "ALTER TABLE tuning_runs ADD COLUMN focus_y_cols TEXT"),
        ("model_runs",   "ALTER TABLE model_runs ADD COLUMN model_type TEXT DEFAULT 'dae'"),
        ("model_runs",   "ALTER TABLE model_runs ADD COLUMN model_params TEXT"),
    ]
    for _tbl, _sql in _migrations:
        try:
            with engine.begin() as _conn:
                _conn.execute(text(_sql))
        except Exception:
            pass   # column already exists

    from app.database import SessionLocal
    from app.models.db_models import ModelRun, TuningRun
    db = SessionLocal()
    try:
        # Reset stuck training runs
        stuck = db.query(ModelRun).filter(ModelRun.status.in_(["pending", "training"])).all()
        for run in stuck:
            run.status = "stopped"
            existing = run.metrics or {}
            existing["error"] = "Server was restarted while training was in progress."
            run.metrics = dict(existing)
        if stuck:
            db.commit()
            print(f"[startup] Reset {len(stuck)} stuck training run(s) to 'stopped'.")

        # Reset stuck tuning runs
        stuck_t = db.query(TuningRun).filter(TuningRun.status.in_(["pending", "running"])).all()
        for tr in stuck_t:
            tr.status  = "error"
            tr.results = {"error": "Server was restarted while tuning was in progress."}
        if stuck_t:
            db.commit()
            print(f"[startup] Reset {len(stuck_t)} stuck tuning run(s) to 'error'.")
    finally:
        db.close()

    yield


app = FastAPI(
    title="Multi X-Y DAE Dashboard",
    description="Denoising AutoEncoder for Multi-Input Multi-Output modeling",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React frontend to call this API.
# FRONTEND_ORIGIN env var can be set to the Vercel deployment URL in production.
_extra_origins = os.getenv("FRONTEND_ORIGIN", "").split(",")
_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    *[o.strip() for o in _extra_origins if o.strip()],
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route groups
app.include_router(data.router)
app.include_router(model.router)


@app.get("/")
def root():
    return {"message": "Multi X-Y DAE API is running!", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
