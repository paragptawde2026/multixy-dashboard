"""
database.py
-----------
Sets up the database connection using SQLAlchemy.
Supports both SQLite (local dev) and PostgreSQL (production via DATABASE_URL env var).
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./multixy.db")

# connect_args is only needed for SQLite (allows multi-threaded access).
# PostgreSQL does not accept check_same_thread.
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency: opens a DB session, closes it when request is done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
