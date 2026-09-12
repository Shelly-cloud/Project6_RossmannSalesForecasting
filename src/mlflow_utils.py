"""MLflow experiment tracking helpers (Task 2.7).

Points MLflow at a local file store under ``mlruns/`` so no external server is
required for the assignment, and centralises the experiment name so every
script logs into the same place. ``launch_ui`` gives the one-line command to
open the dashboard for the screenshots the brief asks for.
"""
from __future__ import annotations

import mlflow

from src.config import MLFLOW_EXPERIMENT, MLRUNS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def _db_uri() -> str:
    """SQLite backend under mlruns/. MLflow 3.x retired the plain file store
    (it's in maintenance-only mode and raises on new experiments), so a local
    database is now the zero-server option; artifacts still land as plain
    files under mlruns/artifacts/.
    """
    db_path = MLRUNS_DIR / "mlflow.db"
    return f"sqlite:///{db_path.resolve().as_posix()}"


def init_mlflow() -> None:
    """Point MLflow at the project-local SQLite store and select the experiment."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(_db_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info("MLflow tracking at %s, experiment '%s'", _db_uri(), MLFLOW_EXPERIMENT)


def launch_ui_command() -> str:
    """The shell command to open the MLflow UI against this project's runs."""
    return f'mlflow ui --backend-store-uri "{_db_uri()}" --port 5000'
