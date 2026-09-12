"""Model serialization with timestamped filenames (Task 2.5).

The brief assumes daily retraining, so every save must be independently
addressable: ``10-08-2020-16-32-31-00.pkl``. Keeping the full history (rather
than overwriting a single ``model.pkl``) is what lets a bad retrain be rolled
back and lets predictions be traced to the exact model version that produced
them.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib

from src.config import MODEL_TIMESTAMP_FMT, MODELS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def timestamp_now() -> str:
    """Timestamp in the brief's exact format, e.g. 10-08-2020-16-32-31-00."""
    # The brief's example has a trailing two-digit field after seconds
    # (hundredths); microseconds//10000 reproduces that.
    now = datetime.now()
    return now.strftime(MODEL_TIMESTAMP_FMT[:-3]) + f"-{now.microsecond // 10000:02d}"


def save_model(pipe, model_name: str = "rf", metadata: dict | None = None) -> Path:
    """Save a fitted pipeline as ``models/{model_name}_{timestamp}.pkl``.

    A sibling ``.json`` file with the same stem carries metrics and parameters
    so a later run can be identified without unpickling it.
    """
    ts = timestamp_now()
    filename = f"{model_name}_{ts}.pkl"
    path = MODELS_DIR / filename

    joblib.dump(pipe, path)
    logger.info("saved model -> %s", path)

    meta = {"timestamp": ts, "model_name": model_name, "file": filename}
    if metadata:
        meta.update(metadata)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("saved metadata -> %s", meta_path)

    return path


def load_latest_model(model_name: str | None = None) -> tuple:
    """Load the most recently saved model (optionally filtered by name).

    Returns ``(pipeline, metadata_dict)``.
    """
    pattern = f"{model_name}_*.pkl" if model_name else "*.pkl"
    candidates = sorted(MODELS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no saved models matching {pattern!r} in {MODELS_DIR}")

    path = candidates[-1]
    pipe = joblib.load(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    logger.info("loaded model <- %s", path)
    return pipe, meta


def list_models(model_name: str | None = None) -> list[dict]:
    """List every saved model version with its metadata, newest first."""
    pattern = f"{model_name}_*.json" if model_name else "*.json"
    metas = []
    for meta_path in sorted(MODELS_DIR.glob(pattern), reverse=True):
        try:
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return metas
