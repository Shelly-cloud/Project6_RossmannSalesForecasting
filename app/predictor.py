"""Inference wrapper shared by the Flask backend.

Loads the latest serialized pipeline once at process start and exposes a
single ``predict_for_store`` entry point that takes a small per-date input
frame and returns sales predictions (plus a confidence interval when the
underlying model is a Random Forest). All the cleaning/feature-engineering
logic is the pipeline itself -- this module does not duplicate any of it,
which is the entire point of shipping the pipeline as one artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.confidence import tree_prediction_interval_sales
from src.data_loader import load_store
from src.features import build_holiday_calendar
from src.logger import get_logger
from src.serialize import list_models, load_latest_model

logger = get_logger("app.predictor")

_STATE = {"pipe": None, "meta": None, "store_df": None, "holiday_calendar": None}


def _ensure_loaded(model_name: str | None = "lightgbm") -> None:
    # LightGBM is the default: it validated more accurately than the Random
    # Forest (lower RMSPE) and is roughly 20x smaller on disk, which matters
    # for deploy cold-starts and memory. The Random Forest remains loadable
    # via reload_model() when its tree-spread confidence interval is wanted.
    if _STATE["pipe"] is not None:
        return
    try:
        pipe, meta = load_latest_model(model_name)
    except FileNotFoundError:
        pipe, meta = load_latest_model(None)  # fall back to any model
    _STATE["pipe"] = pipe
    _STATE["meta"] = meta
    _STATE["store_df"] = load_store()
    # An empty calendar is fine here: the dashboard's uploaded CSV supplies
    # its own IsHoliday column directly rather than relying on a lookup.
    _STATE["holiday_calendar"] = pd.DatetimeIndex([])
    logger.info("loaded model for serving: %s", meta.get("file", "unknown"))


def available_models() -> list[dict]:
    return list_models()


def reload_model(model_file: str | None = None) -> dict:
    """Force-reload, optionally pinning a specific saved model file by name."""
    import joblib

    from src.config import MODELS_DIR

    _STATE["pipe"] = None
    if model_file:
        path = MODELS_DIR / model_file
        _STATE["pipe"] = joblib.load(path)
        meta_path = path.with_suffix(".json")
        import json

        _STATE["meta"] = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        )
        _STATE["store_df"] = load_store()
        logger.info("pinned model: %s", model_file)
    else:
        _ensure_loaded()
    return _STATE["meta"]


def _build_input_frame(store_id: int, rows: pd.DataFrame) -> pd.DataFrame:
    """Assemble a raw-schema frame the pipeline can consume for one store.

    ``rows`` carries the date-dependent inputs the dashboard collects (Date,
    IsHoliday, IsWeekend, IsPromo, SchoolHoliday). Anything the pipeline
    needs beyond that is filled from the store's static attributes or a safe
    default, exactly as a fresh forecast request would be missing them.
    """
    store_df = _STATE["store_df"]
    store_row = store_df[store_df["Store"] == store_id]
    if store_row.empty:
        raise ValueError(f"unknown Store id: {store_id}")

    n = len(rows)
    frame = pd.DataFrame({"Store": [store_id] * n})
    frame["Date"] = pd.to_datetime(rows["Date"]).values
    frame["DayOfWeek"] = frame["Date"].dt.dayofweek + 1
    frame["Open"] = 1
    frame["Promo"] = rows.get("IsPromo", pd.Series([0] * n)).astype(int).values
    frame["StateHoliday"] = (
        rows.get("IsHoliday", pd.Series([0] * n)).astype(int).map({1: "a", 0: "0"}).values
    )
    frame["SchoolHoliday"] = rows.get("SchoolHoliday", pd.Series([0] * n)).astype(int).values

    for col in store_row.columns:
        if col == "Store":
            continue
        frame[col] = store_row.iloc[0][col]

    return frame


def predict_for_store(
    store_id: int, rows: pd.DataFrame, with_interval: bool = True
) -> pd.DataFrame:
    """Predict Sales (and an interval, where available) for a store over dates.

    Returns a frame with Date, PredictedSales, and (if available)
    PredictedSalesLower / PredictedSalesUpper.
    """
    _ensure_loaded()
    pipe = _STATE["pipe"]

    frame = _build_input_frame(store_id, rows)
    pred_log = pipe.predict(frame)
    pred = np.expm1(pred_log)

    out = pd.DataFrame({"Date": frame["Date"].values, "PredictedSales": pred})

    from sklearn.ensemble import RandomForestRegressor

    if with_interval and isinstance(pipe.named_steps["model"], RandomForestRegressor):
        try:
            intervals = tree_prediction_interval_sales(pipe, frame)
            out["PredictedSalesLower"] = intervals["lower"].values
            out["PredictedSalesUpper"] = intervals["upper"].values
        except Exception:  # noqa: BLE001 -- interval is a bonus, never block a prediction
            logger.exception("failed to compute prediction interval")

    # A basic customer estimate for the dashboard: sales / the store's own
    # historical sales-per-customer ratio, learned during training and
    # carried inside the pipeline's StoreAggregateFeatures step.
    agg_step = pipe.named_steps.get("store_aggregates")
    if agg_step is not None and hasattr(agg_step, "store_stats_"):
        stats = agg_step.store_stats_
        if store_id in stats.index and "StoreSalesPerCustomer" in stats.columns:
            ratio = stats.loc[store_id, "StoreSalesPerCustomer"]
            if ratio and ratio > 0:
                out["PredictedCustomers"] = (out["PredictedSales"] / ratio).round().astype(int)

    out.loc[frame["Open"].values == 0, ["PredictedSales"]] = 0
    return out


def model_metadata() -> dict:
    _ensure_loaded()
    return _STATE["meta"] or {}


def list_store_ids() -> list[int]:
    return sorted(load_store()["Store"].unique().tolist())
