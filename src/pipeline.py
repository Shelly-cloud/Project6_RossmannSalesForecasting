"""End-to-end sklearn Pipeline assembly (Task 2.2).

The whole cleaning -> feature-engineering -> encoding -> model chain is a single
``Pipeline`` object. That matters for two reasons the brief cares about:

1. **Reproducibility / no leakage.** Every stateful step (store aggregates,
   scaler, one-hot categories) learns its parameters inside ``fit`` on the
   training fold only. A random ``train_test_split`` of pre-computed features
   would leak the validation fold's own statistics back into training.

2. **Serving is trivial.** ``joblib.dump(pipeline)`` produces one artefact that
   takes a *raw joined dataframe* and returns predictions. The Flask/Streamlit
   app therefore contains no duplicated preprocessing code, which is the usual
   source of train/serve skew.

One deliberate exception: dropping closed days cannot live inside the pipeline,
because an sklearn transformer may not change the row count during ``fit``
(``y`` would no longer align). That filtering is a separate, explicit step in
``train_ml.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import MissingValueCleaner
from src.config import CATEGORICAL_FEATURES, RANDOM_STATE
from src.features import (
    CompetitionPromoFeatures,
    DateFeatureExtractor,
    HolidayFeatureExtractor,
    StoreAggregateFeatures,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Columns that must never become features.
#   Sales/Customers -> target and a value unknown at forecast time
#   Date/Id         -> row keys, already expanded into calendar features
#   IsSalesOutlier  -> derived from the target itself, so pure leakage
#   Open            -> constant 1 once closed days are filtered out
#   StateName       -> redundant with the State code
EXCLUDED_FROM_FEATURES = {
    "Sales",
    "Customers",
    "Date",
    "Id",
    "IsSalesOutlier",
    "Open",
    "StateName",
    "LogSales",
}


def infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split an engineered frame into (numeric, categorical) feature names."""
    usable = [c for c in df.columns if c not in EXCLUDED_FROM_FEATURES]

    categorical = [c for c in usable if c in CATEGORICAL_FEATURES]
    numeric = [
        c
        for c in usable
        if c not in categorical and pd.api.types.is_numeric_dtype(df[c])
    ]

    dropped = set(usable) - set(categorical) - set(numeric)
    if dropped:
        logger.warning("ignoring non-numeric, non-categorical columns: %s", sorted(dropped))

    logger.info(
        "feature columns: %d numeric, %d categorical", len(numeric), len(categorical)
    )
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categoricals.

    ``StandardScaler`` is applied as the brief requires. It is genuinely
    necessary for the LSTM and for any distance-based model; for the tree
    models here it is a no-op on accuracy but harmless, and keeping it in the
    pipeline means swapping in a linear or SVR baseline needs no other change.

    ``handle_unknown="ignore"`` matters in production: the test window contains
    only StateHoliday values ``0`` and ``a``, and a dashboard user could submit
    a category the training fold never saw. Ignoring it yields an all-zero
    dummy block instead of a crash.
    """
    numeric_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_branch, numeric_features),
            ("cat", categorical_branch, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(
    holiday_calendar: pd.DatetimeIndex | None = None,
) -> Pipeline:
    """The cleaning + feature-engineering half of the chain (no model)."""
    return Pipeline(
        steps=[
            ("clean_missing", MissingValueCleaner()),
            ("date_features", DateFeatureExtractor()),
            ("holiday_features", HolidayFeatureExtractor(calendar=holiday_calendar)),
            ("competition_promo", CompetitionPromoFeatures()),
            ("store_aggregates", StoreAggregateFeatures()),
        ]
    )


def build_full_pipeline(
    model=None,
    holiday_calendar: pd.DatetimeIndex | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Assemble raw-dataframe -> prediction as one fittable, picklable object.

    ``numeric_features``/``categorical_features`` are normally left as ``None``
    and resolved by :func:`fit_full_pipeline`, which needs one pass of feature
    engineering to know what columns exist.
    """
    if model is None:
        # The brief names Random Forest as a reasonable starting point.
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    steps = list(build_feature_pipeline(holiday_calendar).steps)

    if numeric_features is not None and categorical_features is not None:
        steps.append(
            ("preprocess", build_preprocessor(numeric_features, categorical_features))
        )
    steps.append(("model", model))
    return Pipeline(steps=steps)


def fit_full_pipeline(
    raw_df: pd.DataFrame,
    y: np.ndarray,
    model=None,
    holiday_calendar: pd.DatetimeIndex | None = None,
) -> Pipeline:
    """Resolve feature columns, then fit the complete pipeline.

    The feature half is run once on a small sample purely to discover the
    output column names, which ``ColumnTransformer`` needs up front. The
    sample is drawn at random rather than taken from the head: the frame is
    sorted by Store then Date, so the first N rows would only cover a
    handful of stores and could miss rarer categories (State, StoreType).
    """
    sample_n = min(20000, len(raw_df))
    sample = raw_df.sample(sample_n, random_state=RANDOM_STATE).copy()
    feature_half = build_feature_pipeline(holiday_calendar)
    engineered = feature_half.fit_transform(sample)
    numeric, categorical = infer_feature_columns(engineered)

    pipe = build_full_pipeline(
        model=model,
        holiday_calendar=holiday_calendar,
        numeric_features=numeric,
        categorical_features=categorical,
    )
    logger.info("fitting full pipeline on %d rows", len(raw_df))
    pipe.fit(raw_df, y)
    logger.info("pipeline fitted")
    return pipe


def get_feature_names(pipe: Pipeline) -> list[str]:
    """Feature names as seen by the final estimator (post one-hot expansion)."""
    return list(pipe.named_steps["preprocess"].get_feature_names_out())
