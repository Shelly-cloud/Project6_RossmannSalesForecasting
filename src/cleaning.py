"""Data cleaning: missing-value and outlier handling (Task 1).

Implemented as sklearn-compatible transformers so the exact same cleaning runs
in the notebook, in ``train_ml.py`` and inside the served model. That is the
whole point of the pipeline requirement: cleaning logic cannot drift between
training and serving if there is only one copy of it.

Design decisions worth defending in the write-up
------------------------------------------------
``CompetitionDistance`` NaN does **not** mean "distance zero" -- it means no
competitor is on record, which behaves like a *very distant* competitor.
Imputing 0 or the mean would invert the signal, so we impute a large sentinel
and add an explicit ``HasCompetition`` flag.

Sales outliers are **detected and flagged, not blindly removed**. The largest
outliers in this dataset are the pre-Christmas trading peaks, which are real,
repeating and exactly what the finance team needs forecast. Removing them
would make the model systematically under-forecast December.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import MIN_TRAINABLE_SALES
from src.logger import get_logger

logger = get_logger(__name__)

# No competitor on record behaves like a far-away competitor, not a near one.
# ~75 km is an order of magnitude beyond the observed maximum (~76 km).
NO_COMPETITION_DISTANCE = 200_000.0


class MissingValueCleaner(BaseEstimator, TransformerMixin):
    """Impute the known-missing columns of the joined Rossmann frame.

    Stateless with respect to the data (all rules are domain constants), so
    ``fit`` is a no-op and there is no train/test leakage risk.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()

        # test.csv has 11 rows (store 622) with Open missing. The store trades
        # on every other weekday in train, so "open" is the safe assumption;
        # assuming closed would hard-zero 11 real forecasts.
        if "Open" in df.columns and df["Open"].isna().any():
            n = int(df["Open"].isna().sum())
            df["Open"] = df["Open"].fillna(1)
            logger.info("imputed %d missing Open values as 1 (open)", n)

        if "CompetitionDistance" in df.columns:
            missing = df["CompetitionDistance"].isna()
            df["HasCompetition"] = (~missing).astype(int)
            df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
                NO_COMPETITION_DISTANCE
            )
            if missing.any():
                logger.info(
                    "imputed %d missing CompetitionDistance as %.0f m + flag",
                    int(missing.sum()),
                    NO_COMPETITION_DISTANCE,
                )

        # A missing competition-open date means "competitor predates our
        # records". Zero is a valid sentinel here because the downstream
        # feature (months since opening) also checks the companion flag.
        for col in ("CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"):
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Promo2Since* are missing exactly when the store never joined Promo2.
        for col in ("Promo2SinceWeek", "Promo2SinceYear"):
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        if "PromoInterval" in df.columns:
            df["PromoInterval"] = df["PromoInterval"].fillna("None")

        # Weather / trend gaps: forward-fill within a store's own time series
        # (weather is highly autocorrelated day to day), then fall back to the
        # column median for any leading gap.
        weather_like = [
            c
            for c in df.columns
            if c.startswith(("Max_", "Mean_", "Min_", "Precipitation", "CloudCover"))
            or c in ("TrendState", "TrendDE")
        ]
        for col in weather_like:
            if df[col].isna().any():
                if "Store" in df.columns:
                    df[col] = df.groupby("Store")[col].ffill().bfill()
                df[col] = df[col].fillna(df[col].median())

        if "StateHoliday" in df.columns:
            df["StateHoliday"] = df["StateHoliday"].fillna("0").astype(str)

        remaining = int(df.isna().sum().sum())
        if remaining:
            logger.warning("%d NaNs remain after cleaning", remaining)
        return df


class SalesOutlierHandler(BaseEstimator, TransformerMixin):
    """Flag, and optionally cap, per-store sales outliers.

    Outliers are measured *within each store* because absolute turnover varies
    by an order of magnitude across the estate; a global threshold would label
    every day of the busiest stores an outlier.

    Parameters
    ----------
    strategy:
        ``"flag"``  -- add an ``IsSalesOutlier`` column, change nothing else
                       (the default; preserves genuine seasonal peaks).
        ``"cap"``   -- winsorise to the per-store IQR fence.
        ``"remove"``-- drop the offending rows.
    iqr_multiplier:
        Width of the Tukey fence. 3.0 (rather than the usual 1.5) keeps normal
        seasonal trading inside the fence.
    """

    def __init__(self, strategy: str = "flag", iqr_multiplier: float = 3.0):
        self.strategy = strategy
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        if self.strategy not in {"flag", "cap", "remove"}:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        open_rows = X[X.get("Open", 1) == 1]
        grouped = open_rows.groupby("Store")["Sales"]
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.iqr_multiplier * iqr).clip(lower=0)
        self.upper_ = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()
        lower = df["Store"].map(self.lower_)
        upper = df["Store"].map(self.upper_)

        is_open = df.get("Open", pd.Series(1, index=df.index)) == 1
        outlier = is_open & ((df["Sales"] < lower) | (df["Sales"] > upper))
        df["IsSalesOutlier"] = outlier.astype(int)
        logger.info(
            "detected %d sales outliers (%.2f%% of open days) via per-store IQR x%.1f",
            int(outlier.sum()),
            100 * outlier.sum() / max(int(is_open.sum()), 1),
            self.iqr_multiplier,
        )

        if self.strategy == "cap":
            df.loc[outlier, "Sales"] = df.loc[outlier, "Sales"].clip(
                lower=lower[outlier], upper=upper[outlier]
            )
            logger.info("capped %d outliers to per-store IQR fence", int(outlier.sum()))
        elif self.strategy == "remove":
            before = len(df)
            df = df[~outlier].reset_index(drop=True)
            logger.info("removed %d outlier rows (%d -> %d)", before - len(df), before, len(df))

        return df


def drop_closed_and_zero_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only genuinely trading days for model training.

    Closed days are ~17% of the data and always have ``Sales == 0``. Training
    on them teaches the model to predict zero; instead we drop them here and
    re-impose ``Sales = 0`` at inference whenever ``Open == 0``, which is a
    deterministic business rule rather than something worth learning.

    A further 54 rows are open with zero turnover -- data-entry artefacts that
    would otherwise act as extreme low outliers.
    """
    before = len(df)
    mask = (df["Open"] == 1) & (df["Sales"] >= MIN_TRAINABLE_SALES)
    out = df[mask].reset_index(drop=True)
    logger.info(
        "kept %d trading rows of %d (dropped %d closed/zero-sales)",
        len(out),
        before,
        before - len(out),
    )
    return out


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tabulate missing values by column -- used in the EDA notebook."""
    n = len(df)
    counts = df.isna().sum()
    report = pd.DataFrame(
        {
            "missing": counts,
            "pct": (100 * counts / n).round(3),
            "dtype": df.dtypes.astype(str),
        }
    )
    return report[report["missing"] > 0].sort_values("missing", ascending=False)


def outlier_report(df: pd.DataFrame, column: str = "Sales") -> pd.DataFrame:
    """Per-store outlier counts and the z-score of the worst offender."""
    open_rows = df[df.get("Open", 1) == 1]
    g = open_rows.groupby("Store")[column]
    stats = pd.DataFrame({"mean": g.mean(), "std": g.std(), "max": g.max()})
    stats["max_zscore"] = ((stats["max"] - stats["mean"]) / stats["std"]).round(2)
    return stats.sort_values("max_zscore", ascending=False)


def clean_dataset(
    df: pd.DataFrame,
    outlier_strategy: str = "flag",
    for_training: bool = True,
) -> pd.DataFrame:
    """Run the full cleaning sequence on a joined dataset.

    ``for_training=False`` skips the row-dropping and outlier steps, which are
    only valid when ``Sales`` is present.
    """
    logger.info("cleaning dataset (%d rows, for_training=%s)", len(df), for_training)
    df = MissingValueCleaner().fit_transform(df)

    if for_training and "Sales" in df.columns:
        df = drop_closed_and_zero_sales(df)
        df = SalesOutlierHandler(strategy=outlier_strategy).fit_transform(df)

    logger.info("cleaning complete: %d rows, %d columns", len(df), df.shape[1])
    return df
