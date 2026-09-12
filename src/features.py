"""Feature engineering (Task 2.1).

Covers every feature the brief asks for -- weekdays, weekends, days to/after a
holiday, month phase -- plus the "think of more features" extras:

* calendar parts (year, month, day, ISO week, day-of-year, quarter)
* cyclical sin/cos encodings so December sits next to January
* payday proximity (German salaries land at month end -> a real demand driver)
* months since the nearest competitor opened, and whether one exists at all
* ``PromoInterval`` decoded into a genuine per-row ``IsPromo2Active`` flag
* per-store historical aggregates (mean/median sales, sales per customer)
* per-store-per-weekday aggregates, which capture each store's own trading
  rhythm far better than a global DayOfWeek effect

The aggregates are learned in ``fit`` from the training rows only, so the
6-week validation fold never sees its own statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.logger import get_logger

logger = get_logger(__name__)

MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
}


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Expand ``Date`` into calendar, month-phase and cyclical features."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()
        d = pd.to_datetime(df["Date"])

        df["Year"] = d.dt.year
        df["Month"] = d.dt.month
        df["Day"] = d.dt.day
        df["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
        df["DayOfYear"] = d.dt.dayofyear
        df["Quarter"] = d.dt.quarter

        # DayOfWeek ships with the data (1=Mon .. 7=Sun); recompute so the
        # column is guaranteed consistent even on hand-built dashboard input.
        df["DayOfWeek"] = d.dt.dayofweek + 1
        df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)
        df["IsSunday"] = (df["DayOfWeek"] == 7).astype(int)

        # Month phase, as requested: beginning / mid / end of month.
        df["DaysInMonth"] = d.dt.days_in_month
        df["IsMonthStart"] = (df["Day"] <= 10).astype(int)
        df["IsMidMonth"] = ((df["Day"] > 10) & (df["Day"] <= 20)).astype(int)
        df["IsMonthEnd"] = (df["Day"] > 20).astype(int)
        df["MonthPhase"] = np.select(
            [df["Day"] <= 10, df["Day"] <= 20], [0, 1], default=2
        )

        # German wages are typically paid at month end, so the days either side
        # of the boundary carry a spending spike that day-of-month alone blurs.
        df["DaysToMonthEnd"] = df["DaysInMonth"] - df["Day"]
        df["IsPayWeek"] = (
            (df["DaysToMonthEnd"] <= 3) | (df["Day"] <= 3)
        ).astype(int)

        # Cyclical encodings: without these, month 12 and month 1 look 11 units
        # apart to the model instead of adjacent.
        df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
        df["DayOfWeekSin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeekCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfYearSin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
        df["DayOfYearCos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)

        # A single monotonic time index lets tree models express overall drift.
        df["TimeIndex"] = (d - pd.Timestamp("2013-01-01")).dt.days

        # Christmas run-up is the single largest seasonal effect in this data.
        df["IsDecember"] = (df["Month"] == 12).astype(int)
        df["DaysToChristmas"] = (
            pd.to_datetime(df["Year"].astype(str) + "-12-25") - d
        ).dt.days
        df["IsChristmasRunUp"] = (
            df["DaysToChristmas"].between(0, 21)
        ).astype(int)

        return df


class HolidayFeatureExtractor(BaseEstimator, TransformerMixin):
    """Add days-to-next-holiday and days-since-last-holiday.

    The holiday calendar is learned at ``fit`` from rows where
    ``StateHoliday != '0'``. Pass ``calendar`` explicitly to reuse a calendar
    built from train *and* test -- necessary because a forecast row needs to
    know about a holiday that falls after the training window ends.
    """

    def __init__(self, calendar: pd.DatetimeIndex | None = None):
        self.calendar = calendar

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        if self.calendar is not None:
            self.calendar_ = pd.DatetimeIndex(sorted(set(self.calendar)))
        else:
            holidays = pd.to_datetime(
                X.loc[X["StateHoliday"].astype(str) != "0", "Date"].unique()
            )
            self.calendar_ = pd.DatetimeIndex(sorted(holidays))
        logger.info("holiday calendar: %d distinct holiday dates", len(self.calendar_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()
        dates = pd.to_datetime(df["Date"])

        if len(self.calendar_) == 0:
            df["DaysToNextHoliday"] = 999
            df["DaysAfterLastHoliday"] = 999
        else:
            cal = self.calendar_.values.astype("datetime64[D]")
            target = dates.values.astype("datetime64[D]")

            # searchsorted gives the insertion point; the neighbours either
            # side of it are the next and previous holiday.
            idx = np.searchsorted(cal, target, side="left")

            nxt = np.where(
                idx < len(cal),
                (cal[np.clip(idx, 0, len(cal) - 1)] - target).astype("timedelta64[D]").astype(int),
                999,
            )
            prev_idx = idx - 1
            prv = np.where(
                prev_idx >= 0,
                (target - cal[np.clip(prev_idx, 0, len(cal) - 1)]).astype("timedelta64[D]").astype(int),
                999,
            )
            df["DaysToNextHoliday"] = np.clip(nxt, 0, 999)
            df["DaysAfterLastHoliday"] = np.clip(prv, 0, 999)

        df["IsStateHoliday"] = (df["StateHoliday"].astype(str) != "0").astype(int)
        # "Around a holiday" windows -- the brief asks about before/during/after.
        df["IsHolidayWeek"] = (
            (df["DaysToNextHoliday"] <= 3) | (df["DaysAfterLastHoliday"] <= 3)
        ).astype(int)
        df["IsDayBeforeHoliday"] = (df["DaysToNextHoliday"] == 1).astype(int)
        df["IsDayAfterHoliday"] = (df["DaysAfterLastHoliday"] == 1).astype(int)
        return df


class CompetitionPromoFeatures(BaseEstimator, TransformerMixin):
    """Turn the raw competition / Promo2 columns into usable signals."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()

        # --- competition -------------------------------------------------
        # Months elapsed since the nearest competitor opened. Zero year means
        # "predates our records", which we express as a long-established
        # competitor rather than a missing value.
        has_date = (df["CompetitionOpenSinceYear"] > 0)
        months_open = (
            12 * (df["Year"] - df["CompetitionOpenSinceYear"])
            + (df["Month"] - df["CompetitionOpenSinceMonth"])
        )
        df["CompetitionOpenMonths"] = np.where(has_date, months_open.clip(lower=0), 0)
        df["CompetitionIsNew"] = (
            has_date & months_open.between(0, 6)
        ).astype(int)
        df["CompetitionNotYetOpen"] = (has_date & (months_open < 0)).astype(int)

        # Distance is heavily right-skewed (70 m to 76 km), so a log transform
        # makes the "twice as far" relationship linear.
        df["CompetitionDistanceLog"] = np.log1p(df["CompetitionDistance"])
        # Very short distances proxy for dense city-centre locations -- used to
        # answer the "does distance matter in city centres?" question.
        df["IsCityCentre"] = (df["CompetitionDistance"] <= 500).astype(int)

        # --- Promo2 ------------------------------------------------------
        # PromoInterval names the months in which Promo2 restarts, e.g.
        # "Feb,May,Aug,Nov". A row is in an active Promo2 round only if its
        # month is listed AND the store had already joined by that date.
        month_abbr = df["Month"].map(MONTH_ABBR)
        interval = df["PromoInterval"].fillna("None").astype(str)
        month_in_interval = [
            (m in i.split(",")) if i not in ("None", "nan", "") else False
            for m, i in zip(month_abbr, interval)
        ]

        promo2_start = pd.to_datetime(
            df["Promo2SinceYear"].astype(int).astype(str)
            + df["Promo2SinceWeek"].astype(int).astype(str).str.zfill(2)
            + "1",
            format="%Y%W%w",
            errors="coerce",
        )
        started = pd.to_datetime(df["Date"]) >= promo2_start

        df["IsPromo2Active"] = (
            (df["Promo2"] == 1) & np.array(month_in_interval) & started.fillna(False)
        ).astype(int)

        df["Promo2ActiveWeeks"] = np.where(
            df["Promo2"] == 1,
            ((pd.to_datetime(df["Date"]) - promo2_start).dt.days / 7)
            .fillna(0)
            .clip(lower=0),
            0,
        )

        # Both promo mechanics firing at once -- worth its own interaction term.
        df["PromoAndPromo2"] = (
            (df["Promo"] == 1) & (df["IsPromo2Active"] == 1)
        ).astype(int)
        df["AnyPromo"] = (
            (df["Promo"] == 1) | (df["IsPromo2Active"] == 1)
        ).astype(int)

        return df


class StoreAggregateFeatures(BaseEstimator, TransformerMixin):
    """Per-store historical aggregates learned from the training fold only.

    These are the single strongest feature family for this problem: a store's
    own trading history predicts its future turnover better than any calendar
    or promo attribute. Fitting them on the training fold only is what keeps
    the validation score honest.
    """

    def __init__(self, include_customers: bool = True):
        self.include_customers = include_customers

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        open_rows = X[(X.get("Open", 1) == 1) & (X["Sales"] > 0)]

        g = open_rows.groupby("Store")["Sales"]
        self.store_stats_ = pd.DataFrame(
            {
                "StoreMeanSales": g.mean(),
                "StoreMedianSales": g.median(),
                "StoreStdSales": g.std().fillna(0),
                "StoreMaxSales": g.max(),
                "StoreOpenDays": g.size(),
            }
        )

        if self.include_customers and "Customers" in open_rows.columns:
            cg = open_rows.groupby("Store")["Customers"]
            self.store_stats_["StoreMeanCustomers"] = cg.mean()
            # Basket size: separates "busy but cheap" from "quiet but premium"
            # stores, and is the metric used to answer whether promos bring
            # new shoppers or just bigger baskets.
            self.store_stats_["StoreSalesPerCustomer"] = (
                open_rows.groupby("Store")
                .apply(lambda d: d["Sales"].sum() / max(d["Customers"].sum(), 1),
                       include_groups=False)
            )

        # Each store's own weekday rhythm.
        self.store_dow_mean_ = (
            open_rows.groupby(["Store", "DayOfWeek"])["Sales"].mean().rename("StoreDowMeanSales")
        )
        # Each store's own promo response -- also the basis of the promo
        # targeting recommendation in the EDA.
        self.store_promo_mean_ = (
            open_rows.groupby(["Store", "Promo"])["Sales"].mean().rename("StorePromoMeanSales")
        )
        self.store_month_mean_ = (
            open_rows.groupby(["Store", "Month"])["Sales"].mean().rename("StoreMonthMeanSales")
        )

        self.global_mean_ = float(open_rows["Sales"].mean())
        logger.info(
            "learned store aggregates for %d stores (global mean sales %.0f)",
            len(self.store_stats_),
            self.global_mean_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()

        df = df.merge(self.store_stats_, on="Store", how="left")
        df = df.merge(self.store_dow_mean_, on=["Store", "DayOfWeek"], how="left")
        df = df.merge(self.store_promo_mean_, on=["Store", "Promo"], how="left")
        df = df.merge(self.store_month_mean_, on=["Store", "Month"], how="left")

        # Stores present in test but absent from the training fold fall back to
        # the global mean rather than to NaN.
        agg_cols = list(self.store_stats_.columns) + [
            "StoreDowMeanSales",
            "StorePromoMeanSales",
            "StoreMonthMeanSales",
        ]
        for col in agg_cols:
            if col in df.columns and df[col].isna().any():
                fill = self.global_mean_ if "Sales" in col else df[col].median()
                df[col] = df[col].fillna(fill)

        # Relative indices: how this weekday/month compares with the store's
        # own baseline. Scale-free, so they transfer across stores.
        df["DowSalesIndex"] = df["StoreDowMeanSales"] / df["StoreMeanSales"].replace(0, np.nan)
        df["MonthSalesIndex"] = df["StoreMonthMeanSales"] / df["StoreMeanSales"].replace(0, np.nan)
        df[["DowSalesIndex", "MonthSalesIndex"]] = df[
            ["DowSalesIndex", "MonthSalesIndex"]
        ].fillna(1.0)

        return df


def build_holiday_calendar(*frames: pd.DataFrame) -> pd.DatetimeIndex:
    """Union of all state-holiday dates across the supplied frames.

    Built from train *and* test so that forecast rows know about holidays
    falling beyond the end of the training window.
    """
    dates: list[pd.Timestamp] = []
    for df in frames:
        if "StateHoliday" in df.columns and "Date" in df.columns:
            mask = df["StateHoliday"].astype(str) != "0"
            dates.extend(pd.to_datetime(df.loc[mask, "Date"].unique()))
    return pd.DatetimeIndex(sorted(set(dates)))


def engineer_features(
    df: pd.DataFrame,
    holiday_calendar: pd.DatetimeIndex | None = None,
    store_aggregates: StoreAggregateFeatures | None = None,
) -> tuple[pd.DataFrame, StoreAggregateFeatures | None]:
    """Apply the full stateless feature chain, plus aggregates if supplied.

    Returns the transformed frame and the (possibly newly fitted) aggregate
    transformer so it can be reused on the validation/test split.
    """
    out = DateFeatureExtractor().fit_transform(df)
    out = HolidayFeatureExtractor(calendar=holiday_calendar).fit_transform(out)
    out = CompetitionPromoFeatures().fit_transform(out)

    if store_aggregates is not None:
        out = store_aggregates.transform(out)
    elif "Sales" in out.columns:
        store_aggregates = StoreAggregateFeatures()
        out = store_aggregates.fit_transform(out)

    logger.info("engineered features: %d columns", out.shape[1])
    return out, store_aggregates
