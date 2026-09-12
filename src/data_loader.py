"""Raw data loading and joining (train/test + store + locality extras).

The three Kaggle files give us sales, store attributes and the forecast rows.
The fast.ai mirror ships three extra files that let us model the *locality*
factor the brief calls out explicitly:

* ``store_states.csv``  -> which German state each store sits in
* ``weather.csv``       -> daily weather per state
* ``googletrend.csv``   -> weekly Google search interest in "Rossmann" per state

Joining those is optional (``with_external=False`` skips it) so the core
pipeline still runs if only the three Kaggle CSVs are present.
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    GOOGLETREND_CSV,
    RAW_DTYPES,
    STATE_NAMES_CSV,
    STORE_CSV,
    STORE_STATES_CSV,
    TEST_CSV,
    TRAIN_CSV,
    WEATHER_CSV,
)
from src.logger import get_logger, log_dataframe

logger = get_logger(__name__)

# Weather has 24 columns, most of them near-duplicates (three humidity
# readings, three pressure readings...). We keep the handful that plausibly
# change shopping behaviour and drop the rest to avoid feature bloat.
# store_states.csv treats Bremen + Lower Saxony as one region ("HB,NI"); the
# Google-trend file only publishes a series for "NI". Lower Saxony dominates
# that combined region by population, so borrowing its series is the sensible
# approximation.
TREND_STATE_MAP = {"HB,NI": "NI"}

WEATHER_KEEP = [
    "Max_TemperatureC",
    "Mean_TemperatureC",
    "Min_TemperatureC",
    "Mean_Humidity",
    "Mean_Wind_SpeedKm_h",
    "Precipitationmm",
    "CloudCover",
]


def load_train() -> pd.DataFrame:
    """Load train.csv with Date parsed and StateHoliday forced to string."""
    df = pd.read_csv(
        TRAIN_CSV, dtype=RAW_DTYPES, parse_dates=["Date"], low_memory=False
    )
    log_dataframe(logger, df, "loaded train.csv")
    return df


def load_test() -> pd.DataFrame:
    """Load test.csv (no Sales/Customers columns, carries an Id row key)."""
    df = pd.read_csv(TEST_CSV, dtype=RAW_DTYPES, parse_dates=["Date"])
    log_dataframe(logger, df, "loaded test.csv")
    return df


def load_store() -> pd.DataFrame:
    """Load the static per-store attribute table."""
    df = pd.read_csv(STORE_CSV)
    log_dataframe(logger, df, "loaded store.csv")
    return df


def _load_state_lookup() -> pd.DataFrame:
    """Store -> State code, plus the long-form StateName used by weather.csv."""
    store_states = pd.read_csv(STORE_STATES_CSV)
    state_names = pd.read_csv(STATE_NAMES_CSV)
    return store_states.merge(state_names, on="State", how="left")


def _load_weather() -> pd.DataFrame:
    """Daily weather per state, keyed on (StateName, Date)."""
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["Date"])
    weather = weather.rename(columns={"file": "StateName"})
    cols = ["StateName", "Date"] + [c for c in WEATHER_KEEP if c in weather.columns]
    return weather[cols]


def _load_googletrend() -> pd.DataFrame:
    """Weekly Google-trend index per state, expanded to a week-start date.

    The ``file`` column looks like ``Rossmann_DE_SN``; the suffix is the state
    code. ``Rossmann_DE`` (no suffix) is the national series, which we keep
    separately as a country-wide trend feature.
    """
    trend = pd.read_csv(GOOGLETREND_CSV)
    # "2012-12-02 - 2012-12-08" -> take the start of the week
    trend["WeekStart"] = pd.to_datetime(trend["week"].str.split(" - ").str[0])
    trend["State"] = trend["file"].str.replace("Rossmann_DE_?", "", regex=True)

    national = (
        trend.loc[trend["State"] == "", ["WeekStart", "trend"]]
        .rename(columns={"trend": "TrendDE"})
        .drop_duplicates("WeekStart")
    )
    per_state = trend.loc[trend["State"] != "", ["State", "WeekStart", "trend"]].rename(
        columns={"trend": "TrendState"}
    )
    return per_state, national


def merge_store(df: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Left-join the static store attributes onto a daily sales frame."""
    merged = df.merge(store, on="Store", how="left")
    if len(merged) != len(df):
        raise ValueError(
            f"store join changed row count {len(df)} -> {len(merged)}; "
            "store.csv should have one row per Store"
        )
    return merged


def merge_external(df: pd.DataFrame) -> pd.DataFrame:
    """Attach state, weather and Google-trend features. Requires a Date column."""
    states = _load_state_lookup()
    df = df.merge(states, on="Store", how="left")

    weather = _load_weather()
    df = df.merge(weather, on=["StateName", "Date"], how="left")

    per_state, national = _load_googletrend()

    # The trend file's week labels run Sunday-to-Saturday, so each row must be
    # snapped back to the *preceding Sunday*. Snapping to Monday (the pandas
    # default week start) misses every key and silently yields all-NaN trends.
    df["WeekStart"] = df["Date"] - pd.to_timedelta(
        (df["Date"].dt.dayofweek + 1) % 7, unit="D"
    )
    # store_states labels Bremen and Lower Saxony jointly as "HB,NI", but the
    # trend file only carries "NI". Map across so those stores get a trend.
    df["TrendStateCode"] = df["State"].replace(TREND_STATE_MAP)

    df = df.merge(
        per_state.rename(columns={"State": "TrendStateCode"}),
        on=["TrendStateCode", "WeekStart"],
        how="left",
    )
    df = df.merge(national, on="WeekStart", how="left")
    df = df.drop(columns=["WeekStart", "StateName", "TrendStateCode"])
    return df


def load_dataset(kind: str = "train", with_external: bool = True) -> pd.DataFrame:
    """Load and fully join either the train or the test dataset.

    Parameters
    ----------
    kind:
        ``"train"`` or ``"test"``.
    with_external:
        Join the locality extras (state / weather / Google trend). Set False to
        run on the three Kaggle CSVs alone.
    """
    if kind not in {"train", "test"}:
        raise ValueError(f"kind must be 'train' or 'test', got {kind!r}")

    logger.info("loading '%s' dataset (with_external=%s)", kind, with_external)
    df = load_train() if kind == "train" else load_test()
    df = merge_store(df, load_store())

    if with_external:
        df = merge_external(df)

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    log_dataframe(logger, df, f"joined {kind} dataset")
    return df
