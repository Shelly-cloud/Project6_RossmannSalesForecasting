"""LSTM deep-learning forecaster (Task 2.6).

Follows the brief's seven steps in order:

1. Isolate the data as a time series
2. Test stationarity (Augmented Dickey-Fuller)
3. Difference if the test calls for it
4. Inspect ACF / PACF to choose a lookback window
5. Turn the series into supervised (window -> next-value) examples
6. Scale to (-1, 1), the range that matches the LSTM's tanh activations
7. Train a shallow (two-layer) LSTM regressor

Design choice: rather than one single aggregate series (which would give only
~900 training points -- too little to fit a network with any confidence), the
supervised windows are built **per store** and then stacked into one global
training set. Each store is scaled independently to (-1, 1) with its own
``MinMaxScaler`` before windowing, because Q12 showed store size varies
sixfold; a single global scaler would let the largest stores dominate the
loss exactly as RMSE would (src/metrics.py). Store identity is not fed to the
network -- only the windowed sales history is -- so the model is a genuine
sequence-to-one forecaster, not a lookup table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf, adfuller, pacf

from src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------- steps 1-4
def isolate_series(df: pd.DataFrame, store: int | None = None) -> pd.Series:
    """Step 1: extract a single daily sales series.

    ``store=None`` aggregates total network sales per day (used for the
    stationarity/ACF diagnostics, where one clean series is easier to read).
    A specific store id isolates that store's own trading history instead.
    """
    work = df[df["Sales"] > 0]
    if store is not None:
        work = work[work["Store"] == store]
        series = work.set_index("Date")["Sales"].sort_index()
    else:
        series = work.groupby("Date")["Sales"].sum().sort_index()
    series.index = pd.DatetimeIndex(series.index)
    return series


def test_stationarity(series: pd.Series, label: str = "series") -> dict:
    """Step 2: Augmented Dickey-Fuller test.

    Null hypothesis: the series has a unit root (is non-stationary). A
    p-value below 0.05 rejects that, i.e. the series is stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pvalue, n_lags, n_obs = result[0], result[1], result[2], result[3]
    is_stationary = pvalue < 0.05
    logger.info(
        "ADF on %s: statistic=%.4f p=%.4g lags=%d -> %s",
        label, stat, pvalue, n_lags, "stationary" if is_stationary else "non-stationary",
    )
    return {
        "statistic": stat, "pvalue": pvalue, "n_lags": n_lags,
        "n_obs": n_obs, "is_stationary": is_stationary,
    }


def difference_if_needed(series: pd.Series, label: str = "series") -> tuple[pd.Series, bool]:
    """Steps 2-3: test, and first-difference only if the test says to.

    Returns ``(series_to_use, was_differenced)``.
    """
    result = test_stationarity(series, label)
    if result["is_stationary"]:
        logger.info("%s is already stationary (p=%.4g) -- no differencing applied",
                    label, result["pvalue"])
        return series, False

    diffed = series.diff().dropna()
    result2 = test_stationarity(diffed, f"{label} (1st difference)")
    logger.info(
        "%s was non-stationary (p=%.4g); after first-differencing p=%.4g -> %s",
        label, result["pvalue"], result2["pvalue"],
        "stationary" if result2["is_stationary"] else "still non-stationary",
    )
    return diffed, True


def acf_pacf(series: pd.Series, nlags: int = 21) -> pd.DataFrame:
    """Step 4: ACF and PACF values, used to pick the LSTM lookback window.

    Retail sales have a strong 7-day cycle, so lags around 7 and 14 are the
    ones to read off this table when choosing the window length.
    """
    acf_vals = acf(series.dropna(), nlags=nlags, fft=True)
    pacf_vals = pacf(series.dropna(), nlags=nlags)
    return pd.DataFrame({"lag": range(nlags + 1), "acf": acf_vals, "pacf": pacf_vals})


# ---------------------------------------------------------------- steps 5-7
def make_windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Step 5: slide a fixed window over a 1-D array to build (X, y) pairs.

    ``X[i]`` is ``values[i : i+window]`` and ``y[i]`` is the value
    immediately following that window -- the standard sliding-window
    transform from time series into a supervised learning problem.
    """
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i : i + window])
        y.append(values[i + window])
    return np.array(X), np.array(y)


def build_store_dataset(
    df: pd.DataFrame, stores: list[int], window: int = 14, val_days: int = 42,
) -> dict:
    """Steps 5-6 across many stores: per-store (-1, 1) scaling, then windowing.

    Each store is split chronologically (last ``val_days`` held out) *before*
    scaling and windowing, so no validation-period value leaks into a
    training window and no scaler is fit on validation data.
    """
    X_train, y_train, X_val, y_val = [], [], [], []
    scalers: dict[int, MinMaxScaler] = {}
    val_meta = []

    for store in stores:
        series = isolate_series(df, store=store)
        if len(series) < window + val_days + 10:
            continue  # not enough history for a meaningful window + holdout

        train_part = series.iloc[: -val_days]
        val_part = series.iloc[-val_days - window :]  # extra `window` for context

        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_scaled = scaler.fit_transform(train_part.values.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val_part.values.reshape(-1, 1)).flatten()

        Xt, yt = make_windows(train_scaled, window)
        Xv, yv = make_windows(val_scaled, window)

        if len(Xt) == 0 or len(Xv) == 0:
            continue

        X_train.append(Xt)
        y_train.append(yt)
        X_val.append(Xv)
        y_val.append(yv)
        scalers[store] = scaler
        val_meta.append((store, val_part.index[window:]))

    X_train = np.concatenate(X_train)[..., np.newaxis]
    y_train = np.concatenate(y_train)
    X_val = np.concatenate(X_val)[..., np.newaxis]
    y_val = np.concatenate(y_val)

    logger.info(
        "built windowed dataset: %d stores, window=%d, train=%d sequences, val=%d sequences",
        len(scalers), window, len(X_train), len(X_val),
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "scalers": scalers, "val_meta": val_meta, "window": window,
    }


def build_lstm(window: int, units: tuple[int, int] = (50, 25)) -> "tf.keras.Model":
    """Step 7: a two-layer LSTM regressor.

    Kept intentionally shallow per the brief ("should not be very deep") so it
    trains comfortably on a CPU or a free Colab instance. The first layer
    returns full sequences so the second LSTM layer has a sequence to
    consume; the second returns only the final hidden state, which is fed
    through a small dense head to the single-value forecast.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(window, 1)),
            layers.LSTM(units[0], return_sequences=True),
            layers.LSTM(units[1]),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def inverse_scale_per_store(
    y_scaled: np.ndarray, stores: list[int], scalers: dict[int, MinMaxScaler]
) -> np.ndarray:
    """Undo the per-store (-1, 1) scaling, one row at a time.

    Needed because every store carries its own scaler -- there is no single
    global inverse transform to apply to the stacked array.
    """
    out = np.empty_like(y_scaled, dtype=float)
    for i, store in enumerate(stores):
        out[i] = scalers[store].inverse_transform([[y_scaled[i]]])[0, 0]
    return out
