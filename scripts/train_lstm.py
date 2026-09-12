"""Task 2.6 runner: LSTM deep-learning forecaster.

Follows the brief's seven steps end to end and writes the diagnostic charts
(stationarity, ACF/PACF, training curve, prediction-vs-actual) to
reports/figures so they can go straight into the write-up.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src import viz
from src.cleaning import clean_dataset
from src.config import RANDOM_STATE, REPORTS_DIR
from src.data_loader import load_dataset
from src.lstm_model import (
    acf_pacf,
    build_lstm,
    build_store_dataset,
    difference_if_needed,
    inverse_scale_per_store,
    isolate_series,
)
from src.logger import get_logger
from src.metrics import evaluate
from src.mlflow_utils import init_mlflow
from src.serialize import timestamp_now
from src.config import MODELS_DIR

logger = get_logger("train_lstm")
viz.apply_theme()
np.random.seed(RANDOM_STATE)

logger.info("=" * 70)
logger.info("TASK 2.6 - DEEP LEARNING (LSTM) FORECASTER")
logger.info("=" * 70)

train_raw = load_dataset("train")
clean = clean_dataset(train_raw, outlier_strategy="flag", for_training=True)

# ---- Step 1: isolate a time series -----------------------------------
national = isolate_series(clean, store=None)
logger.info("isolated national daily-sales series: %d points, %s -> %s",
            len(national), national.index.min().date(), national.index.max().date())

# ---- Steps 2-3: stationarity + differencing ----------------------------
series_for_acf, was_differenced = difference_if_needed(national, label="national daily sales")

# ---- Step 4: ACF / PACF -------------------------------------------------
acf_table = acf_pacf(series_for_acf, nlags=21)
acf_table.to_csv(REPORTS_DIR / "lstm_acf_pacf.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
ax.plot(national.index, national.values, color=viz.SERIES[0], linewidth=1.2)
viz.finish(ax, "National daily sales, raw series",
           f"{'Non-' if was_differenced else ''}stationary by ADF test",
           "Total sales", "Date")

ax = axes[1]
ax.bar(acf_table["lag"], acf_table["acf"], color=viz.SERIES[0], width=0.6)
ax.axhline(0, color=viz.BASELINE, linewidth=1)
for lag in (7, 14, 21):
    ax.axvline(lag, color=viz.GRIDLINE, linewidth=1, linestyle="--", zorder=0)
viz.finish(ax, "Autocorrelation (ACF)",
           "Dashed lines at lag 7/14/21 -- the weekly cycle", "ACF", "Lag (days)")

ax = axes[2]
ax.bar(acf_table["lag"], acf_table["pacf"], color=viz.SERIES[1], width=0.6)
ax.axhline(0, color=viz.BASELINE, linewidth=1)
viz.finish(ax, "Partial autocorrelation (PACF)",
           "Direct lag dependence after removing shorter lags", "PACF", "Lag (days)")
fig.tight_layout()
viz.save(fig, "lstm_stationarity_acf_pacf")
plt.close(fig)

# Window length: ACF/PACF both show the dominant structure at lag 7 (the
# weekly cycle); a 14-day window covers two full cycles, giving the network
# both this week's and last week's same-weekday value to work from.
WINDOW = 14
VAL_DAYS = 42  # the brief's 6-week forecast horizon

logger.info("ACF/PACF support a weekly cycle (lag ~7) -> using a %d-day window", WINDOW)

# ---- Steps 5-6: per-store windowing + (-1, 1) scaling -------------------
rng = np.random.RandomState(RANDOM_STATE)
all_stores = sorted(clean["Store"].unique())
sample_stores = list(rng.choice(all_stores, size=min(120, len(all_stores)), replace=False))

data = build_store_dataset(clean, sample_stores, window=WINDOW, val_days=VAL_DAYS)
logger.info(
    "training tensor %s, validation tensor %s, stores used %d",
    data["X_train"].shape, data["X_val"].shape, len(data["scalers"]),
)

# ---- Step 7: build and train the LSTM -----------------------------------
model = build_lstm(window=WINDOW, units=(50, 25))
model.summary(print_fn=lambda line: logger.info(line))

init_mlflow()

with mlflow.start_run(run_name="lstm"):
    mlflow.log_params({
        "window": WINDOW, "val_days": VAL_DAYS, "n_stores": len(data["scalers"]),
        "lstm_units_1": 50, "lstm_units_2": 25, "scale_range": "(-1, 1)",
        "differenced_for_diagnostics": was_differenced,
    })

    t0 = time.time()
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=25, batch_size=256, verbose=0,
        callbacks=[
            __import__("tensorflow").keras.callbacks.EarlyStopping(
                patience=4, restore_best_weights=True
            )
        ],
    )
    fit_seconds = time.time() - t0
    logger.info("LSTM trained in %.1fs over %d epochs", fit_seconds, len(history.history["loss"]))
    mlflow.log_metric("fit_seconds", fit_seconds)

    for epoch, (loss, val_loss) in enumerate(
        zip(history.history["loss"], history.history["val_loss"])
    ):
        mlflow.log_metrics({"train_loss": loss, "val_loss": val_loss}, step=epoch)

    # ---- evaluate back on the real Sales scale --------------------------
    pred_scaled = model.predict(data["X_val"], verbose=0).flatten()

    val_stores = []
    for store, dates in data["val_meta"]:
        val_stores.extend([store] * len(dates))
    val_stores = val_stores[: len(pred_scaled)]

    pred_sales = inverse_scale_per_store(pred_scaled, val_stores, data["scalers"])
    actual_sales = inverse_scale_per_store(data["y_val"], val_stores, data["scalers"])

    metrics = evaluate(actual_sales, pred_sales, prefix="valid")
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    logger.info("LSTM validation metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    ts = timestamp_now()
    model_path = MODELS_DIR / f"lstm_{ts}.keras"
    model.save(model_path)
    mlflow.log_param("serialized_path", str(model_path))
    logger.info("saved LSTM -> %s", model_path)

# ---- diagnostic charts ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

ax = axes[0]
ax.plot(history.history["loss"], color=viz.SERIES[0], label="Train")
ax.plot(history.history["val_loss"], color=viz.SERIES[1], label="Validation")
ax.legend()
viz.finish(ax, "Training curve", "MSE loss in scaled (-1, 1) space", "Loss", "Epoch")

ax = axes[1]
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(pred_sales), size=min(300, len(pred_sales)), replace=False)
ax.scatter(actual_sales[sample_idx], pred_sales[sample_idx], s=10, alpha=0.4,
           color=viz.SERIES[0], edgecolors="none")
lims = [0, max(actual_sales[sample_idx].max(), pred_sales[sample_idx].max())]
ax.plot(lims, lims, color=viz.BASELINE, linestyle="--", linewidth=1.2)
viz.finish(ax, f"Predicted vs actual (RMSPE={metrics['valid_rmspe']:.3f})",
           "300 sampled validation points across all stores", "Predicted sales", "Actual sales")
fig.tight_layout()
viz.save(fig, "lstm_training_and_fit")
plt.close(fig)

print("\nTASK 2.6 (LSTM) COMPLETE")
print(f"  stores used     : {len(data['scalers'])}")
print(f"  window          : {WINDOW} days")
print(f"  train sequences : {len(data['X_train']):,}")
print(f"  valid sequences : {len(data['X_val']):,}")
for k, v in metrics.items():
    print(f"  {k:14s}: {v:.4f}")
print(f"  saved -> {model_path}")
