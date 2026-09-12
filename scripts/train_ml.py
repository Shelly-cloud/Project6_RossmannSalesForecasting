"""Task 2 runner: preprocessing -> sklearn pipeline -> loss -> post-prediction
analysis -> serialization -> MLflow (sections 2.1-2.5, 2.7).

Trains two models behind the identical cleaning/feature-engineering pipeline
so only the final estimator differs -- a direct comparison, and it produces
the multiple MLflow runs / model versions the interim submission asks for:

  * RandomForestRegressor  -- the brief's suggested starting point
  * LightGBM               -- gradient-boosted trees, the "innovative
                               approach" upgrade; typically stronger and
                               faster to train on tabular data like this

Both train on log1p(Sales) and are scored back on the real Sales scale.
Validation is the **final 6 weeks of the training window**, matching the
brief's forecast horizon -- not a random split, which would leak each
store's own future statistics into its past.
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
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from src import viz
from src.cleaning import clean_dataset
from src.config import RANDOM_STATE, REPORTS_DIR, VALIDATION_WEEKS
from src.confidence import (
    model_feature_importance,
    permutation_importance_report,
    tree_prediction_interval_sales,
)
from src.data_loader import load_dataset
from src.features import build_holiday_calendar
from src.logger import get_logger
from src.metrics import evaluate
from src.mlflow_utils import init_mlflow
from src.pipeline import fit_full_pipeline, get_feature_names
from src.serialize import save_model

logger = get_logger("train_ml")
viz.apply_theme()

logger.info("=" * 70)
logger.info("TASK 2 - PREDICTION OF STORE SALES (machine learning models)")
logger.info("=" * 70)

# ---------------------------------------------------------------- 2.1 data
train_raw = load_dataset("train")
test_raw = load_dataset("test")
calendar = build_holiday_calendar(train_raw, test_raw)

clean = clean_dataset(train_raw, outlier_strategy="flag", for_training=True)

cutoff = clean["Date"].max() - pd.Timedelta(weeks=VALIDATION_WEEKS)
tr = clean[clean["Date"] <= cutoff].reset_index(drop=True)
va = clean[clean["Date"] > cutoff].reset_index(drop=True)
logger.info(
    "chronological split at %s: train %d rows / valid %d rows (%d weeks)",
    cutoff.date(), len(tr), len(va), VALIDATION_WEEKS,
)

# Train on log1p(Sales): right-skewed target (Q14), and squared error in log
# space approximates the relative error that RMSPE measures (src/metrics.py).
y_tr_log = np.log1p(tr["Sales"].values)
y_va = va["Sales"].values

init_mlflow()

MODEL_CONFIGS = [
    {
        "name": "random_forest",
        "estimator": RandomForestRegressor(
            n_estimators=200,
            max_depth=22,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "params": {
            "n_estimators": 200, "max_depth": 22,
            "min_samples_leaf": 2, "max_features": "sqrt",
        },
    },
    {
        "name": "lightgbm",
        "estimator": LGBMRegressor(
            n_estimators=600,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "params": {
            "n_estimators": 600, "num_leaves": 63,
            "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        },
    },
]

results = {}

for cfg in MODEL_CONFIGS:
    name = cfg["name"]
    logger.info("-" * 70)
    logger.info("training model: %s", name)

    with mlflow.start_run(run_name=name):
        mlflow.log_params(cfg["params"])
        mlflow.log_param("validation_weeks", VALIDATION_WEEKS)
        mlflow.log_param("target_transform", "log1p")
        mlflow.log_param("train_rows", len(tr))
        mlflow.log_param("valid_rows", len(va))

        t0 = time.time()
        pipe = fit_full_pipeline(tr, y_tr_log, model=cfg["estimator"], holiday_calendar=calendar)
        fit_seconds = time.time() - t0
        logger.info("%s fitted in %.1fs", name, fit_seconds)
        mlflow.log_metric("fit_seconds", fit_seconds)

        # ---- 2.3/2.4 evaluate on the held-out 6 weeks -----------------
        pred_log = pipe.predict(va)
        pred = np.expm1(pred_log)
        metrics = evaluate(y_va, pred, prefix="valid")
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        logger.info("%s validation metrics: %s", name,
                    {k: round(v, 4) for k, v in metrics.items()})

        n_features = len(get_feature_names(pipe))
        mlflow.log_param("n_encoded_features", n_features)

        # ---- 2.4 feature importance ------------------------------------
        built_in = model_feature_importance(pipe)
        built_in_path = REPORTS_DIR / f"feature_importance_builtin_{name}.csv"
        built_in.to_csv(built_in_path, index=False)
        mlflow.log_artifact(str(built_in_path))

        perm = permutation_importance_report(
            pipe, va.drop(columns=[]), y_va, n_repeats=3, max_samples=8000
        )
        perm_path = REPORTS_DIR / f"feature_importance_permutation_{name}.csv"
        perm.to_csv(perm_path, index=False)
        mlflow.log_artifact(str(perm_path))
        logger.info("%s top-5 permutation-importance features:\n%s",
                    name, perm.head(5).to_string())

        fig, ax = plt.subplots(figsize=(8, 6))
        top = perm.head(15).iloc[::-1]
        ax.barh(top["feature"], top["importance_mean"],
                xerr=top["importance_std"], color=viz.SERIES[0],
                edgecolor=viz.SURFACE, linewidth=2, error_kw={"ecolor": viz.INK_MUTED})
        viz.finish(ax, f"{name}: top 15 features by permutation importance",
                   "Drop in -RMSE when the column is shuffled; error bars = across repeats",
                   "", "Importance")
        fig.tight_layout()
        fig_path = viz.save(fig, f"feature_importance_{name}")
        mlflow.log_artifact(fig_path)
        plt.close(fig)

        # ---- 2.4 confidence intervals (RandomForest only) --------------
        if isinstance(cfg["estimator"], RandomForestRegressor):
            intervals = tree_prediction_interval_sales(pipe, va.head(5000))
            coverage = float(
                (
                    (y_va[:5000] >= intervals["lower"].values)
                    & (y_va[:5000] <= intervals["upper"].values)
                ).mean()
            )
            mean_width = float((intervals["upper"] - intervals["lower"]).mean())
            mlflow.log_metric("interval_95_coverage", coverage)
            mlflow.log_metric("interval_95_mean_width", mean_width)
            logger.info(
                "%s: 95%% tree-spread interval empirical coverage %.1f%%, mean width %.0f",
                name, 100 * coverage, mean_width,
            )

            interval_path = REPORTS_DIR / f"prediction_intervals_{name}.csv"
            out = intervals.copy()
            out["actual"] = y_va[:5000]
            out["Store"] = va["Store"].values[:5000]
            out["Date"] = va["Date"].values[:5000]
            out.to_csv(interval_path, index=False)
            mlflow.log_artifact(str(interval_path))

        # ---- 2.5 serialize with timestamp -------------------------------
        model_path = save_model(
            pipe, model_name=name,
            metadata={"metrics": metrics, "fit_seconds": fit_seconds,
                      "n_features": n_features, "train_rows": len(tr)},
        )
        mlflow.log_param("serialized_path", str(model_path))
        mlflow.sklearn.log_model(pipe, name="model")

        results[name] = {"pipe": pipe, "metrics": metrics, "path": model_path}

# ---------------------------------------------------------------- summary
logger.info("=" * 70)
logger.info("MODEL COMPARISON")
for name, r in results.items():
    logger.info("  %-14s RMSPE=%.4f  RMSE=%.1f  MAE=%.1f  R2=%.4f",
                name, r["metrics"]["valid_rmspe"], r["metrics"]["valid_rmse"],
                r["metrics"]["valid_mae"], r["metrics"]["valid_r2"])

best_name = min(results, key=lambda n: results[n]["metrics"]["valid_rmspe"])
logger.info("best model by validation RMSPE: %s", best_name)

print("\nTASK 2 (ML) COMPLETE")
for name, r in results.items():
    m = r["metrics"]
    print(f"  {name:14s} RMSPE={m['valid_rmspe']:.4f}  RMSE={m['valid_rmse']:.1f}  "
          f"MAE={m['valid_mae']:.1f}  MAPE={m['valid_mape']:.4f}  R2={m['valid_r2']:.4f}")
    print(f"    saved -> {r['path']}")
print(f"  best model: {best_name}")
print(f"\n  MLflow UI: mlflow ui --backend-store-uri \"{Path('mlruns').resolve().as_uri()}\"")
