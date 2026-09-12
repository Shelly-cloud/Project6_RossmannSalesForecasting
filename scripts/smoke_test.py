"""Fast end-to-end check of loader -> cleaning -> features -> pipeline.

Runs on a 60-store subsample so it finishes in seconds. Purpose is to catch
shape/dtype/leakage errors before committing to a full training run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.cleaning import clean_dataset
from src.config import RANDOM_STATE, VALIDATION_WEEKS
from src.data_loader import load_dataset
from src.features import build_holiday_calendar
from src.logger import get_logger
from src.metrics import evaluate
from src.pipeline import EXCLUDED_FROM_FEATURES, fit_full_pipeline, get_feature_names

logger = get_logger("smoke_test")

train = load_dataset("train")
test = load_dataset("test")

calendar = build_holiday_calendar(train, test)
logger.info("holiday calendar spans %s -> %s", calendar.min().date(), calendar.max().date())

# Subsample stores so the smoke test is quick.
rng = np.random.RandomState(RANDOM_STATE)
stores = rng.choice(sorted(train.Store.unique()), size=60, replace=False)
sub = train[train.Store.isin(stores)].copy()
logger.info("subsampled %d stores -> %d rows", len(stores), len(sub))

clean = clean_dataset(sub, outlier_strategy="flag", for_training=True)

# Chronological split: last 6 weeks held out, mirroring the real 6-week horizon.
cutoff = clean.Date.max() - pd.Timedelta(weeks=VALIDATION_WEEKS)
tr = clean[clean.Date <= cutoff]
va = clean[clean.Date > cutoff]
logger.info("split at %s -> train %d / valid %d", cutoff.date(), len(tr), len(va))

# Train on log1p(Sales): the target is right-skewed and squared error in log
# space approximates relative error, which is what RMSPE measures.
y_tr = np.log1p(tr.Sales.values)
y_va = va.Sales.values

pipe = fit_full_pipeline(
    tr, y_tr,
    model=RandomForestRegressor(
        n_estimators=40, min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=RANDOM_STATE,
    ),
    holiday_calendar=calendar,
)

pred = np.expm1(pipe.predict(va))
metrics = evaluate(y_va, pred, prefix="valid")
logger.info("metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

names = get_feature_names(pipe)
logger.info("model sees %d features after encoding", len(names))

# --- leakage assertions -------------------------------------------------
leaked = [n for n in names if any(bad in n for bad in ("Sales", "Customers"))]
store_agg_ok = [n for n in leaked if n.startswith("Store") or "Index" in n]
unexpected = [n for n in leaked if n not in store_agg_ok]
assert not unexpected, f"unexpected target-derived features reached model: {unexpected}"
for bad in ("Customers", "Sales", "IsSalesOutlier"):
    assert bad not in names, f"{bad} leaked into features"
logger.info("leakage check passed (store-history aggregates allowed: %d)", len(store_agg_ok))

# --- test-set inference check ------------------------------------------
test_clean = clean_dataset(test, for_training=False)
test_pred = np.expm1(pipe.predict(test_clean.head(2000)))
logger.info(
    "test inference OK: %d preds, min %.0f, mean %.0f, max %.0f",
    len(test_pred), test_pred.min(), test_pred.mean(), test_pred.max(),
)

print("\nSMOKE TEST PASSED")
print(f"  valid RMSPE : {metrics['valid_rmspe']:.4f}")
print(f"  valid RMSE  : {metrics['valid_rmse']:.1f}")
print(f"  valid MAE   : {metrics['valid_mae']:.1f}")
print(f"  valid R2    : {metrics['valid_r2']:.4f}")
print(f"  features    : {len(names)}")
