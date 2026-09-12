"""Loss functions and evaluation metrics (Task 2.3).

Chosen primary metric: RMSPE (Root Mean Square Percentage Error)
----------------------------------------------------------------
    RMSPE = sqrt( mean( ((y - yhat) / y)^2 ) )

Why this one, and not plain RMSE:

1. **Scale invariance across a heterogeneous estate.** Daily turnover in this
   dataset spans roughly 3,000 to 40,000+ across stores. Under RMSE, a 10%
   miss on a flagship store contributes ~100x more squared error than the same
   10% miss on a small store, so the optimiser effectively ignores small
   stores. The finance team forecasts *every* store, so an error measure that
   treats them equally is the one that matches the business objective.

2. **Direct interpretability for the stakeholder.** "Our forecasts are within
   12% on average" is a sentence a finance analyst can act on. "RMSE is 840"
   requires knowing each store's baseline before it means anything.

3. **Squared, not absolute.** Keeping the square penalises the occasional
   badly-wrong forecast more than many slightly-wrong ones. For inventory and
   staffing planning a single large miss is genuinely more costly than spread
   noise, so that asymmetry is desirable.

4. **It is the metric the original Kaggle competition scored on**, which makes
   our numbers comparable with published benchmarks (top solutions ~0.10).

Caveat we handle explicitly: the ratio is undefined at ``y == 0``. Closed days
are excluded from training entirely and forced to zero at inference by a
business rule, so those rows never legitimately enter the metric. ``rmspe``
masks them rather than silently producing ``inf``.

Reported alongside RMSPE for a fuller picture: RMSE and MAE in currency units,
MAPE as the linear-penalty counterpart, and R^2 for variance explained.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score


def rmspe(y_true, y_pred) -> float:
    """Root Mean Square Percentage Error. Rows with ``y_true == 0`` are masked."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not mask.any():
        return float("nan")

    pct_err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct_err**2)))


def rmspe_expm1(y_true_log, y_pred_log) -> float:
    """RMSPE computed after undoing a ``log1p`` target transform.

    We train on ``log1p(Sales)`` (see ``train_ml.py``): the target is strongly
    right-skewed, and optimising squared error in log space is close to
    optimising *relative* error in the original space, which is exactly what
    RMSPE measures. This helper scores such a model on the real sales scale.
    """
    return rmspe(np.expm1(y_true_log), np.expm1(y_pred_log))


def rmse(y_true, y_pred) -> float:
    """Root Mean Square Error, in currency units."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error, in currency units -- the easiest figure to quote."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error. Linear-penalty counterpart to RMSPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def evaluate(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """Full metric bundle for logging to MLflow and printing in notebooks."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}rmspe": rmspe(y_true, y_pred),
        f"{p}rmse": rmse(y_true, y_pred),
        f"{p}mae": mae(y_true, y_pred),
        f"{p}mape": mape(y_true, y_pred),
        f"{p}r2": float(r2_score(y_true, y_pred)),
    }


# Negated because sklearn maximises scorers; lower RMSPE is better.
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
