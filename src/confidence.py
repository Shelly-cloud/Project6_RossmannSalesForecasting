"""Post-prediction analysis: feature importance and confidence intervals (Task 2.4).

Confidence interval approach
-----------------------------
A Random Forest already trains ~100 independent trees on bootstrap samples of
the data. Rather than fit a second model for uncertainty, we reuse those trees
directly: each tree gives its own prediction for a row, and the spread of
those predictions across trees is a direct, essentially free estimate of the
model's uncertainty about that row.

    interval = percentile([tree.predict(x) for tree in forest], [2.5, 97.5])

This is the "creative, low-cost" interval the brief asks for. It captures
*model* uncertainty (how much the trees disagree) rather than full predictive
uncertainty (which would also include irreducible noise), so it should be
read as "how confident the forest is in its own estimate," not as a full
prediction interval. It still does the practical job: wider intervals flag
rows -- new stores, holidays, unusual promo combinations -- where the
forecast should be trusted less.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.logger import get_logger

logger = get_logger(__name__)


def tree_prediction_interval(
    pipe: Pipeline, X_raw: pd.DataFrame, lower: float = 2.5, upper: float = 97.5
) -> pd.DataFrame:
    """Per-row prediction interval from the spread of individual tree outputs.

    Runs the pipeline's preprocessing once, then queries every tree in the
    forest on the transformed matrix directly -- far cheaper than calling
    ``pipe.predict`` per tree, which would re-run preprocessing each time.
    """
    model = pipe.named_steps["model"]
    if not isinstance(model, RandomForestRegressor):
        raise TypeError(
            f"tree_prediction_interval requires a RandomForestRegressor, got {type(model)}"
        )

    # Replay every step except the final estimator to get the model-ready matrix.
    X = X_raw
    for name, step in pipe.steps[:-1]:
        X = step.transform(X)

    tree_preds = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)

    result = pd.DataFrame(
        {
            "prediction": tree_preds.mean(axis=0),
            "lower": np.percentile(tree_preds, lower, axis=0),
            "upper": np.percentile(tree_preds, upper, axis=0),
            "tree_std": tree_preds.std(axis=0),
        },
        index=X_raw.index,
    )
    logger.info(
        "computed %.0f-%.0f%% intervals for %d rows (mean width %.3f)",
        lower, upper, len(result), (result["upper"] - result["lower"]).mean(),
    )
    return result


def tree_prediction_interval_sales(
    pipe: Pipeline, X_raw: pd.DataFrame, lower: float = 2.5, upper: float = 97.5
) -> pd.DataFrame:
    """Same interval, but undoing the log1p target transform back to Sales units.

    Because log1p is monotonic, percentiles computed in log space are still
    valid percentiles after transforming back -- expm1 of a percentile is the
    corresponding percentile on the original scale. ``tree_std`` is a spread
    *in log space*, not a point value, so expm1 does not apply to it -- it is
    carried through unchanged, renamed to make that explicit.
    """
    log_result = tree_prediction_interval(pipe, X_raw, lower, upper)
    result = log_result[["prediction", "lower", "upper"]].apply(np.expm1)
    result["tree_std_log"] = log_result["tree_std"]
    return result


def permutation_importance_report(
    pipe: Pipeline, X_raw: pd.DataFrame, y: np.ndarray, n_repeats: int = 5,
    random_state: int = 42, max_samples: int = 20000,
) -> pd.DataFrame:
    """Permutation importance on the *raw* feature columns.

    Preferred over ``RandomForestRegressor.feature_importances_``, which is
    biased toward high-cardinality numeric columns (it counts how often a
    feature is used to split, not how much it actually helps). Permutation
    importance instead measures the drop in score when a column's values are
    shuffled, so it reflects genuine predictive contribution and is comparable
    across a mix of numeric and categorical columns.
    """
    from sklearn.inspection import permutation_importance

    y = np.asarray(y)
    if len(X_raw) > max_samples:
        # Sample by integer position so X and y -- which may not share an
        # index (y is often a bare numpy array) -- stay aligned row-for-row.
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_raw), size=max_samples, replace=False)
        X_raw = X_raw.iloc[idx]
        y = y[idx]

    result = permutation_importance(
        pipe, X_raw, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    report = pd.DataFrame(
        {
            "feature": X_raw.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    logger.info("permutation importance computed for %d raw features", len(report))
    return report


def model_feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """Fast built-in Gini/gain importance on the encoded feature space.

    Cheap to compute and useful as a sanity cross-check against permutation
    importance, but reported second because of the high-cardinality bias
    noted above.
    """
    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = model.feature_importances_
    return pd.DataFrame(
        {"feature": names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
