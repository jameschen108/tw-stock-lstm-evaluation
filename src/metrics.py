"""
Shared evaluation helpers.

Two reasons this module exists:

1. It keeps one definition of the RMSE/MAE/R2 block, which several entry
   points need and would otherwise drift apart.
2. More importantly, it provides the **naive persistence baseline**.
   Daily closing prices are highly autocorrelated, so "tomorrow's close ==
   today's close" already scores R2 ~ 0.99 on its own. Reporting a model's
   R2 without that reference number makes a useless model look excellent,
   which is the single most common way stock-prediction projects fool
   themselves. Every metric table in this project now shows the baseline
   next to the model.
"""
import math
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, pred) -> Dict[str, float]:
    """RMSE / MAE / R2 for a pair of 1-D (or column-vector) price series."""
    actual = np.asarray(actual).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    return {
        "RMSE": math.sqrt(mean_squared_error(actual, pred)),
        "MAE": mean_absolute_error(actual, pred),
        "R2": r2_score(actual, pred),
    }


def naive_forecast(refs: np.ndarray) -> np.ndarray:
    """Persistence baseline: predict the next close as the last observed close.

    ``refs`` is the per-window reference price from ``prepare_dataset`` -
    i.e. the close on the decision day - so "no change" *is* that price.
    In the model's normalised return space this baseline is the constant 0.
    """
    return np.asarray(refs, dtype="float64").reshape(-1)


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """Fraction of the baseline's error the model removes.

    > 0 means the model beats persistence, 0 means it merely matches it,
    < 0 means you would have been better off predicting no change at all.
    """
    if baseline_rmse == 0:
        return 0.0
    return 1.0 - (model_rmse / baseline_rmse)


def direction_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Accuracy of an up/down call, against the baselines that matter.

    Reporting bare accuracy invites the reader to compare it with 50%, but
    50% is the wrong reference: equity returns are mildly up-biased, so
    "always predict up" already scores the base rate. ``majority_accuracy``
    is that free strategy, and ``edge_vs_majority`` is what the model adds
    on top of it. ``p_value`` is a two-sided binomial test of the model's
    hit count against the majority rate - with a few hundred test days,
    swings of several percentage points are routine noise.
    """
    y_true = np.asarray(y_true).reshape(-1)
    pred = (np.asarray(y_prob).reshape(-1) > threshold).astype(float)

    n = len(y_true)
    hits = int((pred == y_true).sum())
    accuracy = hits / n if n else float("nan")

    up_rate = float(y_true.mean()) if n else float("nan")
    majority_accuracy = max(up_rate, 1.0 - up_rate)

    p_value = float("nan")
    if n:
        from scipy.stats import binomtest
        p_value = float(binomtest(hits, n, majority_accuracy).pvalue)

    return {
        "accuracy": accuracy,
        "up_rate": up_rate,
        "majority_accuracy": majority_accuracy,
        "edge_vs_majority": accuracy - majority_accuracy,
        "p_value": p_value,
        "n_samples": float(n),
        "predicted_up_rate": float(pred.mean()) if n else float("nan"),
    }


def mean_and_ci(values, confidence: float = 0.95) -> Dict[str, float]:
    """Mean of repeated runs with a t-based confidence interval.

    Every experiment in this project is repeated across seeds, because a
    single training run of a neural net on noisy financial data is not a
    measurement: on this data the same configuration produces a different
    answer on every seed.
    """
    values = np.asarray(values, dtype="float64").reshape(-1)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean, "n_runs": float(n)}

    from scipy.stats import t
    std = float(values.std(ddof=1))
    half = float(t.ppf(0.5 + confidence / 2, df=n - 1)) * std / np.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "n_runs": float(n),
    }


def format_comparison(model_name: str, model: Dict[str, float], baseline: Dict[str, float]) -> str:
    """Render a model-vs-baseline metric table for the console."""
    lines = [
        "=" * 58,
        f"{'':<18}{'RMSE':<12}{'MAE':<12}{'R2':<12}",
        "-" * 58,
        f"{model_name:<18}{model['RMSE']:<12.4f}{model['MAE']:<12.4f}{model['R2']:<12.4f}",
        f"{'naive (persist)':<18}{baseline['RMSE']:<12.4f}{baseline['MAE']:<12.4f}{baseline['R2']:<12.4f}",
        "-" * 58,
    ]
    skill = skill_score(model["RMSE"], baseline["RMSE"])
    verdict = "beats" if skill > 0 else "LOSES TO"
    lines.append(f"RMSE skill score: {skill:+.4f}  ({model_name} {verdict} the naive baseline)")
    lines.append("=" * 58)
    return "\n".join(lines)
