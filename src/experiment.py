"""
Ablation study: does the feature engineering actually buy anything?

Research question
-----------------
Nine technical and macro features (Gaussian trend, trend slope, RSI, MACD
histogram, Bollinger position, Bias, VIX, S&P 500, USD/TWD) are standard
additions to a price model. Whether they help is an empirical question, and
answering it needs a single-feature control plus enough repetition to tell
an effect from seed noise.

Design
------
Three arms, identical in every respect except the input width:

    close      Close price only                    (control)
    technical  + 6 technical indicators
    full       + 3 macro series

Both tasks are measured, because they fail in different ways:

* **regression** - next-day close. Scored against the naive persistence
  baseline via a skill score, since R^2 alone is ~0.997 for doing nothing.
* **classification** - next-day direction. Scored against the majority-class
  baseline ("always predict up"), with a binomial test, since equity
  returns are up-biased and 50% is the wrong reference point.

Every (arm x task x ticker) cell is repeated over ``config.EXPERIMENT_SEEDS``
and reported as a mean with a 95% confidence interval. A result that does
not survive its own seed variance is not a result.

Usage:
    python -m src.experiment                      # full study, 5 tickers
    python -m src.experiment --ticker 2330.TW     # one ticker
    python -m src.experiment --seeds 0 1 2 --epochs 20
"""
import argparse
import json
from datetime import date

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from . import config, scaling
from .data import prepare_dataset
from .metrics import (
    direction_metrics, mean_and_ci, naive_forecast, regression_metrics, skill_score,
)
from .model import CLASSIFICATION, REGRESSION, build_model
from .train import set_seeds


def _run_once(ds, model_type: str, task: str, seed: int, epochs: int, batch_size: int) -> dict:
    """Train one model on one seed and score it on the held-out tail."""
    set_seeds(seed)

    y_train = ds["y_train"]
    if task == CLASSIFICATION:
        y_train = scaling.direction_labels(y_train)

    model = build_model(
        model_type,
        seq_length=ds["seq_length"],
        n_features=ds["n_features"],
        task=task,
    )
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(
        ds["X_train"], y_train,
        epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop],
    )

    raw = model.predict(ds["X_test"], verbose=0)
    refs = ds["refs_test"]

    if task == CLASSIFICATION:
        truth = scaling.direction_labels(ds["y_test"])
        return direction_metrics(truth, raw)

    preds = scaling.to_price(raw, refs)
    actual = scaling.to_price(ds["y_test"], refs)
    scores = regression_metrics(actual, preds)
    baseline = regression_metrics(actual, naive_forecast(refs))
    scores["naive_RMSE"] = baseline["RMSE"]
    scores["skill_vs_naive"] = skill_score(scores["RMSE"], baseline["RMSE"])
    return scores


# The headline number for each task: what the arm is judged on.
HEADLINE = {REGRESSION: "skill_vs_naive", CLASSIFICATION: "edge_vs_majority"}


def run_study(
    tickers=None,
    feature_sets=None,
    tasks=(REGRESSION, CLASSIFICATION),
    model_type: str = "LSTM",
    seeds=config.EXPERIMENT_SEEDS,
    start: str = config.START_DATE,
    end: str = None,
    epochs: int = 20,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    standardise: bool = False,
):
    tickers = list(tickers or config.PORTFOLIO_CONFIG)
    feature_sets = list(feature_sets or config.FEATURE_SETS)

    rows = []
    for ticker in tickers:
        # One download / windowing per (ticker, feature set); the seeds
        # differ only in model initialisation, never in the data.
        for feature_set in feature_sets:
            print(f"[experiment] {ticker} / {feature_set} ...", flush=True)
            ds = prepare_dataset(
                ticker, start=start, end=end, feature_set=feature_set,
                standardise=standardise,
            )
            for task in tasks:
                for seed in seeds:
                    scores = _run_once(ds, model_type, task, seed, epochs, batch_size)
                    rows.append({
                        "ticker": ticker,
                        "feature_set": feature_set,
                        "n_features": ds["n_features"],
                        "standardise": standardise,
                        "task": task,
                        "seed": seed,
                        "test_days": len(ds["X_test"]),
                        **scores,
                    })

    return pd.DataFrame(rows)


def ablation_contrasts(runs: pd.DataFrame, control: str = "close") -> pd.DataFrame:
    """Paired comparison of each arm against the control arm.

    This is the actual research question, and it needs a *paired* test:
    every arm is run on the same tickers with the same seeds, so comparing
    the pooled means throws away that pairing and inflates the variance.
    Differencing within each (ticker, seed) cell removes both the
    between-ticker and between-seed noise, leaving only the effect of the
    features themselves.

    A confidence interval spanning zero means the extra features made no
    measurable difference - which is a finding, not a failed experiment.
    """
    from scipy import stats

    out = []
    for task, task_runs in runs.groupby("task", sort=False):
        metric = "accuracy" if task == CLASSIFICATION else HEADLINE[task]
        wide = task_runs.pivot_table(
            index=["ticker", "seed"], columns="feature_set", values=metric
        )
        if control not in wide:
            continue

        for arm in wide.columns:
            if arm == control:
                continue
            paired = wide[[arm, control]].dropna()
            diff = (paired[arm] - paired[control]).to_numpy()
            n = len(diff)
            result = stats.ttest_rel(paired[arm], paired[control])
            low, high = stats.t.interval(
                0.95, n - 1, loc=diff.mean(), scale=stats.sem(diff)
            ) if n > 1 and diff.std() > 0 else (diff.mean(), diff.mean())

            out.append({
                "task": task,
                "metric": metric,
                "arm": arm,
                "control": control,
                "mean_difference": float(diff.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value": float(result.pvalue),
                "n_pairs": n,
                "verdict": "no measurable effect" if low <= 0 <= high else (
                    "arm better" if diff.mean() > 0 else "arm worse"
                ),
            })
    return pd.DataFrame(out)


def architecture_contrasts(
    runs_by_model: dict, feature_set: str = config.DEFAULT_FEATURE_SET
) -> pd.DataFrame:
    """Paired comparison between architectures on one feature set.

    The prior work this project builds on selected an LSTM because it scored
    best of LSTM/GRU/CNN on one stock. That is a ranking read off a single
    run, and the seed variance documented throughout this project is easily
    large enough to produce such a ranking by chance. Since every
    architecture is run on the same tickers with the same seeds, the honest
    comparison is paired on (ticker, seed) - the same argument
    ``ablation_contrasts`` makes for feature sets.

    ``runs_by_model`` maps model name -> that model's ``runs`` sheet. Only
    architectures whose runs share tickers and seeds can be paired, so the
    intersection is used and ``n_pairs`` reports what survived.
    """
    from itertools import combinations

    from scipy import stats

    out = []
    for task, metric in ((REGRESSION, "skill_vs_naive"), (CLASSIFICATION, "accuracy")):
        series = {}
        for name, runs in runs_by_model.items():
            block = runs[(runs["task"] == task) & (runs["feature_set"] == feature_set)]
            if len(block):
                series[name] = block.set_index(["ticker", "seed"])[metric]

        for a, b in combinations(series, 2):
            paired = pd.concat(
                [series[a].rename("a"), series[b].rename("b")], axis=1
            ).dropna()
            n = len(paired)
            if n < 2:
                continue
            diff = (paired["a"] - paired["b"]).to_numpy()
            result = stats.ttest_rel(paired["a"], paired["b"])
            low, high = stats.t.interval(
                0.95, n - 1, loc=diff.mean(), scale=stats.sem(diff)
            ) if diff.std() > 0 else (diff.mean(), diff.mean())

            out.append({
                "task": task,
                "metric": metric,
                "feature_set": feature_set,
                "model_a": a,
                "model_b": b,
                "mean_difference": float(diff.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value": float(result.pvalue),
                "n_pairs": n,
                "verdict": "indistinguishable" if low <= 0 <= high else (
                    f"{a} better" if diff.mean() > 0 else f"{b} better"
                ),
            })
    return pd.DataFrame(out)


def summarise_by_ticker(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker means with seed-only confidence intervals.

    The pooled summary mixes two sources of variation - between tickers and
    between random seeds - into one interval. This breakdown isolates the
    seed variance, which is the one that tells you whether a single training
    run means anything.
    """
    out = []
    for (ticker, feature_set, task), group in runs.groupby(
        ["ticker", "feature_set", "task"], sort=False
    ):
        metric = HEADLINE[task]
        stats = mean_and_ci(group[metric].values)
        out.append({
            "ticker": ticker,
            "feature_set": feature_set,
            "task": task,
            "metric": metric,
            "mean": stats["mean"],
            "std_across_seeds": stats["std"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "n_runs": int(stats["n_runs"]),
        })
    return pd.DataFrame(out)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-seed runs into means with 95% confidence intervals."""
    out = []
    for (feature_set, task), group in runs.groupby(["feature_set", "task"], sort=False):
        metric = HEADLINE[task]
        stats = mean_and_ci(group[metric].values)
        row = {
            "feature_set": feature_set,
            "task": task,
            "n_features": int(group["n_features"].iloc[0]),
            "metric": metric,
            "mean": stats["mean"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "std": stats["std"],
            "n_runs": int(stats["n_runs"]),
        }
        if task == CLASSIFICATION:
            row["mean_accuracy"] = float(group["accuracy"].mean())
            row["majority_accuracy"] = float(group["majority_accuracy"].mean())
            row["median_p_value"] = float(group["p_value"].median())
        out.append(row)
    return pd.DataFrame(out)


def _print_summary(summary: pd.DataFrame, runs: pd.DataFrame) -> None:
    for task in summary["task"].unique():
        block = summary[summary["task"] == task]
        metric = HEADLINE[task]
        print("\n" + "=" * 84)
        print(f"TASK: {task}   (headline metric: {metric}; >0 beats the baseline)")
        print("-" * 84)
        if task == CLASSIFICATION:
            header = f"{'feature set':<12}{'k':>3}{'accuracy':>11}{'majority':>11}{'edge':>10}{'95% CI':>20}{'p':>8}"
        else:
            header = f"{'feature set':<12}{'k':>3}{'skill':>11}{'95% CI':>24}{'std':>10}"
        print(header)
        for _, r in block.iterrows():
            ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            if task == CLASSIFICATION:
                print(f"{r['feature_set']:<12}{r['n_features']:>3}{r['mean_accuracy']:>11.4f}"
                      f"{r['majority_accuracy']:>11.4f}{r['mean']:>+10.4f}{ci:>20}"
                      f"{r['median_p_value']:>8.3f}")
            else:
                print(f"{r['feature_set']:<12}{r['n_features']:>3}{r['mean']:>+11.4f}{ci:>24}{r['std']:>10.4f}")
        print("=" * 84)

    print("\nA confidence interval spanning 0 means the arm is indistinguishable "
          "from its baseline.")
    print(f"Runs: {len(runs)} = "
          f"{runs['ticker'].nunique()} tickers x {runs['feature_set'].nunique()} feature sets "
          f"x {runs['task'].nunique()} tasks x {runs['seed'].nunique()} seeds")


def _print_contrasts(contrasts: pd.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("ABLATION CONTRASTS - each arm minus the close-only control, paired by (ticker, seed)")
    print("-" * 92)
    print(f"{'task':<16}{'arm':<12}{'metric':<18}{'difference':>12}{'95% CI':>24}{'p':>8}")
    for _, r in contrasts.iterrows():
        ci = f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]"
        print(f"{r['task']:<16}{r['arm']:<12}{r['metric']:<18}"
              f"{r['mean_difference']:>+12.5f}{ci:>24}{r['p_value']:>8.3f}")
    print("=" * 92)
    for _, r in contrasts.iterrows():
        print(f"  {r['task']}/{r['arm']}: {r['verdict']}")


def _compare_architectures(paths) -> None:
    """Print the paired architecture table in README section 3.2."""
    import glob
    import re

    runs_by_model = {}
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            match = re.search(r"Ablation_([A-Za-z]+)[_\.]", str(path))
            name = match.group(1) if match else str(path)
            runs_by_model[name] = pd.read_excel(path, sheet_name="runs")

    if len(runs_by_model) < 2:
        raise SystemExit(
            "need at least two Ablation_<MODEL>_*.xlsx reports to compare; "
            "run `python -m src.experiment --model-type GRU` first."
        )

    contrasts = architecture_contrasts(runs_by_model)
    print("=" * 96)
    print("ARCHITECTURE CONTRASTS - paired on (ticker, seed), close-only arm")
    print("-" * 96)
    print(f"{'task':<16}{'contrast':<18}{'difference':>13}{'95% CI':>26}{'p':>8}  verdict")
    for _, r in contrasts.iterrows():
        ci = f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]"
        print(f"{r['task']:<16}{r['model_a'] + ' - ' + r['model_b']:<18}"
              f"{r['mean_difference']:>+13.5f}{ci:>26}{r['p_value']:>8.3f}  {r['verdict']}")
    print("=" * 96)
    if (contrasts["verdict"] == "indistinguishable").all():
        print("No pair is distinguishable: an architecture ranking taken from a single")
        print("stock and a single window is measuring run-to-run noise, not model quality.")


def main():
    parser = argparse.ArgumentParser(description="Run the feature ablation study.")
    parser.add_argument(
        "--compare-architectures", nargs="*", metavar="REPORT",
        help="Skip training; print the paired architecture table (README 3.2) from "
             "existing Ablation_<MODEL>_*.xlsx reports. Defaults to all of them.",
    )
    parser.add_argument("--ticker", action="append", dest="tickers",
                        help="Repeatable; defaults to the whole portfolio.")
    parser.add_argument("--feature-set", action="append", dest="feature_sets",
                        choices=sorted(config.FEATURE_SETS))
    parser.add_argument("--task", action="append", dest="tasks",
                        choices=[REGRESSION, CLASSIFICATION])
    parser.add_argument("--model-type", default="LSTM", choices=config.VALID_MODEL_TYPES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.EXPERIMENT_SEEDS))
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--standardise", action="store_true",
                        help="Z-score each input channel, fitted on training rows only. "
                             "Tests whether the multi-feature arms fail because the "
                             "features carry no signal or because their scales differ "
                             "by two orders of magnitude.")
    parser.add_argument("--output", default=None, help="Output .xlsx path")
    args = parser.parse_args()

    if args.compare_architectures is not None:
        _compare_architectures(
            args.compare_architectures
            or [str(config.REPORT_DIR / "Ablation_LSTM_2*.xlsx"),
                str(config.REPORT_DIR / "Ablation_GRU_*.xlsx"),
                str(config.REPORT_DIR / "Ablation_CNN_*.xlsx")]
        )
        return

    runs = run_study(
        tickers=args.tickers,
        feature_sets=args.feature_sets,
        tasks=tuple(args.tasks) if args.tasks else (REGRESSION, CLASSIFICATION),
        model_type=args.model_type,
        seeds=args.seeds,
        start=args.start,
        end=args.end,
        epochs=args.epochs,
        standardise=args.standardise,
    )
    summary = summarise(runs)

    suffix = "_standardised" if args.standardise else ""
    output_path = args.output or (
        config.REPORT_DIR /
        f"Ablation_{args.model_type}{suffix}_{date.today().isoformat()}.xlsx"
    )
    meta = {
        "model_type": args.model_type,
        "seeds": json.dumps(args.seeds),
        "epochs": args.epochs,
        "start": args.start,
        "end": args.end or date.today().isoformat(),
        "seq_length": config.SEQ_LENGTH,
        "split_ratio": config.SPLIT_RATIO,
        "macro_lag_days": config.MACRO_LAG_DAYS,
        "standardise": args.standardise,
    }
    contrasts = ablation_contrasts(runs)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        contrasts.to_excel(writer, sheet_name="contrasts", index=False)
        summarise_by_ticker(runs).to_excel(writer, sheet_name="by_ticker", index=False)
        runs.to_excel(writer, sheet_name="runs", index=False)
        pd.DataFrame([meta]).to_excel(writer, sheet_name="run_config", index=False)
    print(f"\n[experiment] results saved -> {output_path}")

    _print_summary(summary, runs)
    _print_contrasts(contrasts)


if __name__ == "__main__":
    main()
