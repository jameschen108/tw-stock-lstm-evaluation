"""
Capacity and training-length sensitivity.

Why this module exists
----------------------
Every result in this project comes from one architecture size: two
recurrent layers of 50 units, 20 epochs, a 10-day input window. A negative
result reported at a single operating point invites the obvious objection -
*the model simply was not trained hard enough, or was too small to learn
anything*. That objection is cheap to make and, without this module,
impossible to answer.

So the same close-only experiment is repeated across a grid of capacities,
training lengths and window lengths. The claim being tested is not "one of
these configurations is best" - picking a winner on the test set would
be model selection on the reporting data - but the weaker and more useful
"no configuration in this range beats the naive persistence baseline".

Design
------
* The grid is deliberately coarse and pre-declared (see ``GRID``), not
  searched. Nothing here is selected on test-set performance.
* Every cell is repeated over several seeds and reported as a mean with a
  95% confidence interval, like every other experiment in this project.
* Only the close-only arm is swept. The feature question is the ablation's
  job (``src/experiment.py``); this module asks about capacity alone.
* Windows are rebuilt per ``seq_length``, so a longer window means fewer
  samples - that is a real property of the design, not a bug.

Usage:
    python -m src.tuning
    python -m src.tuning --ticker 2330.TW --seeds 0 1 2
"""
import argparse
from datetime import date

import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from . import config, scaling
from .data import prepare_dataset
from .metrics import mean_and_ci, naive_forecast, regression_metrics, skill_score
from .model import build_model
from .train import set_seeds

# (units, epochs, seq_length). Pre-declared and coarse on purpose: this is a
# robustness check, not a hyperparameter search. The first row is the
# configuration every other experiment in the project uses.
GRID = (
    (50, 20, 10),      # the project default
    (16, 20, 10),      # a quarter of the capacity
    (128, 20, 10),     # two and a half times the capacity
    (50, 60, 10),      # three times the training budget
    (128, 60, 10),     # more capacity and more training
    (50, 20, 30),      # a month of history instead of two weeks
    (50, 60, 30),      # longer window, longer training
    (128, 60, 30),     # the largest cell in the grid
)


def run_cell(
    ticker: str,
    units: int,
    epochs: int,
    seq_length: int,
    seed: int,
    start: str = config.START_DATE,
    end: str = None,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
) -> dict:
    """Train one configuration on one seed, score it against persistence."""
    set_seeds(seed)

    ds = prepare_dataset(
        ticker, start=start, end=end, seq_length=seq_length,
        feature_set=config.DEFAULT_FEATURE_SET,
    )

    model = build_model(
        "LSTM", seq_length=seq_length, n_features=ds["n_features"], units=units
    )
    # Monitors training loss, never the test set - see src/experiment.py.
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(
        ds["X_train"], ds["y_train"],
        epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop],
    )

    refs = ds["refs_test"]
    preds = scaling.to_price(model.predict(ds["X_test"], verbose=0), refs)
    actual = scaling.to_price(ds["y_test"], refs)

    scores = regression_metrics(actual, preds)
    baseline = regression_metrics(actual, naive_forecast(refs))

    return {
        "ticker": ticker,
        "units": units,
        "epochs": epochs,
        "seq_length": seq_length,
        "seed": seed,
        "n_params": int(model.count_params()),
        "train_days": len(ds["X_train"]),
        "test_days": len(ds["X_test"]),
        "RMSE": scores["RMSE"],
        "R2": scores["R2"],
        "naive_RMSE": baseline["RMSE"],
        "skill_vs_naive": skill_score(scores["RMSE"], baseline["RMSE"]),
    }


def run_sweep(tickers=None, seeds=(0, 1, 2), grid=GRID, **kwargs) -> pd.DataFrame:
    tickers = list(tickers or config.PORTFOLIO_CONFIG)

    rows = []
    for units, epochs, seq_length in grid:
        for ticker in tickers:
            print(f"[tuning] units={units} epochs={epochs} seq={seq_length} "
                  f"/ {ticker} ...", flush=True)
            for seed in seeds:
                rows.append(run_cell(ticker, units, epochs, seq_length, seed, **kwargs))
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per grid cell: mean skill with a 95% confidence interval."""
    out = []
    for (units, epochs, seq_length), group in runs.groupby(
        ["units", "epochs", "seq_length"], sort=False
    ):
        stats = mean_and_ci(group["skill_vs_naive"].values)
        out.append({
            "units": units,
            "epochs": epochs,
            "seq_length": seq_length,
            "n_params": int(group["n_params"].iloc[0]),
            "skill_vs_naive": stats["mean"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "std": stats["std"],
            "n_runs": int(stats["n_runs"]),
            "beats_naive": bool(stats["ci_low"] > 0),
        })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(
        description="Check whether the negative result survives a range of capacities."
    )
    parser.add_argument("--ticker", action="append", dest="tickers",
                        help="Repeatable; defaults to the whole portfolio.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=None, help="Output .xlsx path")
    args = parser.parse_args()

    runs = run_sweep(
        tickers=args.tickers, seeds=tuple(args.seeds),
        start=args.start, end=args.end,
    )
    summary = summarise(runs)

    output_path = args.output or (
        config.REPORT_DIR / f"Tuning_LSTM_{date.today().isoformat()}.xlsx"
    )
    meta = {
        "model_type": "LSTM",
        "feature_set": config.DEFAULT_FEATURE_SET,
        "seeds": str(list(args.seeds)),
        "start": args.start,
        "end": args.end or date.today().isoformat(),
        "split_ratio": config.SPLIT_RATIO,
        "grid": str(GRID),
    }
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        runs.to_excel(writer, sheet_name="runs", index=False)
        pd.DataFrame([meta]).to_excel(writer, sheet_name="run_config", index=False)
    print(f"\n[tuning] results saved -> {output_path}")

    print("\n" + "=" * 92)
    print("CAPACITY SENSITIVITY - close-only arm, skill vs the naive persistence baseline")
    print(f"{len(args.seeds)} seeds x {runs['ticker'].nunique()} tickers per cell; "
          "a positive lower bound would mean the model beats doing nothing")
    print("-" * 92)
    print(f"{'units':>6}{'epochs':>8}{'seq':>6}{'params':>10}{'skill':>12}{'95% CI':>26}{'beats naive':>14}")
    for _, r in summary.iterrows():
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"{r['units']:>6}{r['epochs']:>8}{r['seq_length']:>6}{r['n_params']:>10}"
              f"{r['skill_vs_naive']:>+12.4f}{ci:>26}{str(r['beats_naive']):>14}")
    print("=" * 92)
    if not summary["beats_naive"].any():
        print("No configuration in this grid beats the naive baseline, so the negative")
        print("result is not an artefact of the capacity that was chosen.")


if __name__ == "__main__":
    main()
