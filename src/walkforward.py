"""
Walk-forward (rolling-origin) evaluation.

Why the single split is not enough
----------------------------------
``backtest.py`` trains once on the first 80% of history and evaluates on the
last 20%. Every number it produces is therefore a statement about one
particular stretch of market - here, a strong bull run from January 2024.
A strategy can look good or bad on one window purely by luck of the regime,
and the reader has no way to tell which.

Walk-forward fixes that by moving the origin. History is cut into ``n_folds``
consecutive test blocks; for each block the model is retrained on everything
that came before it and then evaluated on the block. Concatenating the blocks
gives one long out-of-sample series in which **every prediction was made by a
model that had never seen the day it is predicting**, spanning several years
and several regimes rather than one.

Two training modes:

* ``expanding`` (default) - the training set grows with each fold, mimicking
  a practitioner who retrains periodically and keeps all history.
* ``rolling`` - the training window is a fixed length that slides forward,
  which tests whether old data still helps or merely dilutes.

Usage:
    python -m src.walkforward
    python -m src.walkforward --feature-set technical --folds 6
    python -m src.walkforward --mode rolling --seeds 0 1 2
"""
import argparse
from datetime import date
from typing import List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from . import config, scaling
from .backtest import evaluate_portfolio
from .data import prepare_windows
from .metrics import direction_metrics, mean_and_ci, naive_forecast, regression_metrics, skill_score
from .model import build_model
from .train import set_seeds

EXPANDING = "expanding"
ROLLING = "rolling"
VALID_MODES = (EXPANDING, ROLLING)


def walk_forward_splits(
    n_samples: int,
    n_folds: int = config.WF_FOLDS,
    min_train_ratio: float = config.WF_MIN_TRAIN_RATIO,
    mode: str = EXPANDING,
) -> List[Tuple[int, int, int]]:
    """Return ``(train_start, train_end, test_end)`` index triples.

    The first ``min_train_ratio`` of the data is reserved to train the very
    first model; what remains is divided into ``n_folds`` equal test blocks
    (the last block absorbs any remainder). Training always ends exactly
    where the test block begins, so no fold can see its own future.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

    min_train = int(n_samples * min_train_ratio)
    remaining = n_samples - min_train
    if remaining < n_folds:
        raise ValueError(
            f"{n_samples} samples with min_train_ratio={min_train_ratio} leaves only "
            f"{remaining} test rows, too few for {n_folds} folds."
        )

    block = remaining // n_folds
    splits = []
    for i in range(n_folds):
        train_end = min_train + i * block
        test_end = n_samples if i == n_folds - 1 else train_end + block
        train_start = 0 if mode == EXPANDING else max(0, train_end - min_train)
        splits.append((train_start, train_end, test_end))
    return splits


def walk_forward_predictions(
    ticker: str,
    start: str = config.START_DATE,
    end: str = None,
    seq_length: int = config.SEQ_LENGTH,
    feature_set: str = config.DEFAULT_FEATURE_SET,
    n_folds: int = config.WF_FOLDS,
    mode: str = EXPANDING,
    model_type: str = "LSTM",
    epochs: int = 20,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
    verbose: bool = True,
):
    """Retrain per fold and stitch the folds into one out-of-sample frame.

    Returns ``(frame, per_fold)`` where ``frame`` has the same
    ``prev``/``close``/``pred`` columns ``backtest.py`` expects, plus the
    fold index each row came from.
    """
    windows = prepare_windows(
        ticker, start=start, end=end, seq_length=seq_length, feature_set=feature_set
    )
    X, y, refs, dates = windows["X"], windows["y"], windows["refs"], windows["dates"]
    splits = walk_forward_splits(len(X), n_folds=n_folds, mode=mode)

    pieces, per_fold = [], []
    for fold, (train_start, train_end, test_end) in enumerate(splits):
        # Reseed per fold so folds are independent draws, not a continuation
        # of one training run.
        set_seeds(seed + fold)

        model = build_model(
            model_type, seq_length=seq_length, n_features=windows["n_features"]
        )
        early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(
            X[train_start:train_end], y[train_start:train_end],
            epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop],
        )

        prev = naive_forecast(refs[train_end:test_end])
        pred = scaling.to_price(model.predict(X[train_end:test_end], verbose=0), prev)
        close = scaling.to_price(y[train_end:test_end], prev)

        piece = pd.DataFrame(
            {"prev": prev, "close": close, "pred": pred, "fold": fold},
            index=dates[train_end:test_end],
        )
        pieces.append(piece)

        scores = regression_metrics(close, pred)
        baseline = regression_metrics(close, prev)
        per_fold.append({
            "ticker": ticker,
            "feature_set": feature_set,
            "mode": mode,
            "fold": fold,
            "train_days": train_end - train_start,
            "test_days": test_end - train_end,
            "test_start": str(dates[train_end].date()),
            "test_end": str(dates[test_end - 1].date()),
            "skill_vs_naive": skill_score(scores["RMSE"], baseline["RMSE"]),
            **direction_metrics(scaling.direction_labels(y[train_end:test_end]),
                                (pred > prev).astype(float)),
        })
        if verbose:
            f = per_fold[-1]
            print(f"    fold {fold}: train {f['train_days']:>5}d  "
                  f"test {f['test_start']} -> {f['test_end']}  "
                  f"skill {f['skill_vs_naive']:+.4f}  acc {f['accuracy']:.4f}")

    return pd.concat(pieces), pd.DataFrame(per_fold)


def run_walk_forward(
    tickers=None,
    start: str = config.START_DATE,
    end: str = None,
    feature_set: str = config.DEFAULT_FEATURE_SET,
    n_folds: int = config.WF_FOLDS,
    mode: str = EXPANDING,
    threshold: float = config.SIGNAL_THRESHOLD,
    epochs: int = 20,
    seeds=(config.RANDOM_SEED,),
):
    """Walk-forward the whole portfolio, then score it like any backtest."""
    tickers = list(tickers or config.PORTFOLIO_CONFIG)
    end = end or date.today().isoformat()

    all_folds, all_perf, details, all_meta = [], [], {}, []
    for seed in seeds:
        per_ticker = {}
        for ticker in tickers:
            print(f"[walk-forward] {ticker} ({feature_set}, {mode}, seed {seed}) ...")
            frame, folds = walk_forward_predictions(
                ticker, start=start, end=end, feature_set=feature_set,
                n_folds=n_folds, mode=mode, epochs=epochs, seed=seed,
            )
            per_ticker[ticker] = frame
            folds["seed"] = seed
            all_folds.append(folds)

        detail, performance, meta = evaluate_portfolio(
            per_ticker, start=start, end=end, threshold=threshold
        )
        performance["seed"] = seed
        all_perf.append(performance)
        details[seed] = detail
        all_meta.append(meta)

    folds_df = pd.concat(all_folds, ignore_index=True)
    perf_df = pd.concat(all_perf, ignore_index=True)

    # Exposure and trade counts are per-seed quantities. Reporting the last
    # seed's values next to a seed-averaged performance table would mix two
    # different populations in one summary, so average them explicitly.
    meta = dict(all_meta[-1])
    meta["avg_exposure_pct"] = float(np.mean([m["avg_exposure_pct"] for m in all_meta]))
    meta["total_position_changes"] = float(
        np.mean([m["total_position_changes"] for m in all_meta])
    )
    meta["exposure_pct_by_seed"] = ", ".join(
        f"{s}:{m['avg_exposure_pct']:.1f}" for s, m in zip(seeds, all_meta)
    )

    # Keep every seed's equity curve, not just the last one. A single curve
    # would misrepresent the spread that section 3.5 of the README documents.
    equity = pd.DataFrame({
        f"strategy_seed{seed}": d["strategy_equity"] for seed, d in details.items()
    })
    equity["buy_hold"] = details[seeds[-1]]["buy_hold_equity"]
    equity["market"] = details[seeds[-1]]["market_equity"]

    meta.update({
        "feature_set": feature_set,
        "split": f"walk-forward ({mode})",
        "n_folds": n_folds,
        "seeds": str(list(seeds)),
        "min_train_ratio": config.WF_MIN_TRAIN_RATIO,
    })
    return details[seeds[-1]], perf_df, folds_df, equity, meta


def _print_results(perf_df: pd.DataFrame, folds_df: pd.DataFrame, meta: dict) -> None:
    print("\n" + "=" * 88)
    print(f"WALK-FORWARD  {meta['split']}, {meta['n_folds']} folds, seeds {meta['seeds']}")
    print(f"out-of-sample: {meta['oos_start']} -> {meta['oos_end']} "
          f"({meta['oos_trading_days']} trading days), feature set: {meta['feature_set']}")
    print(f"average exposure: {meta['avg_exposure_pct']:.1f}% "
          f"(by seed {meta['exposure_pct_by_seed']}), "
          f"position changes: {meta['total_position_changes']:.0f} (mean per seed)")
    print("-" * 88)
    print(f"{'strategy':<32}{'total %':>10}{'CAGR %':>10}{'vol %':>9}{'Sharpe':>9}{'maxDD %':>10}")
    for name, group in perf_df.groupby("strategy", sort=False):
        print(f"{name:<32}{group['total_return_pct'].mean():>10.2f}"
              f"{group['cagr_pct'].mean():>10.2f}{group['annual_vol_pct'].mean():>9.2f}"
              f"{group['sharpe'].mean():>9.2f}{group['max_drawdown_pct'].mean():>10.2f}")
    print("=" * 88)

    print("\nPer-fold prediction skill (pooled over tickers and seeds):")
    print(f"{'fold':<6}{'test window':<26}{'skill vs naive':>18}{'direction acc':>16}")
    for fold, group in folds_df.groupby("fold"):
        window = f"{group['test_start'].min()} -> {group['test_end'].max()}"
        skill = mean_and_ci(group["skill_vs_naive"].values)
        print(f"{fold:<6}{window:<26}{skill['mean']:>+18.4f}{group['accuracy'].mean():>16.4f}")

    overall = mean_and_ci(folds_df["skill_vs_naive"].values)
    print(f"\nskill across all folds: {overall['mean']:+.4f} "
          f"[95% CI {overall['ci_low']:+.4f}, {overall['ci_high']:+.4f}] "
          f"over {int(overall['n_runs'])} ticker-fold-seed cells")
    print("An interval spanning 0 means the model does not beat doing nothing.")


def main():
    parser = argparse.ArgumentParser(description="Run the walk-forward backtest.")
    parser.add_argument("--ticker", action="append", dest="tickers")
    parser.add_argument("--feature-set", default=config.DEFAULT_FEATURE_SET,
                        choices=sorted(config.FEATURE_SETS))
    parser.add_argument("--folds", type=int, default=config.WF_FOLDS)
    parser.add_argument("--mode", default=EXPANDING, choices=VALID_MODES)
    parser.add_argument("--threshold", type=float, default=config.SIGNAL_THRESHOLD)
    parser.add_argument("--seeds", type=int, nargs="+", default=[config.RANDOM_SEED])
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", default=None, help="Output .xlsx path")
    args = parser.parse_args()

    detail, perf_df, folds_df, equity, meta = run_walk_forward(
        tickers=args.tickers, start=args.start, end=args.end,
        feature_set=args.feature_set, n_folds=args.folds, mode=args.mode,
        threshold=args.threshold, epochs=args.epochs, seeds=tuple(args.seeds),
    )

    output_path = args.output or (
        config.REPORT_DIR /
        f"WalkForward_{args.feature_set}_{args.mode}_{meta['oos_start']}_{meta['oos_end']}.xlsx"
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        perf_df.to_excel(writer, sheet_name="performance", index=False)
        folds_df.to_excel(writer, sheet_name="per_fold", index=False)
        equity.to_excel(writer, sheet_name="equity")
        pd.DataFrame([meta]).to_excel(writer, sheet_name="run_config", index=False)
        detail.to_excel(writer, sheet_name="daily_detail")
    print(f"[walk-forward] report saved -> {output_path}")

    _print_results(perf_df, folds_df, meta)


if __name__ == "__main__":
    main()
