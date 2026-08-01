"""
Sensitivity of the conclusion to the preprocessing regime.

This module produces the table in README section 4.1. It trains the same
architecture on the same ticker and the same train/test split three times,
changing only how the close price is normalised:

``minmax_global``
    A single ``MinMaxScaler`` fitted on the **whole** series before the
    split. The test period's high and low leak into training, so the model
    knows the price range it is about to be asked to predict.

``minmax_train_only``
    The same scaler, fitted on the training rows only. The leak is gone,
    but a new problem surfaces: on a trending series the test data maps far
    outside [0, 1] (2330.TW trains on roughly 70-630 and is tested up to
    2510, so test inputs reach ~4.4 after scaling), the network saturates
    and systematically under-predicts.

``window_return``
    The representation this project uses - each window normalised against
    its own last close, so the model predicts a percentage move. Scale
    invariant, and leak-free by construction because nothing is fitted.

All three are scored the same way, in price units, against the same naive
persistence baseline, so the numbers are directly comparable. The point of
the table is that fixing the leak alone makes results look *worse*: the leak
was masking the representation's defect, and both have to be addressed.

Usage:
    python -m src.sensitivity
    python -m src.sensitivity --ticker 2317.TW --seeds 0 1 2
"""
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from . import config, scaling
from .data import download_close
from .metrics import regression_metrics, skill_score
from .model import build_model
from .train import set_seeds

MINMAX_GLOBAL = "minmax_global"
MINMAX_TRAIN_ONLY = "minmax_train_only"
WINDOW_RETURN = "window_return"
REGIMES = (MINMAX_GLOBAL, MINMAX_TRAIN_ONLY, WINDOW_RETURN)

DESCRIPTIONS = {
    MINMAX_GLOBAL: "MinMax fitted before the split (leaks test range)",
    MINMAX_TRAIN_ONLY: "MinMax fitted on training rows only",
    WINDOW_RETURN: "Window-relative returns (this project)",
}


def _minmax_windows(prices: np.ndarray, seq_length: int, split_ratio: float, fit_on_train: bool):
    """Build the legacy MinMax representation, returning price-space arrays.

    Kept deliberately close to the textbook formulation this table is about,
    rather than reusing scaling.py, so the comparison is honest.
    """
    prices = np.asarray(prices, dtype="float64").reshape(-1, 1)
    n_windows = len(prices) - seq_length
    train_size = int(n_windows * split_ratio)

    scaler = MinMaxScaler(feature_range=(0, 1))
    if fit_on_train:
        # Training window i spans rows [i, i+seq_length]; the last training
        # window therefore ends at row train_size+seq_length-1.
        scaler.fit(prices[: train_size + seq_length])
    else:
        scaler.fit(prices)
    scaled = scaler.transform(prices)

    X = np.stack([scaled[i:i + seq_length] for i in range(n_windows)])
    y = np.stack([scaled[i + seq_length] for i in range(n_windows)])
    return X, y, scaler, train_size


def run_regime(
    ticker: str,
    regime: str,
    seed: int,
    start: str = config.START_DATE,
    end: str = None,
    seq_length: int = config.SEQ_LENGTH,
    split_ratio: float = config.SPLIT_RATIO,
    epochs: int = 20,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
) -> dict:
    """Train under one preprocessing regime, score in price units."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")

    set_seeds(seed)
    prices = download_close(ticker, start=start, end=end)["Close"].to_numpy()

    if regime == WINDOW_RETURN:
        X, y, refs = scaling.make_windows(prices, seq_length)
        train_size = int(len(X) * split_ratio)
        to_price = lambda arr, idx: scaling.to_price(arr, refs[idx])  # noqa: E731
    else:
        X, y, scaler, train_size = _minmax_windows(
            prices, seq_length, split_ratio, fit_on_train=(regime == MINMAX_TRAIN_ONLY)
        )
        to_price = lambda arr, idx: scaler.inverse_transform(  # noqa: E731
            np.asarray(arr).reshape(-1, 1)
        ).flatten()

    test = slice(train_size, len(X))
    model = build_model("LSTM", seq_length=seq_length, n_features=1)
    model.fit(
        X[:train_size], y[:train_size],
        epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
    )

    preds = to_price(model.predict(X[test], verbose=0), test)
    actual = to_price(y[test], test)
    # Naive persistence in every regime is the last close inside the window.
    naive = np.asarray(prices[seq_length - 1: -1])[test]

    scores = regression_metrics(actual, preds)
    baseline = regression_metrics(actual, naive)
    return {
        "ticker": ticker,
        "regime": regime,
        "seed": seed,
        "RMSE": scores["RMSE"],
        "R2": scores["R2"],
        "naive_RMSE": baseline["RMSE"],
        "naive_R2": baseline["R2"],
        "skill_vs_naive": skill_score(scores["RMSE"], baseline["RMSE"]),
    }


def run_sensitivity(ticker: str, seeds=(0, 1, 2), **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for regime in REGIMES:
        for seed in seeds:
            print(f"[sensitivity] {ticker} / {regime} / seed {seed} ...", flush=True)
            rows.append(run_regime(ticker, regime, seed, **kwargs))

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby("regime", sort=False)
        .agg(RMSE=("RMSE", "mean"), R2=("R2", "mean"),
             skill_vs_naive=("skill_vs_naive", "mean"),
             skill_std=("skill_vs_naive", "std"))
        .reindex(REGIMES)
        .reset_index()
    )
    summary["description"] = summary["regime"].map(DESCRIPTIONS)
    return runs, summary


def main():
    parser = argparse.ArgumentParser(
        description="Measure how the preprocessing regime changes the conclusion."
    )
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    runs, summary = run_sensitivity(
        args.ticker, seeds=tuple(args.seeds),
        start=args.start, end=args.end, epochs=args.epochs,
    )

    output_path = args.output or (config.REPORT_DIR / f"Sensitivity_{args.ticker}.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        runs.to_excel(writer, sheet_name="runs", index=False)
    print(f"\n[sensitivity] results saved -> {output_path}")

    naive_rmse = runs["naive_RMSE"].mean()
    naive_r2 = runs["naive_R2"].mean()
    print("\n" + "=" * 96)
    print(f"PREPROCESSING SENSITIVITY - {args.ticker}, seeds {args.seeds}, "
          f"same split and architecture throughout")
    print(f"naive persistence baseline: RMSE {naive_rmse:.4f}, R2 {naive_r2:.4f}")
    print("-" * 96)
    print(f"{'regime':<20}{'RMSE':>12}{'R2':>10}{'skill vs naive':>18}{'std':>10}")
    for _, r in summary.iterrows():
        std = 0.0 if pd.isna(r["skill_std"]) else r["skill_std"]
        print(f"{r['regime']:<20}{r['RMSE']:>12.4f}{r['R2']:>10.4f}"
              f"{r['skill_vs_naive']:>+18.4f}{std:>10.4f}")
    print("=" * 96)
    print("Fixing the leak alone makes the model look worse: the leak was masking")
    print("the representation's failure on a trending series. Both must be addressed.")


if __name__ == "__main__":
    main()
