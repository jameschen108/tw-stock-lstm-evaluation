"""
Data download and windowing.

One place where price history is fetched and turned into model inputs, so
every entry point (training, backtest, walk-forward, ablation) sees exactly
the same windows.
"""
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf

from . import config, scaling


def download_close(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Download daily close prices for a ticker as a single-column DataFrame.

    Uses ``auto_adjust=True`` so stock splits / dividends don't create
    artificial price gaps. Without it, a split shows up as a sudden drop
    that the model would learn as a real move (2603.TW is one such case).
    """
    end = end or date.today().isoformat()
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df is None or len(df) == 0:
        raise ValueError(
            f"No data returned for {ticker} ({start} -> {end}). "
            "Check the ticker symbol and your network connection."
        )

    # yfinance sometimes returns MultiIndex columns (ticker, field) -
    # normalize to a single 'Close' column regardless.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df.xs("Close", axis=1, level=0)
        else:
            df = df.iloc[:, [0]]
        df = df.iloc[:, [0]]
        df.columns = ["Close"]
    else:
        df = df[["Close"]]

    return df.dropna()


def prepare_dataset(
    ticker: str,
    start: str = config.START_DATE,
    end: Optional[str] = config.END_DATE,
    seq_length: int = config.SEQ_LENGTH,
    split_ratio: float = config.SPLIT_RATIO,
    feature_set: str = config.DEFAULT_FEATURE_SET,
    standardise: bool = False,
):
    """Download + window a ticker's history for the chosen feature set.

    Each window is normalised against its own last close (see
    ``src/scaling.py``), so the model learns percentage moves rather than
    price levels. Nothing is fitted across the split, which makes scaling
    leakage impossible by construction. That stays true however many feature
    channels are added, because every feature is a ratio, a return or a
    bounded oscillator rescaled by a constant.

    ``standardise=True`` adds a per-channel z-score on top, fitted on the
    training rows only. It exists so the ablation can separate "the extra
    features carry no signal" from "the extra features arrived on wildly
    different scales"; it is the one place in this project where a fitted
    parameter exists at all, and it is off by default.

    Returns a dict with X_train/X_test, y_train/y_test, the per-window
    reference prices (refs_train/refs_test) needed to convert predictions
    back into money, and the dates of each *target* day.
    """
    windows = prepare_windows(
        ticker, start=start, end=end, seq_length=seq_length, feature_set=feature_set
    )
    X, y, refs = windows["X"], windows["y"], windows["refs"]
    train_size = int(len(X) * split_ratio)

    X_train, X_test = X[:train_size], X[train_size:]
    channel_stats = None
    if standardise:
        # Fitted on the training rows alone, then applied unchanged to the
        # test rows - the test segment never influences the statistics.
        channel_stats = scaling.fit_channel_stats(X_train)
        X_train = scaling.apply_channel_stats(X_train, channel_stats)
        X_test = scaling.apply_channel_stats(X_test, channel_stats)

    return {
        **windows,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y[:train_size],
        "y_test": y[train_size:],
        "refs_train": refs[:train_size],
        "refs_test": refs[train_size:],
        "train_size": train_size,
        "standardise": standardise,
        "channel_stats": channel_stats,
    }


def prepare_windows(
    ticker: str,
    start: str = config.START_DATE,
    end: Optional[str] = config.END_DATE,
    seq_length: int = config.SEQ_LENGTH,
    feature_set: str = config.DEFAULT_FEATURE_SET,
):
    """Download and window a ticker without imposing any train/test split.

    Split-free on purpose: ``prepare_dataset`` layers a single split on top,
    while ``walkforward.py`` slices the same arrays into many folds. Both
    therefore see byte-identical windows, so a walk-forward result and a
    single-split result differ only in how the data was partitioned.
    """
    from .features import build_features  # imported here to avoid a cycle

    frame, kinds = build_features(ticker, start=start, end=end, feature_set=feature_set)
    X, y, refs = scaling.make_feature_windows(frame, kinds, seq_length)

    return {
        "X": X,
        "y": y,
        "refs": refs,
        "dates": frame.index[seq_length:],
        "seq_length": seq_length,
        "feature_set": feature_set,
        "feature_names": list(frame.columns),
        "n_features": X.shape[2],
        "raw_df": frame,
    }
