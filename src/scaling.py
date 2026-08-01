"""
Window-relative return scaling.

Why not MinMaxScaler
--------------------
The obvious approach is to feed raw closing prices through a
``MinMaxScaler``, and on a trending series both ways of fitting it fail.

Fitted on the training rows only - the leak-free way - the scaler learns
2330.TW's training range of roughly 70-630. The test period runs up to
2510, so test inputs land between 0.89 and 4.37 while the network has only
ever seen [0, 1]. Its outputs saturate near the top of the trained range
and it systematically under-predicts.

Fitted on the whole series instead, everything lands inside [0, 1] - but
only because the test period's high and low were used to place the
boundaries, which is exactly the leak this project is trying to avoid.

So the range problem and the leak trade off against each other, and
``src/sensitivity.py`` measures both: neither variant comes close to the
naive persistence baseline.

What this uses instead
----------------------
Each window is expressed as simple returns relative to its **own** last
close::

    ref_i    = prices[i + seq_length - 1]      # the decision-day price
    X[i][j]  = prices[i + j]     / ref_i - 1
    y[i]     = prices[i + seq_length] / ref_i - 1

Two properties follow directly:

* **Scale invariance.** A 1% move looks identical at NT$100 and NT$2350, so
  a model trained on the cheap era stays valid in the expensive one.
* **No leakage is possible.** There are no fitted parameters at all - the
  transform for a window depends only on that window - so there is nothing
  the training split could learn about the test split.

The model therefore predicts *tomorrow's percentage move*, and the naive
"no change" baseline is simply the constant 0 in this space.

One optional extra
------------------
``fit_channel_stats`` / ``apply_channel_stats`` add a per-channel
standardisation for the multi-feature arms, whose channels differ in
magnitude by two orders of magnitude. That step *does* fit parameters, so
it is fitted on the training rows only and is off by default; see its
docstring for why the ablation needs it.
"""
from typing import Tuple

import numpy as np

MODE = "window_return"


def make_windows(prices: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, refs) from a raw close-price series.

    ``prices`` may be shape (N,) or (N, 1). Returns X as (n, seq_length, 1),
    y as (n, 1) and refs as (n,) where ``refs[i]`` is the price each window
    is normalised against - i.e. the last close the model can actually see.
    """
    prices = np.asarray(prices, dtype="float64").reshape(-1)
    n = len(prices) - seq_length
    if n <= 0:
        raise ValueError(
            f"need more than seq_length={seq_length} prices to build a window, got {len(prices)}"
        )

    windows = np.lib.stride_tricks.sliding_window_view(prices, seq_length)[:n]
    refs = prices[seq_length - 1: seq_length - 1 + n]
    targets = prices[seq_length:]

    if np.any(refs == 0):
        raise ValueError("close price of 0 encountered; cannot normalise a window against it")

    X = (windows / refs[:, None] - 1.0).reshape(n, seq_length, 1)
    y = (targets / refs - 1.0).reshape(n, 1)
    return X, y, refs


def make_feature_windows(
    frame, kinds, seq_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-feature version of :func:`make_windows`.

    ``frame`` must have the prediction target ``Close`` as its first column;
    ``kinds`` maps each column to how it is normalised inside a window:

    * ``price``    - divided by the window's reference close, minus 1, so
      every price-like channel shares one scale-free representation;
    * ``relative`` - already a ratio or return, passed through unchanged;
    * ``bounded``  - a fixed-range oscillator, passed through unchanged
      (features.py has already rescaled it by a constant).

    None of these involve a fitted parameter, so the leak-free guarantee of
    the single-feature path carries over unchanged. Returns X of shape
    (n, seq_length, n_features), y of shape (n, 1) holding the next-day
    close as a return, and refs of shape (n,).
    """
    columns = list(frame.columns)
    if columns[0] != "Close":
        raise ValueError(f"first column must be 'Close', got {columns[0]!r}")

    values = np.asarray(frame.to_numpy(), dtype="float64")
    n = len(values) - seq_length
    if n <= 0:
        raise ValueError(
            f"need more than seq_length={seq_length} rows to build a window, got {len(values)}"
        )

    windows = np.lib.stride_tricks.sliding_window_view(values, seq_length, axis=0)[:n]
    # sliding_window_view puts the time axis last -> (n, features, seq)
    windows = np.transpose(windows, (0, 2, 1))

    refs = values[seq_length - 1: seq_length - 1 + n, 0]
    if np.any(refs == 0):
        raise ValueError("close price of 0 encountered; cannot normalise a window against it")

    X = windows.copy()
    for j, name in enumerate(columns):
        if kinds[name] == "price":
            X[:, :, j] = X[:, :, j] / refs[:, None] - 1.0
        elif kinds[name] not in ("relative", "bounded"):
            raise ValueError(f"unknown normalisation kind {kinds[name]!r} for column {name!r}")

    targets = values[seq_length:, 0]
    y = (targets / refs - 1.0).reshape(n, 1)
    return X, y, refs


def fit_channel_stats(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation, fitted on training rows only.

    Why this exists
    ---------------
    The window-relative transform above leaves each feature in its own
    natural units, and those units are not comparable. Measured on this
    data the channel standard deviations span roughly two orders of
    magnitude - MACDHist around 0.005, Bias and TrendSlope around 0.035,
    RSI around 0.21, BollPos around 0.63 - while the Close-return channel
    the model is actually asked to predict sits near 0.02. A network fed
    those channels raw is dominated by whichever one happens to swing
    hardest, which is a plausible alternative explanation for the
    multi-feature arms failing to beat the close-only control: not "the
    features carry no signal" but "the features were never given a fair
    hearing". ``src/experiment.py --standardise`` exists to tell those two
    explanations apart.

    Leakage
    -------
    Unlike the window transform, this *is* a fitted parameter, so it must
    be fitted on the training rows alone and then applied unchanged to the
    test rows. That is weaker than the "no fitted state at all" guarantee
    of ``make_feature_windows`` and the README says so explicitly.

    Returns ``(mean, std)``, each of shape ``(n_features,)``. Channels with
    zero variance in training keep a scale of 1 so they pass through
    unchanged rather than blowing up.
    """
    X_train = np.asarray(X_train, dtype="float64")
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std = np.where(std > 0, std, 1.0)
    return mean, std


def apply_channel_stats(X: np.ndarray, stats: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Apply ``fit_channel_stats`` output to any split of the same windows."""
    mean, std = stats
    return (np.asarray(X, dtype="float64") - mean) / std


def to_price(normalised, refs) -> np.ndarray:
    """Convert normalised returns back into prices: ``ref * (1 + r)``."""
    normalised = np.asarray(normalised, dtype="float64").reshape(-1)
    refs = np.asarray(refs, dtype="float64").reshape(-1)
    return refs * (1.0 + normalised)


def direction_labels(y_returns) -> np.ndarray:
    """Binary up/down labels from next-day returns (1 = up)."""
    return (np.asarray(y_returns, dtype="float64").reshape(-1) > 0).astype("float64").reshape(-1, 1)


def window_from_prices(prices, seq_length: int) -> Tuple[np.ndarray, float]:
    """Normalise the most recent ``seq_length`` closes into a model input.

    Returns the (1, seq_length, 1) tensor and the reference price needed to
    turn the prediction back into money.
    """
    prices = np.asarray(prices, dtype="float64").reshape(-1)
    if len(prices) < seq_length:
        raise ValueError(f"need at least {seq_length} prices, got {len(prices)}")

    window = prices[-seq_length:]
    ref = float(window[-1])
    if ref == 0:
        raise ValueError("last close is 0; cannot normalise")
    return (window / ref - 1.0).reshape(1, seq_length, 1), ref
