"""
Feature engineering - technical indicators and macro factors.

Every function here is **causal**: the value at row t is computed only from
rows <= t. This is not a stylistic preference, it is the whole point. Two
specific traps are avoided:

1. **Centred smoothing.** ``scipy.ndimage.gaussian_filter1d`` places the
   kernel symmetrically around each point, so the "trend" it reports for
   today is partly computed from tomorrow's price. Used as a model feature
   that leaks the answer. ``causal_gaussian_trend`` below uses a one-sided
   kernel instead: today's trend value depends only on today and earlier.

2. **Overnight foreign markets.** The US session for calendar day t closes
   at ~05:00 Taipei time on day t+1, i.e. *after* the Taiwan close on day
   t. Our backtest decides positions at the Taiwan close of day t, so the
   US day-t close is not yet knowable at decision time. Macro series are
   therefore lagged by ``config.MACRO_LAG_DAYS`` (default 1) before they
   ever reach the model. Dropping this lag is one of the easiest ways to
   manufacture a strategy that looks profitable and is not.

Every feature is also expressed in a form that needs no fitted scaler -
a ratio to the current price, a bounded oscillator rescaled by a constant,
or a return. That keeps the "no fitted state can leak across the split"
property established in src/scaling.py.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .data import download_close

# How each column must be normalised inside a window (see scaling.py):
#   "price"    - a price level; divided by the window's reference price
#   "relative" - already a ratio / percentage deviation; passed through
#   "bounded"  - a fixed-range oscillator; rescaled by a constant
KIND_PRICE = "price"
KIND_RELATIVE = "relative"
KIND_BOUNDED = "bounded"


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, in [0, 100]. Uses only past values via ewm."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # A zero average loss means an unbroken winning streak -> RSI 100.
    return out.where(avg_loss != 0.0, 100.0)


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (MACD line minus its signal line), in price units."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def bollinger_position(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Where price sits inside its Bollinger band: -1 = lower, +1 = upper.

    Reported as a position rather than as raw bands so the feature is
    already scale-free.
    """
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    width = num_std * sd
    return ((close - mid) / width.replace(0.0, np.nan)).clip(-3.0, 3.0)


def bias(close: pd.Series, window: int = 20) -> pd.Series:
    """Percentage deviation from the moving average (乖離率)."""
    ma = close.rolling(window).mean()
    return close / ma - 1.0


def causal_gaussian_trend(close: pd.Series, sigma: float = 5.0, truncate: float = 3.0) -> pd.Series:
    """One-sided Gaussian-weighted trend line.

    Weights decay with age, so this behaves like the report's smoothing but
    only ever looks backwards. A centred ``gaussian_filter1d`` would blend
    in future prices and leak the target.
    """
    radius = int(truncate * sigma)
    offsets = np.arange(radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()

    # weights[0] applies to the current bar, weights[k] to k bars ago.
    return close.rolling(radius + 1).apply(
        lambda w: float(np.dot(w[::-1], weights)), raw=True
    )


def trend_slope(trend: pd.Series, lookback: int = 10) -> pd.Series:
    """Relative slope of the trend line over ``lookback`` bars."""
    past = trend.shift(lookback)
    return trend / past.replace(0.0, np.nan) - 1.0


def _macro_frame(index: pd.DatetimeIndex, start: str, end: Optional[str]) -> pd.DataFrame:
    """Daily returns of the macro series, aligned onto the TW calendar.

    Reindexing forward-fills (never back-fills) so a Taiwan trading day that
    the foreign market had closed for reuses the last *known* value. The
    lag is applied on top of that.
    """
    out = pd.DataFrame(index=index)
    for name, ticker in config.MACRO_TICKERS.items():
        series = download_close(ticker, start=start, end=end)["Close"]
        aligned = series.reindex(index.union(series.index)).ffill().reindex(index)
        out[name] = aligned.pct_change()

    return out.shift(config.MACRO_LAG_DAYS)


def build_features(
    ticker: str,
    start: str = config.START_DATE,
    end: Optional[str] = config.END_DATE,
    feature_set: str = "close",
) -> "tuple[pd.DataFrame, Dict[str, str]]":
    """Build the feature matrix for one ticker.

    Returns ``(frame, kinds)`` where ``frame`` always has ``Close`` as its
    first column (the prediction target lives there) and ``kinds`` maps every
    column to its normalisation kind. Rows containing NaN from indicator
    warm-up are dropped, so every returned row is fully observed.
    """
    if feature_set not in config.FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {sorted(config.FEATURE_SETS)}, got {feature_set!r}")

    df = download_close(ticker, start, end)
    close = df["Close"]

    frame = pd.DataFrame({"Close": close})
    kinds: Dict[str, str] = {"Close": KIND_PRICE}

    if feature_set in ("technical", "full"):
        trend = causal_gaussian_trend(close)
        frame["Trend"] = trend
        frame["TrendSlope"] = trend_slope(trend)
        frame["RSI"] = rsi(close) / 50.0 - 1.0          # -> roughly [-1, 1]
        frame["MACDHist"] = macd_histogram(close) / close  # -> price-relative
        frame["BollPos"] = bollinger_position(close)
        frame["Bias"] = bias(close)
        kinds.update({
            "Trend": KIND_PRICE,
            "TrendSlope": KIND_RELATIVE,
            "RSI": KIND_BOUNDED,
            "MACDHist": KIND_RELATIVE,
            "BollPos": KIND_BOUNDED,
            "Bias": KIND_RELATIVE,
        })

    if feature_set == "full":
        macro = _macro_frame(frame.index, start, end)
        for name in macro.columns:
            frame[name] = macro[name]
            kinds[name] = KIND_RELATIVE

    frame = frame.dropna()
    if frame.empty:
        raise ValueError(f"{ticker}: no complete rows left after indicator warm-up ({feature_set}).")

    return frame, kinds


def feature_columns(feature_set: str) -> List[str]:
    """Column names produced by ``build_features`` for a feature set."""
    return list(config.FEATURE_SETS[feature_set])
