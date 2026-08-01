"""
Out-of-sample portfolio backtest.

What this does
--------------
For every ticker in ``config.PORTFOLIO_CONFIG``: train an LSTM on the first
``SPLIT_RATIO`` of its history, then walk forward through the *remaining*
(unseen) period. On each day the model predicts tomorrow's close; if the
predicted move clears ``config.SIGNAL_THRESHOLD`` the strategy holds the
stock the next day, otherwise it sits in cash. Position changes pay
brokerage fees and securities transaction tax.

The result is compared against two reference portfolios over the exact same
window: buy-and-hold of the same five stocks, and buy-and-hold of the market
index.

What this deliberately avoids
----------------------------
Four failure modes make a backtest report numbers that mean nothing:

* building the equity curve from realised prices rather than predictions,
  which yields buy-and-hold performance that would be unchanged if the
  model were deleted;
* scoring predictions over a range that includes the training segment;
* computing risk statistics on a series the strategy never traded;
* omitting transaction costs, which flatters any high-turnover rule.

Every number here is out-of-sample and cost-aware, and the model's output
is what actually drives the positions.

Usage:
    python -m src.backtest
    python -m src.backtest --start 2018-01-01 --threshold 0.002
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from . import config, scaling
from .data import download_close, prepare_dataset
from .metrics import naive_forecast
from .model import build_model
from .train import set_seeds


def _oos_predictions(ticker, start, end, seq_length, split_ratio, epochs, batch_size,
                     feature_set=config.DEFAULT_FEATURE_SET):
    """Train on the training split only, return an out-of-sample frame.

    Columns, per row (one row = one trading decision):
      ``prev``  close on the decision day t (the price we could act at)
      ``close`` close on day t+1 (what actually happened)
      ``pred``  model's prediction of day t+1's close, made using data up to t
    """
    ds = prepare_dataset(ticker, start=start, end=end, seq_length=seq_length,
                         split_ratio=split_ratio, feature_set=feature_set)
    if len(ds["X_test"]) == 0:
        raise ValueError(f"{ticker}: split_ratio={split_ratio} leaves no out-of-sample rows.")

    model = build_model("LSTM", seq_length=seq_length, n_features=ds["n_features"])
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(
        ds["X_train"], ds["y_train"],
        epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop],
    )

    # refs_test is the close on each decision day - the price we could act
    # at - and doubles as the naive "no change" forecast.
    prev = naive_forecast(ds["refs_test"])
    pred = scaling.to_price(model.predict(ds["X_test"], verbose=0), prev)
    close = scaling.to_price(ds["y_test"], prev)
    dates = ds["dates"][ds["train_size"]:]

    return pd.DataFrame({"prev": prev, "close": close, "pred": pred}, index=dates)


def _positions_and_returns(frame: pd.DataFrame, threshold: float, fee: float, tax: float):
    """Turn predictions into a long-or-flat position series and net returns.

    ``position`` for a row is decided from information available *before*
    that row's return is realised, so there is no look-ahead.
    """
    expected_ret = frame["pred"] / frame["prev"] - 1.0
    position = (expected_ret > threshold).astype(float)

    gross_ret = position * (frame["close"] / frame["prev"] - 1.0)

    # Cost is charged whenever exposure changes. Assume flat before day one.
    prev_position = position.shift(1).fillna(0.0)
    delta = position - prev_position
    buy_cost = delta.clip(lower=0) * fee
    sell_cost = (-delta).clip(lower=0) * (fee + tax)

    net_ret = gross_ret - buy_cost - sell_cost
    return position, net_ret, delta


def _benchmark_returns(close: pd.Series, common: pd.DatetimeIndex) -> pd.Series:
    """Daily returns of a benchmark series over the out-of-sample calendar.

    The first out-of-sample day must carry a real return. Restricting the
    series to ``common`` and then calling ``pct_change`` leaves a leading NaN
    that can only be filled with 0, which hands the benchmark one flat day
    while the strategy and buy & hold both trade it - a small but free
    handicap. Seeding the series with the last close *before* the window
    removes it.
    """
    aligned = close.reindex(close.index.union(common)).ffill()
    earlier = aligned.index[aligned.index < common[0]]
    span = common if len(earlier) == 0 else earlier[-1:].append(common)
    return aligned.reindex(span).pct_change().reindex(common).fillna(0.0)


def _performance(returns: pd.Series, label: str) -> dict:
    """Standard performance stats for a daily return series."""
    returns = returns.dropna()
    periods = config.TRADING_DAYS_PER_YEAR
    equity = (1.0 + returns).cumprod()
    years = len(returns) / periods

    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(returns.std() * np.sqrt(periods))
    excess = returns.mean() * periods - config.RISK_FREE_RATE
    sharpe = float(excess / vol) if vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0

    return {
        "strategy": label,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "annual_vol_pct": vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": float(drawdown.min()) * 100,
        "final_value": float(config.INITIAL_CAPITAL * equity.iloc[-1]),
    }


def evaluate_portfolio(
    per_ticker: dict,
    start: str,
    end: str,
    threshold: float = config.SIGNAL_THRESHOLD,
    fee: float = config.FEE_RATE,
    tax: float = config.SELL_TAX_RATE,
):
    """Turn per-ticker out-of-sample predictions into portfolio performance.

    ``per_ticker`` maps ticker -> frame with ``prev``/``close``/``pred``
    columns, whatever produced them. Keeping this separate from model
    training is what lets the single-split backtest and the walk-forward
    backtest share one costing and scoring path, so their numbers are
    directly comparable.
    """
    # Align every ticker onto the same out-of-sample trading calendar.
    common = None
    for frame in per_ticker.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) == 0:
        raise ValueError("No overlapping out-of-sample dates across the portfolio.")

    detail = pd.DataFrame(index=common)
    strategy_ret = pd.Series(0.0, index=common)
    hold_ret = pd.Series(0.0, index=common)
    trade_count = 0
    days_in_market = 0.0

    for ticker, weight in config.PORTFOLIO_CONFIG.items():
        frame = per_ticker[ticker].loc[common]
        position, net_ret, delta = _positions_and_returns(frame, threshold, fee, tax)
        bh_ret = frame["close"] / frame["prev"] - 1.0

        detail[f"{ticker}_close"] = frame["close"]
        detail[f"{ticker}_pred"] = frame["pred"]
        detail[f"{ticker}_position"] = position
        detail[f"{ticker}_net_ret"] = net_ret

        strategy_ret += weight * net_ret
        hold_ret += weight * bh_ret
        trade_count += int((delta != 0).sum())
        days_in_market += float(position.mean()) * weight

    print(f"[backtest] loading market index {config.MARKET_TICKER} ...")
    mkt = download_close(config.MARKET_TICKER, start=start, end=end)
    market_ret = _benchmark_returns(mkt["Close"], common)

    detail["strategy_ret"] = strategy_ret
    detail["buy_hold_ret"] = hold_ret
    detail["market_ret"] = market_ret
    detail["strategy_equity"] = (1 + strategy_ret).cumprod() * config.INITIAL_CAPITAL
    detail["buy_hold_equity"] = (1 + hold_ret).cumprod() * config.INITIAL_CAPITAL
    detail["market_equity"] = (1 + market_ret).cumprod() * config.INITIAL_CAPITAL

    performance = pd.DataFrame([
        _performance(strategy_ret, "LSTM long/flat (net of costs)"),
        _performance(hold_ret, "Buy & hold (same 5 stocks)"),
        _performance(market_ret, f"Market index ({config.MARKET_TICKER})"),
    ])

    meta = {
        "oos_start": str(common[0].date()),
        "oos_end": str(common[-1].date()),
        "oos_trading_days": len(common),
        "signal_threshold": threshold,
        "fee_rate": fee,
        "sell_tax_rate": tax,
        "total_position_changes": trade_count,
        "avg_exposure_pct": days_in_market * 100,
    }
    return detail, performance, meta


def run_backtest(
    start: str = config.START_DATE,
    end: str = None,
    seq_length: int = config.SEQ_LENGTH,
    split_ratio: float = config.SPLIT_RATIO,
    threshold: float = config.SIGNAL_THRESHOLD,
    fee: float = config.FEE_RATE,
    tax: float = config.SELL_TAX_RATE,
    epochs: int = 20,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
    feature_set: str = config.DEFAULT_FEATURE_SET,
):
    """Single-split backtest: one training period, one held-out tail."""
    end = end or date.today().isoformat()
    set_seeds(seed)

    per_ticker = {}
    for ticker in config.PORTFOLIO_CONFIG:
        print(f"[backtest] training on {ticker} ({feature_set}, train split only) ...")
        per_ticker[ticker] = _oos_predictions(
            ticker, start, end, seq_length, split_ratio, epochs, batch_size, feature_set
        )

    detail, performance, meta = evaluate_portfolio(
        per_ticker, start=start, end=end, threshold=threshold, fee=fee, tax=tax
    )
    meta.update({"feature_set": feature_set, "seed": seed, "split": "single"})
    return detail, performance, meta


def save_report(detail: pd.DataFrame, performance: pd.DataFrame, meta: dict, output_path):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        performance.to_excel(writer, sheet_name="performance", index=False)
        pd.DataFrame([meta]).to_excel(writer, sheet_name="run_config", index=False)
        detail.to_excel(writer, sheet_name="daily_detail")
    print(f"[backtest] report saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the out-of-sample portfolio backtest.")
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument("--threshold", type=float, default=config.SIGNAL_THRESHOLD,
                        help="Minimum predicted move required to hold a stock.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--feature-set", default=config.DEFAULT_FEATURE_SET,
                        choices=sorted(config.FEATURE_SETS))
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--output", default=None, help="Output .xlsx path")
    args = parser.parse_args()

    detail, performance, meta = run_backtest(
        start=args.start, end=args.end, threshold=args.threshold, epochs=args.epochs,
        feature_set=args.feature_set, seed=args.seed,
    )

    output_path = args.output or (
        config.REPORT_DIR /
        f"Portfolio_Backtest_{args.feature_set}_{meta['oos_start']}_{meta['oos_end']}.xlsx"
    )
    save_report(detail, performance, meta, output_path)

    print("\n" + "=" * 78)
    print(f"OUT-OF-SAMPLE window: {meta['oos_start']} -> {meta['oos_end']} "
          f"({meta['oos_trading_days']} trading days), feature set: {meta['feature_set']}")
    print(f"position changes: {meta['total_position_changes']}, "
          f"average exposure: {meta['avg_exposure_pct']:.1f}%")
    print("-" * 78)
    print(f"{'strategy':<32}{'total %':>10}{'CAGR %':>10}{'vol %':>9}{'Sharpe':>9}{'maxDD %':>10}")
    for _, row in performance.iterrows():
        print(f"{row['strategy']:<32}{row['total_return_pct']:>10.2f}{row['cagr_pct']:>10.2f}"
              f"{row['annual_vol_pct']:>9.2f}{row['sharpe']:>9.2f}{row['max_drawdown_pct']:>10.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
