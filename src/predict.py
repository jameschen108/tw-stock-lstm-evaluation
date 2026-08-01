"""
Load a trained (weights, preprocessing spec) pair for a ticker and
recursively predict the next N trading days.

Requires that src/train.py has already been run for this ticker/model type
(so a matching weights + preproc file exists in models/). Loading is
intentionally explicit rather than silently retraining on every call.

Predictions are recursive: each step's output is appended to the price
history and the window is re-normalised before the next step, so the
uncertainty compounds the further out you go.

Usage:
    python -m src.predict --ticker 2330.TW --model-type LSTM --days 5
"""
import argparse
import json

import pandas as pd

from . import config, scaling
from .model import build_model


def load_trained(ticker: str, model_type: str = "LSTM"):
    """Load a model together with the preprocessing spec it was trained under.

    Returns ``(model, spec)``. Weights loaded under the wrong transform
    produce plausible-looking nonsense, so a missing or mismatched spec is a
    hard error rather than a guess.
    """
    weight_file = config.weights_path(ticker, model_type)
    preproc_file = config.preproc_path(ticker, model_type)

    if not weight_file.exists() or not preproc_file.exists():
        raise FileNotFoundError(
            f"No trained ({model_type}) model found for {ticker}.\n"
            f"Expected: {weight_file}\n"
            f"      and: {preproc_file}\n"
            f"Run: python -m src.train --ticker {ticker} --model-type {model_type}"
        )

    spec = json.loads(preproc_file.read_text())
    if spec.get("mode") != scaling.MODE:
        raise ValueError(
            f"{preproc_file} was written by preprocessing mode {spec.get('mode')!r}, "
            f"but this build expects {scaling.MODE!r}. Retrain {ticker}."
        )
    # Belt and braces: the filename already encodes the architecture, but a
    # spec copied by hand between files would otherwise load silently.
    if spec.get("model_type", model_type) != model_type:
        raise ValueError(
            f"{preproc_file} describes a {spec['model_type']} model but was loaded "
            f"as {model_type}. Retrain {ticker} with --model-type {model_type}."
        )

    seq_length = int(spec.get("seq_length", config.SEQ_LENGTH))
    feature_set = spec.get("feature_set", config.DEFAULT_FEATURE_SET)
    n_features = len(spec.get("feature_names") or config.FEATURE_SETS[feature_set])

    model = build_model(model_type, seq_length=seq_length, n_features=n_features)
    model.load_weights(str(weight_file))
    return model, spec


def predict_future(
    ticker: str, model_type: str = "LSTM", days: int = 5, start: str = None
) -> pd.DataFrame:
    from .features import build_features

    model, spec = load_trained(ticker, model_type)
    seq_length = int(spec["seq_length"])
    feature_set = spec.get("feature_set", config.DEFAULT_FEATURE_SET)

    # Indicators need warm-up, so pull a couple of years rather than a window.
    frame, kinds = build_features(
        ticker, start=start or "2022-01-01", feature_set=feature_set
    )
    if len(frame) < seq_length:
        raise ValueError(f"Not enough recent data for {ticker} to build a prediction window.")

    last_close = float(frame["Close"].iloc[-1])
    last_date = frame.index[-1]

    if feature_set != config.DEFAULT_FEATURE_SET and days > 1:
        raise ValueError(
            f"{ticker} was trained on feature set {feature_set!r}, which needs RSI, MACD, "
            "VIX and friends as inputs. Rolling more than one day ahead would mean "
            "inventing those values for days that have not happened, so multi-step "
            "forecasts are only supported for the close-only models. Use --days 1."
        )

    if feature_set == config.DEFAULT_FEATURE_SET:
        # Roll forward in price space: each step re-normalises the window
        # against its own (predicted) last close, so the recursion stays
        # scale-consistent no matter how far the price drifts.
        history = list(frame["Close"].to_numpy()[-seq_length:])
        preds = []
        for _ in range(days):
            window, ref = scaling.window_from_prices(history, seq_length)
            next_return = float(model.predict(window, verbose=0)[0][0])
            next_price = ref * (1.0 + next_return)
            preds.append(next_price)
            history = history[1:] + [next_price]
    else:
        # Build the final (unlabelled) window by hand: make_feature_windows
        # needs a target row, which by definition does not exist yet.
        window = frame.iloc[-seq_length:]
        values = window.to_numpy(dtype="float64")
        ref = float(values[-1, 0])
        x = values.copy()
        for j, name in enumerate(window.columns):
            if kinds[name] == "price":
                x[:, j] = x[:, j] / ref - 1.0
        next_return = float(model.predict(x.reshape(1, seq_length, -1), verbose=0)[0][0])
        preds = [ref * (1.0 + next_return)]

    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(preds))

    result = pd.DataFrame({"date": future_dates, "predicted_close": preds})
    result.attrs["last_close"] = last_close
    result.attrs["last_date"] = last_date
    result.attrs["feature_set"] = feature_set
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict future closing prices.")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--model-type", default="LSTM", choices=config.VALID_MODEL_TYPES)
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()

    try:
        result = predict_future(args.ticker, args.model_type, args.days)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"\n{exc}")

    print(f"\n{args.ticker} last close ({result.attrs['last_date'].date()}): "
          f"{result.attrs['last_close']:.2f}  [feature set: {result.attrs['feature_set']}]")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
