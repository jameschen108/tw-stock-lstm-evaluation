"""
Train a model for one ticker and save BOTH the weights and the
preprocessing spec they were trained under.

Weights alone are not a model. A normalised prediction cannot be turned
back into a price without knowing how the training data was normalised, so
saving weights without their transform produces files that can never be
used for inference. Every run therefore writes a matching (weights,
preproc) pair per ticker and model type.

Model selection (checkpointing and early stopping) runs against a
validation tail carved out of the *training* segment, never against the
test set. Selecting weights on the same rows you then report metrics on is
the mistake README section 4.4 is about; it would make the reported skill
score a training-time quantity.

Test-set metrics are always reported next to the naive persistence
baseline - see src/metrics.py for why that matters.

Usage:
    python -m src.train --ticker 2330.TW --model-type LSTM --epochs 50
"""
import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from . import config, scaling
from .data import prepare_dataset
from .metrics import format_comparison, naive_forecast, regression_metrics, skill_score
from .model import build_model


def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_preproc(
    ticker: str,
    model_type: str = "LSTM",
    seq_length: int = config.SEQ_LENGTH,
    feature_set: str = config.DEFAULT_FEATURE_SET,
    feature_names=None,
) -> None:
    """Write the preprocessing spec that belongs to this ticker's weights.

    The feature set is part of the spec: weights trained on ten input
    channels cannot be served from a close-only window, and a silent shape
    mismatch is precisely the class of bug this file exists to prevent.
    """
    config.preproc_path(ticker, model_type).write_text(
        json.dumps({
            "mode": scaling.MODE,
            "seq_length": seq_length,
            "ticker": ticker,
            "model_type": model_type,
            "feature_set": feature_set,
            "feature_names": list(feature_names or config.FEATURE_SETS[feature_set]),
        }, indent=2)
    )


def split_validation_tail(X, y, val_ratio: float = config.VAL_RATIO):
    """Hold out the most recent ``val_ratio`` of the *training* rows.

    Split by time rather than at random: the rows are overlapping windows of
    one price series, so a shuffled validation set would sit between windows
    it shares nine of ten days with, and would score like training data.

    Returns ``(X_fit, y_fit, validation_data)`` where ``validation_data`` is
    ``None`` when there are too few rows to spare any.
    """
    n_val = int(len(X) * val_ratio)
    if n_val < 1 or n_val >= len(X):
        return X, y, None
    cut = len(X) - n_val
    return X[:cut], y[:cut], (X[cut:], y[cut:])


def train(
    ticker: str = config.DEFAULT_TICKER,
    model_type: str = "LSTM",
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
    feature_set: str = config.DEFAULT_FEATURE_SET,
    val_ratio: float = config.VAL_RATIO,
):
    set_seeds(seed)

    data = prepare_dataset(ticker, start=start, end=end, feature_set=feature_set)
    model = build_model(
        model_type, seq_length=data["seq_length"], n_features=data["n_features"]
    )

    # Model selection uses a tail of the training segment. The test set is
    # touched exactly once, at scoring time, further down this function.
    X_fit, y_fit, validation_data = split_validation_tail(
        data["X_train"], data["y_train"], val_ratio
    )
    monitor = "val_loss" if validation_data is not None else "loss"

    weight_file = config.weights_path(ticker, model_type)
    checkpoint = ModelCheckpoint(
        str(weight_file),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=0,
    )
    early_stop = EarlyStopping(monitor=monitor, patience=10, restore_best_weights=True)

    model.fit(
        X_fit, y_fit,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )
    model.load_weights(str(weight_file))

    has_test = len(data["X_test"]) > 0

    # Persist the preprocessing spec so predict.py / api.py can reproduce
    # exactly the transform these weights were trained under.
    save_preproc(
        ticker,
        model_type=model_type,
        seq_length=data["seq_length"],
        feature_set=data["feature_set"],
        feature_names=data["feature_names"],
    )

    metrics = {}
    if has_test:
        refs = data["refs_test"]
        preds = scaling.to_price(model.predict(data["X_test"], verbose=0), refs)
        actual = scaling.to_price(data["y_test"], refs)

        metrics = regression_metrics(actual, preds)
        baseline = regression_metrics(actual, naive_forecast(refs))
        metrics["naive_RMSE"] = baseline["RMSE"]
        metrics["skill_vs_naive"] = skill_score(metrics["RMSE"], baseline["RMSE"])

        print(f"\n{ticker} [{model_type}] test-set metrics:")
        print(format_comparison(model_type, metrics, baseline))
        print(f"Saved weights -> {weight_file}")
        print(f"Saved preproc -> {config.preproc_path(ticker, model_type)}")

    return model, data, metrics


def main():
    parser = argparse.ArgumentParser(description="Train a stock prediction model.")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--model-type", default="LSTM", choices=config.VALID_MODEL_TYPES)
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=config.END_DATE)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--feature-set", default=config.DEFAULT_FEATURE_SET,
                        choices=sorted(config.FEATURE_SETS))
    args = parser.parse_args()

    train(
        ticker=args.ticker,
        model_type=args.model_type,
        start=args.start,
        end=args.end,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        feature_set=args.feature_set,
    )


if __name__ == "__main__":
    main()
