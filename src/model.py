"""
Model architectures.

One builder for the three architectures under comparison. Two parameters
carry most of the weight:

* ``n_features`` - the models accept any number of input channels, so one
  architecture serves the whole ablation (close-only through to the full
  technical + macro set) and nothing but the input width differs between
  arms. Holding the body fixed is what makes the comparison a controlled
  experiment.
* ``task`` - a classification head, so "will tomorrow close higher?" can be
  asked directly. Direction accuracy is a far more honest measure of a
  price predictor than RMSE, which autocorrelation alone can make look
  excellent.
* ``units`` - recurrent width. Every reported experiment uses the default
  50; the parameter exists so ``src/tuning.py`` can vary capacity and show
  that the negative result is not an artefact of one arbitrary size.
"""
from tensorflow.keras.layers import (
    LSTM, GRU, Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D,
)
from tensorflow.keras.models import Sequential

from . import config

REGRESSION = "regression"
CLASSIFICATION = "classification"
VALID_TASKS = (REGRESSION, CLASSIFICATION)


def build_model(
    model_type: str,
    seq_length: int = config.SEQ_LENGTH,
    n_features: int = 1,
    task: str = REGRESSION,
    units: int = config.DEFAULT_UNITS,
) -> Sequential:
    model_type = model_type.upper()
    if model_type not in config.VALID_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {config.VALID_MODEL_TYPES}, got {model_type!r}")
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")
    if units < 1:
        raise ValueError(f"units must be >= 1, got {units}")

    model = Sequential(name=f"{model_type}_{task}")
    model.add(Input(shape=(seq_length, n_features)))

    if model_type == "LSTM":
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(0.2))
    elif model_type == "GRU":
        model.add(GRU(units, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units, return_sequences=False))
        model.add(Dropout(0.2))
    elif model_type == "CNN":
        # Left at fixed widths: ``units`` sweeps recurrent capacity only, and
        # the architecture comparison in the README needs the CNN arm to be
        # the same network the saved weights were trained under.
        model.add(Conv1D(filters=64, kernel_size=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=2, activation="relu"))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))

    model.add(Dense(25))

    if task == CLASSIFICATION:
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")

    return model
