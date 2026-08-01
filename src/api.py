"""
FastAPI serving layer.

Serves any ticker and architecture trained through src/train.py, defaulting
to config.DEFAULT_TICKER. Models and their preprocessing spec are loaded
lazily and cached per (ticker, model_type), so an untrained ticker produces
a 404 naming the command that would train it rather than a crash at
startup.

Run locally:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from . import config, scaling
from .predict import load_trained

app = FastAPI(title="StockAI API", description="LSTM/GRU/CNN next-day close price prediction")

_model_cache = {}


def get_model(ticker: str, model_type: str):
    key = (ticker, model_type)
    if key not in _model_cache:
        try:
            _model_cache[key] = load_trained(ticker, model_type)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return _model_cache[key]


class StockInput(BaseModel):
    ticker: str = config.DEFAULT_TICKER
    model_type: str = "LSTM"
    prices: List[float]  # exactly SEQ_LENGTH most recent closing prices, oldest -> newest

    @field_validator("model_type")
    @classmethod
    def _valid_model_type(cls, v):
        v = v.upper()
        if v not in config.VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {config.VALID_MODEL_TYPES}")
        return v


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the StockAI API",
        "default_ticker": config.DEFAULT_TICKER,
        "sequence_length": config.SEQ_LENGTH,
    }


@app.post("/predict")
def predict_price(data: StockInput):
    model, spec = get_model(data.ticker, data.model_type)
    seq_length = int(spec["seq_length"])
    feature_set = spec.get("feature_set", config.DEFAULT_FEATURE_SET)

    if feature_set != config.DEFAULT_FEATURE_SET:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{data.ticker} is trained on feature set '{feature_set}' "
                f"({', '.join(spec.get('feature_names', []))}). Those inputs cannot be "
                "reconstructed from a bare price list, so this endpoint only serves "
                "close-only models. Retrain with --feature-set close, or call "
                "src.predict.predict_future, which downloads what it needs."
            ),
        )

    if len(data.prices) != seq_length:
        raise HTTPException(
            status_code=400,
            detail=f"'prices' must contain exactly {seq_length} values (oldest to newest).",
        )

    window, ref = scaling.window_from_prices(data.prices, seq_length)
    predicted_return = float(model.predict(window, verbose=0)[0][0])
    predicted_price = ref * (1.0 + predicted_return)

    return {
        "ticker": data.ticker,
        "model_type": data.model_type,
        "input_last_price": data.prices[-1],
        "predicted_next_close": predicted_price,
        "predicted_change_pct": predicted_return * 100,
    }
