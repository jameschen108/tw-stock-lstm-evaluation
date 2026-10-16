"""
Central configuration for the StockAI project.

All tunable parameters (tickers, date ranges, sequence length, file
locations) live here so notebooks/scripts don't hardcode them in five
different places.
"""
from pathlib import Path

# --- Project paths -----------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --- Market / tickers ----------------------------------------------------
# Default single ticker served by the API / used by predict.py.
DEFAULT_TICKER = "2330.TW"          # TSMC

# Portfolio used by backtest.py (equal-weighted).
PORTFOLIO_CONFIG = {
    "2330.TW": 0.2,   # TSMC
    "2317.TW": 0.2,   # Foxconn (Hon Hai)
    "2603.TW": 0.2,   # Evergreen Marine
    "2002.TW": 0.2,   # China Steel
    "2891.TW": 0.2,   # CTBC Financial
}
MARKET_TICKER = "^TWII"             # Taiwan Weighted Index (benchmark)

# --- Features ------------------------------------------------------------
# Macro series used by the "full" feature set, as {column name: ticker}.
MACRO_TICKERS = {
    "VIX": "^VIX",
    "SP500": "^GSPC",
    "USDTWD": "TWD=X",
}

# The US session for calendar day t ends ~05:00 Taipei time on day t+1,
# i.e. after the Taiwan close on day t. Positions in backtest.py are decided
# at the Taiwan close of day t, so the day-t macro value is not knowable at
# decision time and must be lagged. Setting this to 0 reintroduces
# look-ahead and will inflate every downstream result.
MACRO_LAG_DAYS = 1

# The three arms of the ablation study, from the single-feature control to
# the full technical + macro set.
FEATURE_SETS = {
    "close": ("Close",),
    "technical": ("Close", "Trend", "TrendSlope", "RSI", "MACDHist", "BollPos", "Bias"),
    "full": ("Close", "Trend", "TrendSlope", "RSI", "MACDHist", "BollPos", "Bias",
             "VIX", "SP500", "USDTWD"),
}
DEFAULT_FEATURE_SET = "close"

# Seeds used to repeat every experiment. A single run of a neural net on
# noisy financial data tells you almost nothing: on this data the same
# configuration swings between 105% and 306% total return on seed alone.
EXPERIMENT_SEEDS = (0, 1, 2, 3, 4)

# --- Walk-forward evaluation ---------------------------------------------
# A single 80/20 split evaluates one stretch of market and cannot tell a
# real edge from a lucky regime. Walk-forward retrains at each fold so every
# prediction is out-of-sample, across several years instead of one window.
WF_FOLDS = 5
WF_MIN_TRAIN_RATIO = 0.5   # history reserved to train the first fold

# --- Data / training parameters -----------------------------------------
START_DATE = "2014-01-01"
END_DATE = None                     # None -> use today's date dynamically
SEQ_LENGTH = 10                     # days of history used to predict next close
SPLIT_RATIO = 0.8                   # train/test split for evaluation

# Fraction of the *training* rows held back for model selection in
# src/train.py (checkpointing and early stopping). Carved off the end of the
# training segment, never taken from the test set: choosing weights by
# test-set loss and then reporting test-set metrics would make the reported
# skill a training-time quantity.
VAL_RATIO = 0.1
INITIAL_CAPITAL = 100_000           # starting capital for backtest, in TWD
RISK_FREE_RATE = 0.0                # annualised, used by the Sharpe ratio
TRADING_DAYS_PER_YEAR = 252

# --- Strategy / trading costs -------------------------------------------
# The backtest is long-or-flat: hold a ticker only when the model predicts
# tomorrow's close to be at least SIGNAL_THRESHOLD above today's close.
SIGNAL_THRESHOLD = 0.0              # e.g. 0.002 -> only act on >0.2% moves

# Taiwan retail cost structure: 0.1425% brokerage each side, plus a 0.3%
# securities transaction tax charged on sells only. Ignoring these is the
# fastest way to make a daily-turnover strategy look profitable.
FEE_RATE = 0.001425
SELL_TAX_RATE = 0.003

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
RANDOM_SEED = 42

# Recurrent width of the LSTM/GRU arms. Every reported experiment uses this
# value; src/tuning.py varies it to check that the negative result is not an
# artefact of one arbitrary capacity.
DEFAULT_UNITS = 50

VALID_MODEL_TYPES = ("LSTM", "GRU", "CNN")


def weights_path(ticker: str, model_type: str = "LSTM") -> Path:
    """Path where a trained model's weights are stored."""
    safe_ticker = ticker.replace("^", "")
    return MODEL_DIR / f"best_{model_type}_{safe_ticker}.weights.h5"


def preproc_path(ticker: str, model_type: str = "LSTM") -> Path:
    """Path of the preprocessing spec saved alongside a ticker's weights.

    Weights are meaningless without the transform that produced their
    training data, so weights saved without it can never be used for
    inference. The spec is
    plain JSON (it holds no fitted parameters, only the transform's
    identity and shape) so it stays readable and diffable.

    Keyed by model type as well as ticker, exactly like ``weights_path``.
    One spec per ticker would let a GRU trained on ``technical`` overwrite
    the spec belonging to a close-only LSTM, after which the LSTM's weights
    would be rebuilt with the wrong input width - the silent mismatch this
    file pairing exists to prevent.
    """
    safe_ticker = ticker.replace("^", "")
    return MODEL_DIR / f"preproc_{model_type}_{safe_ticker}.json"
