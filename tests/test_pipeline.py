"""
Offline sanity tests for the consolidated pipeline.

These tests monkeypatch yfinance.download with synthetic price data so
they can run without any network access (data.py, train.py, predict.py,
and backtest.py all only touch the network through yfinance.download /
period-based download calls).

Run with: python -m pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from src import config, data, model as model_mod


def _fake_prices(n=400, start="2022-01-01"):
    dates = pd.bdate_range(start=start, periods=n)
    # deterministic, mildly noisy upward-trending series
    rng = np.random.default_rng(0)
    trend = np.linspace(100, 150, n)
    noise = rng.normal(0, 1.0, n)
    close = trend + noise
    return pd.DataFrame({"Close": close}, index=dates)


@pytest.fixture(autouse=True)
def patch_yfinance(monkeypatch):
    fake_df = _fake_prices()

    def fake_download(ticker, start=None, end=None, progress=False, auto_adjust=True, period=None, interval=None):
        return fake_df.copy()

    monkeypatch.setattr("yfinance.download", fake_download)
    yield


def test_make_windows_shapes_and_normalisation():
    from src import scaling

    prices = np.arange(1, 51).astype(float)
    X, y, refs = scaling.make_windows(prices, seq_length=10)

    assert X.shape == (40, 10, 1)
    assert y.shape == (40, 1)
    assert refs.shape == (40,)
    # Each window is normalised against its own last close, so that entry
    # is 0 by construction and refs holds the decision-day price.
    assert np.allclose(X[:, -1, 0], 0.0)
    assert np.allclose(refs, prices[9:49])
    # y is the true next-day return relative to that reference.
    assert np.allclose(y.reshape(-1), prices[10:] / prices[9:49] - 1.0)


def test_scaling_is_invariant_to_price_level():
    """The whole point of window-relative returns: a 1% move looks the same
    at NT$100 and at NT$2350, so a model trained cheap stays valid dear."""
    from src import scaling

    prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0])
    X_cheap, y_cheap, _ = scaling.make_windows(prices, seq_length=3)
    X_dear, y_dear, refs_dear = scaling.make_windows(prices * 23.5, seq_length=3)

    assert np.allclose(X_cheap, X_dear)
    assert np.allclose(y_cheap, y_dear)
    # ...and the inverse still lands back on real prices.
    assert np.allclose(scaling.to_price(y_dear, refs_dear), prices[3:] * 23.5)


def test_download_close_normalizes_columns():
    df = data.download_close("FAKE.TW", start="2022-01-01")
    assert list(df.columns) == ["Close"]
    assert len(df) > 0


def test_prepare_dataset_splits():
    seq, ratio = 10, 0.8
    n_rows = len(_fake_prices())
    ds = data.prepare_dataset("FAKE.TW", start="2022-01-01", seq_length=seq, split_ratio=ratio)

    n_windows = n_rows - seq
    assert ds["X_train"].shape[1:] == (seq, 1)
    assert len(ds["X_train"]) + len(ds["X_test"]) == n_windows
    assert len(ds["X_train"]) == int(n_windows * ratio)
    assert len(ds["refs_train"]) == len(ds["X_train"])
    assert len(ds["refs_test"]) == len(ds["X_test"])
    assert len(ds["dates"]) == n_windows


@pytest.mark.parametrize("model_type", config.VALID_MODEL_TYPES)
def test_build_model_forward_pass(model_type):
    m = model_mod.build_model(model_type, seq_length=10)
    dummy = np.random.rand(4, 10, 1).astype("float32")
    out = m.predict(dummy, verbose=0)
    assert out.shape == (4, 1)


def test_train_and_predict_roundtrip(tmp_path, monkeypatch):
    """End-to-end: train a tiny model, save weights + preprocessing spec,
    then load them back via predict.load_trained and run a forward pass."""
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "weights_path", lambda ticker, mt="LSTM": tmp_path / f"best_{mt}_{ticker}.weights.h5")
    monkeypatch.setattr(config, "preproc_path", lambda ticker, mt="LSTM": tmp_path / f"preproc_{mt}_{ticker}.json")

    from src.train import train
    from src.predict import load_trained

    model, ds, metrics = train(ticker="FAKE.TW", model_type="LSTM", start="2022-01-01", epochs=1, batch_size=32)
    assert (tmp_path / "best_LSTM_FAKE.TW.weights.h5").exists()
    assert (tmp_path / "preproc_LSTM_FAKE.TW.json").exists()
    assert "skill_vs_naive" in metrics

    loaded_model, spec = load_trained("FAKE.TW", "LSTM")
    assert spec["seq_length"] == config.SEQ_LENGTH
    assert spec["feature_set"] == "close"
    assert spec["feature_names"] == ["Close"]
    dummy = np.random.rand(1, spec["seq_length"], 1).astype("float32")
    out = loaded_model.predict(dummy, verbose=0)
    assert out.shape == (1, 1)


def test_load_trained_rejects_a_foreign_preprocessing_spec(tmp_path, monkeypatch):
    """Weights must never be served under a transform they weren't trained
    on - that silent mismatch is what broke the original saved models."""
    import json

    monkeypatch.setattr(config, "weights_path", lambda ticker, mt="LSTM": tmp_path / f"best_{mt}_{ticker}.weights.h5")
    monkeypatch.setattr(config, "preproc_path", lambda ticker, mt="LSTM": tmp_path / f"preproc_{mt}_{ticker}.json")

    from src.predict import load_trained
    from src.model import build_model

    build_model("LSTM").save_weights(str(tmp_path / "best_LSTM_FAKE.TW.weights.h5"))
    (tmp_path / "preproc_LSTM_FAKE.TW.json").write_text(json.dumps({"mode": "minmax_v1", "seq_length": 10}))

    with pytest.raises(ValueError, match="minmax_v1"):
        load_trained("FAKE.TW", "LSTM")


def test_preproc_spec_is_keyed_by_architecture_not_just_ticker():
    """Two architectures on one ticker must not share a preprocessing spec.

    weights_path has always been keyed by (ticker, model_type). When
    preproc_path was keyed by ticker alone, training a GRU on 'technical'
    overwrote the spec belonging to a close-only LSTM, and the LSTM's
    weights were then rebuilt with the wrong input width.
    """
    lstm = config.preproc_path("2330.TW", "LSTM")
    gru = config.preproc_path("2330.TW", "GRU")

    assert lstm != gru
    assert lstm.parent == config.weights_path("2330.TW", "LSTM").parent


def test_load_trained_rejects_a_spec_from_another_architecture(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(config, "weights_path", lambda ticker, mt="LSTM": tmp_path / f"best_{mt}_{ticker}.weights.h5")
    monkeypatch.setattr(config, "preproc_path", lambda ticker, mt="LSTM": tmp_path / f"preproc_{mt}_{ticker}.json")

    from src.predict import load_trained
    from src.model import build_model
    from src import scaling

    build_model("LSTM").save_weights(str(tmp_path / "best_LSTM_FAKE.TW.weights.h5"))
    (tmp_path / "preproc_LSTM_FAKE.TW.json").write_text(json.dumps({
        "mode": scaling.MODE, "seq_length": 10, "ticker": "FAKE.TW",
        "model_type": "GRU", "feature_set": "close", "feature_names": ["Close"],
    }))

    with pytest.raises(ValueError, match="GRU"):
        load_trained("FAKE.TW", "LSTM")


def test_training_validation_never_touches_the_test_set(tmp_path, monkeypatch):
    """Model selection must not be scored on the rows we then report on.

    train.py used to pass (X_test, y_test) as validation_data and checkpoint
    on val_loss, then print a test-set skill score for the weights it had
    just chosen using that same data - the mistake README section 4.4 is
    about. The validation split must come out of the training segment.
    """
    monkeypatch.setattr(config, "weights_path", lambda ticker, mt="LSTM": tmp_path / f"best_{mt}_{ticker}.weights.h5")
    monkeypatch.setattr(config, "preproc_path", lambda ticker, mt="LSTM": tmp_path / f"preproc_{mt}_{ticker}.json")

    from src import train as train_mod

    seen = {}
    real_fit = train_mod.build_model

    def spy_build(*args, **kwargs):
        model = real_fit(*args, **kwargs)
        original = model.fit

        def fit(x, y, **kw):
            seen["x"] = x
            seen["validation_data"] = kw.get("validation_data")
            return original(x, y, **kw)

        model.fit = fit
        return model

    monkeypatch.setattr(train_mod, "build_model", spy_build)
    train_mod.train(ticker="FAKE.TW", model_type="LSTM", start="2022-01-01", epochs=1)

    ds = data.prepare_dataset("FAKE.TW", start="2022-01-01")
    val_x = seen["validation_data"][0]
    assert val_x is not None and len(val_x) > 0

    # Validation rows must be training rows, and none of them may appear in
    # the test set.
    assert len(seen["x"]) + len(val_x) == len(ds["X_train"])
    test_rows = {ds["X_test"][i].tobytes() for i in range(len(ds["X_test"]))}
    assert not any(val_x[i].tobytes() in test_rows for i in range(len(val_x)))
    # ...and the split is chronological: validation is the tail of training.
    assert np.allclose(val_x, ds["X_train"][len(seen["x"]):])


def test_benchmark_returns_keep_the_first_day(monkeypatch):
    """The market index must not get a free flat day at the window's start."""
    from src.backtest import _benchmark_returns

    idx = pd.bdate_range("2024-01-01", periods=6)
    close = pd.Series([100.0, 110.0, 121.0, 121.0, 121.0, 121.0], index=idx)
    common = idx[1:]          # out-of-sample window starts on day 2

    ret = _benchmark_returns(close, common)

    # Day 2's return is +10% against day 1's close, not 0.
    assert ret.iloc[0] == pytest.approx(0.10)
    # Compounding over the window matches the underlying price move.
    assert float((1 + ret).prod()) == pytest.approx(close.iloc[-1] / close.iloc[0])


def test_no_fitted_state_can_leak_across_the_split():
    """Regression test for the scaling data leak.

    The old pipeline fitted a MinMaxScaler on the whole series before
    splitting, so the training set implicitly knew the test period's price
    range. Window-relative scaling has no fitted state at all: the windows
    a split produces must be byte-identical to the ones built from the test
    segment on its own.
    """
    from src import scaling

    prices = _fake_prices()["Close"].values
    ds = data.prepare_dataset("FAKE.TW", start="2022-01-01", seq_length=10, split_ratio=0.8)

    # Rebuild the test windows from the tail alone - no training data in sight.
    tail = prices[ds["train_size"]:]
    X_alone, y_alone, refs_alone = scaling.make_windows(tail, seq_length=10)

    assert np.allclose(ds["X_test"], X_alone)
    assert np.allclose(ds["y_test"], y_alone)
    assert np.allclose(ds["refs_test"], refs_alone)


def test_refs_are_the_decision_day_close():
    """refs_test must be the last close *inside* each window, never the target."""
    ds = data.prepare_dataset("FAKE.TW", start="2022-01-01", seq_length=10, split_ratio=0.8)
    prices = _fake_prices()["Close"].values

    targets = prices[10:][ds["train_size"]:]
    assert np.allclose(ds["refs_test"], prices[9:-1][ds["train_size"]:])
    assert not np.allclose(ds["refs_test"], targets), "refs leaked the target day"


def test_naive_forecast_is_the_reference_price():
    from src.metrics import naive_forecast

    ds = data.prepare_dataset("FAKE.TW", start="2022-01-01", seq_length=10, split_ratio=0.8)
    assert np.allclose(naive_forecast(ds["refs_test"]), ds["refs_test"])


def test_positions_have_no_lookahead_and_pay_costs():
    """A wrong prediction must lose money, and trading must cost something."""
    from src.backtest import _positions_and_returns

    # Day 1: model says up, price actually falls 10% -> long, loses.
    # Day 2: model says down, price rises -> flat, earns nothing but pays exit cost.
    frame = pd.DataFrame(
        {"prev": [100.0, 90.0], "close": [90.0, 99.0], "pred": [110.0, 80.0]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    position, net_ret, delta = _positions_and_returns(frame, threshold=0.0, fee=0.001425, tax=0.003)

    assert list(position) == [1.0, 0.0]
    assert net_ret.iloc[0] < -0.10          # -10% move plus entry fee
    assert net_ret.iloc[1] < 0              # flat, but paid the exit fee + tax
    assert list(delta) == [1.0, -1.0]


def test_backtest_never_trains_on_out_of_sample_rows(monkeypatch):
    """_oos_predictions must fit only on X_train and score only on X_test."""
    from src import backtest as bt

    seen = {}
    real_prepare = data.prepare_dataset

    def spy_prepare(ticker, **kwargs):
        ds = real_prepare(ticker, **kwargs)
        seen["train_len"] = len(ds["X_train"])
        seen["test_len"] = len(ds["X_test"])
        return ds

    monkeypatch.setattr(bt, "prepare_dataset", spy_prepare)

    frame = bt._oos_predictions("FAKE.TW", "2022-01-01", None, 10, 0.8, epochs=1, batch_size=32)

    assert len(frame) == seen["test_len"]
    assert len(frame) < seen["train_len"]
    assert list(frame.columns) == ["prev", "close", "pred"]


@pytest.mark.parametrize("feature_set", ["close", "technical", "full"])
def test_features_are_causal(feature_set):
    """The headline guarantee: no feature value may depend on the future.

    Computing the indicators on a truncated series must reproduce exactly
    the values the full series gives for those same rows. A centred
    smoother (e.g. scipy's gaussian_filter1d) fails this test, which is why
    features.py uses a one-sided kernel.
    """
    from src.features import build_features

    full, _ = build_features("FAKE.TW", start="2022-01-01", feature_set=feature_set)
    cutoff = len(full) - 30
    assert cutoff > 0

    # Rebuild using only data up to the cutoff date.
    truncated, _ = build_features(
        "FAKE.TW", start="2022-01-01", end=str(full.index[cutoff].date()),
        feature_set=feature_set,
    )
    overlap = truncated.index.intersection(full.index)
    assert len(overlap) > 20

    pd.testing.assert_frame_equal(
        full.loc[overlap], truncated.loc[overlap], check_exact=False, rtol=1e-9
    )


def test_causal_gaussian_trend_ignores_the_future():
    """Direct check on the smoother, independent of the download path."""
    from src.features import causal_gaussian_trend

    base = pd.Series(np.linspace(100, 120, 80))
    spiked = base.copy()
    spiked.iloc[60] = 1000.0  # a huge shock, strictly after row 50

    assert causal_gaussian_trend(base).iloc[50] == pytest.approx(
        causal_gaussian_trend(spiked).iloc[50]
    )
    # ...and the shock does move the trend once it is in the past.
    assert causal_gaussian_trend(base).iloc[61] != pytest.approx(
        causal_gaussian_trend(spiked).iloc[61]
    )


def test_macro_features_are_lagged():
    """Macro returns must be shifted, or the US close leaks into the TW day."""
    from src import features

    idx = pd.bdate_range("2022-01-03", periods=40)
    macro = features._macro_frame(idx, start="2022-01-01", end=None)

    # The fixture serves one synthetic series for every ticker, so the
    # unlagged return is recoverable and we can assert the shift directly.
    raw = _fake_prices()["Close"].reindex(idx.union(_fake_prices().index)).ffill().reindex(idx)
    expected = raw.pct_change().shift(config.MACRO_LAG_DAYS)

    assert config.MACRO_LAG_DAYS >= 1
    for col in macro.columns:
        pd.testing.assert_series_equal(
            macro[col], expected, check_names=False, rtol=1e-9
        )


def test_feature_windows_normalise_price_channels_only():
    from src import scaling
    from src.features import KIND_BOUNDED, KIND_PRICE, KIND_RELATIVE

    frame = pd.DataFrame({
        "Close": [10.0, 11.0, 12.0, 13.0],
        "Trend": [10.0, 10.5, 11.0, 11.5],
        "RSI": [0.2, 0.3, 0.4, 0.5],
        "Bias": [0.01, 0.02, 0.03, 0.04],
    })
    kinds = {"Close": KIND_PRICE, "Trend": KIND_PRICE, "RSI": KIND_BOUNDED, "Bias": KIND_RELATIVE}

    X, y, refs = scaling.make_feature_windows(frame, kinds, seq_length=3)

    assert X.shape == (1, 3, 4)
    assert refs[0] == pytest.approx(12.0)
    # price channels become ratios to the reference close...
    assert X[0, :, 0] == pytest.approx([10 / 12 - 1, 11 / 12 - 1, 0.0])
    assert X[0, :, 1] == pytest.approx([10 / 12 - 1, 10.5 / 12 - 1, 11 / 12 - 1])
    # ...bounded and relative channels pass through untouched.
    assert X[0, :, 2] == pytest.approx([0.2, 0.3, 0.4])
    assert X[0, :, 3] == pytest.approx([0.01, 0.02, 0.03])
    assert y[0, 0] == pytest.approx(13 / 12 - 1)


def test_multi_feature_dataset_has_no_cross_split_state():
    """The leak-free property must survive the jump to 10 channels."""
    from src import scaling
    from src.features import build_features

    ds = data.prepare_dataset("FAKE.TW", start="2022-01-01", feature_set="technical")
    frame, kinds = build_features("FAKE.TW", start="2022-01-01", feature_set="technical")

    tail = frame.iloc[ds["train_size"]:]
    X_alone, y_alone, refs_alone = scaling.make_feature_windows(tail, kinds, ds["seq_length"])

    assert ds["n_features"] == 7
    assert np.allclose(ds["X_test"], X_alone)
    assert np.allclose(ds["y_test"], y_alone)
    assert np.allclose(ds["refs_test"], refs_alone)


def test_channel_standardisation_is_fitted_on_training_rows_only():
    """The one fitted transform in the project must not see the test rows.

    Scaling the test segment by an arbitrary constant must leave the
    training statistics - and therefore the transformed training data -
    bit-identical. A scaler fitted on the whole array would move.
    """
    from src import scaling

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 10, 3))
    X[:, :, 1] *= 100.0          # a deliberately over-scaled channel
    cut = 160

    stats = scaling.fit_channel_stats(X[:cut])

    tampered = X.copy()
    tampered[cut:] *= 50.0
    tampered_stats = scaling.fit_channel_stats(tampered[:cut])

    assert np.allclose(stats[0], tampered_stats[0])
    assert np.allclose(stats[1], tampered_stats[1])

    # After the transform the training channels are comparable...
    train_z = scaling.apply_channel_stats(X[:cut], stats)
    assert np.allclose(train_z.mean(axis=(0, 1)), 0.0, atol=1e-12)
    assert np.allclose(train_z.std(axis=(0, 1)), 1.0, atol=1e-12)
    # ...and the test rows are transformed with the training statistics,
    # not their own (otherwise their std would also be exactly 1).
    test_z = scaling.apply_channel_stats(X[cut:], stats)
    assert not np.allclose(test_z.std(axis=(0, 1)), 1.0, atol=1e-6)


def test_constant_channels_survive_standardisation():
    """A zero-variance channel must pass through, not divide by zero."""
    from src import scaling

    X = np.ones((50, 10, 2))
    X[:, :, 0] = np.linspace(0, 1, 50)[:, None]

    out = scaling.apply_channel_stats(X, scaling.fit_channel_stats(X))
    assert np.all(np.isfinite(out))
    assert np.allclose(out[:, :, 1], 0.0)


def test_prepare_dataset_standardise_flag_matches_manual_transform():
    from src import scaling

    raw = data.prepare_dataset("FAKE.TW", start="2022-01-01", feature_set="technical")
    std = data.prepare_dataset("FAKE.TW", start="2022-01-01", feature_set="technical",
                               standardise=True)

    stats = scaling.fit_channel_stats(raw["X_train"])
    assert np.allclose(std["X_train"], scaling.apply_channel_stats(raw["X_train"], stats))
    assert np.allclose(std["X_test"], scaling.apply_channel_stats(raw["X_test"], stats))
    # Targets and reference prices are untouched, so predictions still
    # convert back into prices the same way.
    assert np.allclose(std["y_test"], raw["y_test"])
    assert np.allclose(std["refs_test"], raw["refs_test"])
    assert raw["channel_stats"] is None


def test_direction_metrics_use_majority_not_fifty_percent():
    from src.metrics import direction_metrics

    # 70% of days are up; a model that always says "up" scores 0.70 and
    # must be reported as having zero edge, not +20 points over 50%.
    truth = np.array([1.0] * 70 + [0.0] * 30)
    always_up = np.ones(100)

    m = direction_metrics(truth, always_up)
    assert m["accuracy"] == pytest.approx(0.70)
    assert m["majority_accuracy"] == pytest.approx(0.70)
    assert m["edge_vs_majority"] == pytest.approx(0.0)
    assert m["p_value"] > 0.99


def test_classification_model_outputs_probabilities():
    from src.model import CLASSIFICATION, build_model

    m = build_model("LSTM", seq_length=10, n_features=7, task=CLASSIFICATION)
    out = m.predict(np.random.rand(4, 10, 7).astype("float32"), verbose=0)
    assert out.shape == (4, 1)
    assert np.all((out >= 0) & (out <= 1))


def test_walk_forward_splits_never_train_on_the_future():
    """Every fold's training range must end exactly where its test begins."""
    from src.walkforward import EXPANDING, ROLLING, walk_forward_splits

    for mode in (EXPANDING, ROLLING):
        splits = walk_forward_splits(1000, n_folds=5, min_train_ratio=0.5, mode=mode)
        assert len(splits) == 5
        for train_start, train_end, test_end in splits:
            assert 0 <= train_start < train_end < test_end <= 1000

        # Test blocks must tile the tail contiguously, with no overlap...
        starts = [train_end for _, train_end, _ in splits]
        ends = [test_end for _, _, test_end in splits]
        assert starts == [500, 600, 700, 800, 900]
        assert ends == [600, 700, 800, 900, 1000]
        # ...and the folds together must cover every held-out row exactly once.
        covered = [i for s, e in zip(starts, ends) for i in range(s, e)]
        assert covered == list(range(500, 1000))


def test_walk_forward_modes_differ_in_training_window():
    from src.walkforward import EXPANDING, ROLLING, walk_forward_splits

    expanding = walk_forward_splits(1000, n_folds=5, min_train_ratio=0.5, mode=EXPANDING)
    rolling = walk_forward_splits(1000, n_folds=5, min_train_ratio=0.5, mode=ROLLING)

    # Expanding keeps all history; rolling slides a fixed-length window.
    assert [s for s, _, _ in expanding] == [0] * 5
    assert [e - s for s, e, _ in expanding] == [500, 600, 700, 800, 900]
    assert [e - s for s, e, _ in rolling] == [500] * 5


def test_walk_forward_rejects_impossible_configurations():
    from src.walkforward import walk_forward_splits

    with pytest.raises(ValueError, match="too few"):
        walk_forward_splits(100, n_folds=60, min_train_ratio=0.5)
    with pytest.raises(ValueError, match="n_folds"):
        walk_forward_splits(1000, n_folds=0)
    with pytest.raises(ValueError, match="mode"):
        walk_forward_splits(1000, n_folds=5, mode="sideways")


def test_walk_forward_predictions_cover_the_held_out_tail():
    from src.walkforward import walk_forward_predictions

    frame, folds = walk_forward_predictions(
        "FAKE.TW", start="2022-01-01", n_folds=3, epochs=1, verbose=False,
    )
    assert list(frame.columns) == ["prev", "close", "pred", "fold"]
    assert frame.index.is_monotonic_increasing
    assert not frame.index.duplicated().any()
    assert sorted(folds["fold"]) == [0, 1, 2]
    # Later folds train on strictly more data (expanding mode).
    assert folds["train_days"].is_monotonic_increasing
    # Test blocks do not overlap in time.
    assert folds["test_end"].tolist() == sorted(folds["test_end"].tolist())


def test_prepare_windows_and_prepare_dataset_agree():
    """The split-free path and the split path must see identical windows."""
    from src.data import prepare_windows

    w = prepare_windows("FAKE.TW", start="2022-01-01", seq_length=10)
    ds = data.prepare_dataset("FAKE.TW", start="2022-01-01", seq_length=10, split_ratio=0.8)

    assert np.allclose(w["X"][: ds["train_size"]], ds["X_train"])
    assert np.allclose(w["X"][ds["train_size"]:], ds["X_test"])
    assert np.allclose(w["refs"][ds["train_size"]:], ds["refs_test"])


def _fake_runs(effect: float, noise: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """Synthetic ablation runs where 'technical' beats 'close' by `effect`."""
    rng = np.random.default_rng(seed)
    rows = []
    for ticker in ("A", "B", "C", "D", "E"):
        # Large per-ticker offset: the paired test must see through this.
        ticker_offset = rng.normal(0, 0.5)
        for s in range(5):
            shared = rng.normal(0, noise)
            for arm, bump in (("close", 0.0), ("technical", effect)):
                rows.append({
                    "ticker": ticker, "seed": s, "feature_set": arm,
                    "task": "classification",
                    "accuracy": 0.5 + ticker_offset + shared + bump + rng.normal(0, noise / 10),
                })
    return pd.DataFrame(rows)


def test_ablation_contrast_detects_a_real_effect():
    from src.experiment import ablation_contrasts

    result = ablation_contrasts(_fake_runs(effect=0.05)).iloc[0]
    assert result["arm"] == "technical"
    assert result["n_pairs"] == 25
    assert result["mean_difference"] == pytest.approx(0.05, abs=0.01)
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.01
    assert result["verdict"] == "arm better"


def test_ablation_contrast_reports_no_effect_when_there_is_none():
    """Guards against a test that would call noise a discovery."""
    from src.experiment import ablation_contrasts

    result = ablation_contrasts(_fake_runs(effect=0.0)).iloc[0]
    assert result["ci_low"] <= 0 <= result["ci_high"]
    assert result["verdict"] == "no measurable effect"


def test_ablation_contrast_is_paired_not_pooled():
    """The huge per-ticker spread must not leak into the contrast's CI.

    An unpaired comparison of these arms would have a confidence interval
    dominated by the 0.5-wide between-ticker variation and would miss the
    0.05 effect entirely; the paired one should not.
    """
    from src.experiment import ablation_contrasts

    runs = _fake_runs(effect=0.05)
    paired = ablation_contrasts(runs).iloc[0]
    width = paired["ci_high"] - paired["ci_low"]

    pooled_spread = runs.groupby("feature_set")["accuracy"].std().mean()
    assert width < pooled_spread / 10


# Between-ticker offsets are shared by both architectures (that is what the
# pairing removes); the per-run noise is drawn independently for each, so the
# differences are a real distribution rather than a constant.
_ARCH_TICKER_OFFSETS = {
    t: o for t, o in zip("ABCDE", np.random.default_rng(0).normal(0, 0.5, 5))
}


def _fake_arch_runs(offset: float, noise_seed: int, noise: float = 0.01) -> pd.DataFrame:
    """Synthetic runs for one architecture, shifted by `offset`."""
    rng = np.random.default_rng(noise_seed)
    return pd.DataFrame([
        {
            "ticker": ticker, "seed": s, "feature_set": "close",
            "task": "regression",
            "skill_vs_naive": base + offset + rng.normal(0, noise),
        }
        for ticker, base in _ARCH_TICKER_OFFSETS.items()
        for s in range(5)
    ])


def test_architecture_contrast_reports_indistinguishable_when_models_match():
    """The claim in README 3.2 is that the architectures cannot be told apart;
    the function must be able to say that rather than always ranking them."""
    from src.experiment import architecture_contrasts

    out = architecture_contrasts({
        "LSTM": _fake_arch_runs(0.0, noise_seed=1),
        "GRU": _fake_arch_runs(0.0, noise_seed=2),
    })
    row = out[out.task == "regression"].iloc[0]
    assert row["n_pairs"] == 25
    assert row["ci_low"] < 0 < row["ci_high"]      # a genuine interval, not degenerate
    assert row["verdict"] == "indistinguishable"
    assert row["p_value"] > 0.05


def test_architecture_contrast_detects_a_real_gap_through_ticker_noise():
    """A genuine 0.05 gap must survive the much larger between-ticker spread,
    which only the pairing makes possible."""
    from src.experiment import architecture_contrasts

    out = architecture_contrasts({
        "LSTM": _fake_arch_runs(0.05, noise_seed=3),
        "GRU": _fake_arch_runs(0.0, noise_seed=4),
    })
    row = out[out.task == "regression"].iloc[0]
    assert row["mean_difference"] == pytest.approx(0.05, abs=0.01)
    assert row["ci_low"] > 0
    assert row["verdict"] == "LSTM better"
    assert row["p_value"] < 0.01

    # Unpaired, the 0.5-wide between-ticker spread would swamp the 0.05 effect.
    lstm = _fake_arch_runs(0.05, noise_seed=3)["skill_vs_naive"]
    assert row["ci_high"] - row["ci_low"] < lstm.std() / 10


def test_sensitivity_regimes_differ_in_what_the_scaler_saw():
    """The leaky and leak-free MinMax arms must actually differ as claimed.

    Without this, the sensitivity table in README section 4.1 could be
    comparing two identical configurations.
    """
    from src.sensitivity import _minmax_windows

    prices = _fake_prices()["Close"].to_numpy()   # upward trending
    _, _, leaky, train_size = _minmax_windows(prices, 10, 0.8, fit_on_train=False)
    _, y_clean, clean, _ = _minmax_windows(prices, 10, 0.8, fit_on_train=True)

    # The leaky scaler saw the test period's high; the clean one did not.
    assert leaky.data_max_[0] == pytest.approx(prices.max())
    assert clean.data_max_[0] < prices.max()
    assert clean.data_max_[0] == pytest.approx(prices[: train_size + 10].max())
    # ...so leak-free scaling pushes test targets outside the fitted range.
    assert y_clean[train_size:].max() > 1.0


def test_sensitivity_regimes_are_scored_against_the_same_baseline():
    """All three regimes must be comparable: same split, same naive series."""
    from src import scaling, sensitivity

    prices = _fake_prices()["Close"].to_numpy()
    seq, ratio = 10, 0.8

    _, _, _, minmax_train_size = sensitivity._minmax_windows(prices, seq, ratio, True)
    X_win, _, _ = scaling.make_windows(prices, seq)

    assert minmax_train_size == int(len(X_win) * ratio)


def test_build_model_units_change_capacity_without_changing_the_interface():
    from src.model import build_model

    small = build_model("LSTM", seq_length=10, n_features=1, units=16)
    large = build_model("LSTM", seq_length=10, n_features=1, units=128)

    assert small.count_params() < large.count_params()
    dummy = np.random.rand(2, 10, 1).astype("float32")
    assert small.predict(dummy, verbose=0).shape == large.predict(dummy, verbose=0).shape

    with pytest.raises(ValueError, match="units"):
        build_model("LSTM", units=0)


def test_cnn_architecture_is_not_affected_by_the_units_sweep():
    """The saved CNN weights must stay loadable: only LSTM/GRU take `units`."""
    from src.model import build_model

    default = build_model("CNN", seq_length=10, n_features=1)
    swept = build_model("CNN", seq_length=10, n_features=1, units=128)
    assert default.count_params() == swept.count_params()


def test_tuning_grid_is_declared_not_searched():
    """The sweep must be a fixed grid; selecting a winner on the test set
    would be the very error section 4.4 is about."""
    from src import tuning

    assert tuning.GRID[0] == (config.DEFAULT_UNITS, 20, config.SEQ_LENGTH)
    assert len(set(tuning.GRID)) == len(tuning.GRID)
    assert all(u > 0 and e > 0 and s > 0 for u, e, s in tuning.GRID)


def test_tuning_summary_flags_only_intervals_above_zero():
    from src import tuning

    runs = pd.DataFrame([
        {"units": 50, "epochs": 20, "seq_length": 10, "n_params": 1, "skill_vs_naive": v}
        for v in (-0.01, -0.02, -0.015)
    ] + [
        {"units": 16, "epochs": 20, "seq_length": 10, "n_params": 1, "skill_vs_naive": v}
        for v in (0.05, 0.06, 0.055)
    ])
    summary = tuning.summarise(runs).set_index("units")

    assert not summary.loc[50, "beats_naive"]
    assert summary.loc[16, "beats_naive"]


def test_figures_fail_loudly_when_the_experiment_has_not_been_run(tmp_path, monkeypatch):
    """A missing artefact must say so, not produce an empty or stale chart."""
    from src import figures

    monkeypatch.setattr(figures.config, "REPORT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Run the experiments"):
        figures._latest("Ablation_*.xlsx")


def test_ablation_figure_ignores_the_standardised_report(tmp_path, monkeypatch):
    """'Ablation_LSTM_standardised_*' sorts after the plain report and would
    silently become 'the' ablation under a loose glob."""
    from src import figures

    monkeypatch.setattr(figures.config, "REPORT_DIR", tmp_path)
    for name in ("Ablation_LSTM_2026-07-26.xlsx",
                 "Ablation_LSTM_standardised_2026-07-28.xlsx"):
        (tmp_path / name).write_text("")

    assert figures._latest("Ablation_LSTM_2*.xlsx").name == "Ablation_LSTM_2026-07-26.xlsx"
    assert figures._latest("Ablation_LSTM_standardised_*.xlsx").name == (
        "Ablation_LSTM_standardised_2026-07-28.xlsx"
    )


def test_figures_pick_the_newest_matching_report(tmp_path, monkeypatch):
    from src import figures

    monkeypatch.setattr(figures.config, "REPORT_DIR", tmp_path)
    for name in ("Ablation_LSTM_2026-01-01.xlsx", "Ablation_LSTM_2026-07-26.xlsx"):
        (tmp_path / name).write_text("")

    assert figures._latest("Ablation_*.xlsx").name == "Ablation_LSTM_2026-07-26.xlsx"


def test_api_module_imports():
    """Make sure src.api imports cleanly (FastAPI app + pydantic model)."""
    from src import api
    assert api.app is not None


@pytest.mark.skipif(
    not config.weights_path(config.DEFAULT_TICKER, "LSTM").exists()
    or not config.preproc_path(config.DEFAULT_TICKER, "LSTM").exists(),
    reason="needs the committed weights for the default ticker",
)
def test_api_prediction_is_invariant_to_price_level():
    """The scale-invariance claim in README section 4.1, checked end to end.

    Feeding the same shape of window an order of magnitude cheaper must
    return an identical percentage move. This also exercises the real
    (weights, preproc) pair in models/, so weights that no longer match
    their architecture fail here rather than in production.
    """
    from fastapi import HTTPException
    from src.api import StockInput, predict_price

    prices = [2400, 2410, 2395, 2420, 2430, 2415, 2440, 2450, 2405, 2350]
    dear = predict_price(StockInput(ticker=config.DEFAULT_TICKER, model_type="LSTM",
                                    prices=prices))
    cheap = predict_price(StockInput(ticker=config.DEFAULT_TICKER, model_type="LSTM",
                                     prices=[p / 10 for p in prices]))

    assert dear["predicted_change_pct"] == pytest.approx(
        cheap["predicted_change_pct"], abs=1e-9
    )
    assert dear["predicted_next_close"] == pytest.approx(
        cheap["predicted_next_close"] * 10, rel=1e-9
    )

    # A window of the wrong length is a 400, not a silent reshape.
    with pytest.raises(HTTPException) as exc:
        predict_price(StockInput(ticker=config.DEFAULT_TICKER, model_type="LSTM",
                                 prices=prices[:5]))
    assert exc.value.status_code == 400
