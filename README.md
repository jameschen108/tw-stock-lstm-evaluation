# Predicting Taiwan Stock Prices with an LSTM

[![tests](https://github.com/jameschen108/tw-stock-lstm-evaluation/actions/workflows/tests.yml/badge.svg)](https://github.com/jameschen108/tw-stock-lstm-evaluation/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Jing-Zhi Chen · jameschen108@gmail.com · [中文版](README.zh-TW.md)

Trained an LSTM to predict the next-day closing price of five Taiwan large caps
(2330 TSMC, 2317 Hon Hai, 2603 Evergreen, 2002 China Steel, 2891 CTBC), then used its
predictions to drive a long/flat trading strategy.

**It does not work.** The model does not beat "tomorrow's price = today's price", and the
strategy does not beat buying and holding the same five stocks. This README shows how I
measured that.

The approach follows Wang et al. (2025) [[1]](#reference), which compares LSTM, GRU and
CNN for stock prediction and serves them over FastAPI; I took the LSTM and applied it to
the Taiwan market.

## Results

| Question | Answer |
|---|---|
| Does the LSTM beat a "no change" baseline? | No. Skill −0.0062 (negative = worse than the baseline), 95% CI [−0.0091, −0.0033] |
| Do technical indicators and macro factors help? | No measurable difference (§2) |
| Does the trading strategy beat buy-and-hold? | No. Sharpe 1.00 vs 1.56, total return 112% vs 565% |

Direction accuracy is 0.4971 over 1,524 out-of-sample trading days — a coin flip.

![Walk-forward equity curves](reports/figures/equity_walkforward.png)

### Why "no change" is the baseline to beat

Stock prices barely move day to day, so predicting "tomorrow = today" already scores
R² = 0.9845, averaged over the five stocks (single 80/20 split, test period 2024–2026,
5 seeds). My model scores R² = 0.9841 — which looks excellent until you compare it to
the baseline it loses to.

| How you report it | Number | Impression |
|---|---|---|
| R² alone | 0.9841 | near perfect |
| vs "no change" baseline | −0.0073 | slightly worse than doing nothing |
| Direction accuracy vs 50% | 0.5184 | small edge |
| Direction accuracy vs "always up" | −0.0101 | worse than always guessing up |

Model RMSE is 7.5648; the baseline is 7.5504. That gap is the whole result.

## 1. What the model sees

Daily closes from yfinance since 2014. Each 10-day window is normalised against its own
last close, so the model predicts a **percentage move** rather than a price level:

```
ref     = the last close in the window
X[j]    = prices[j] / ref - 1
y       = tomorrow's close / ref - 1
```

This matters, because the obvious alternative — a `MinMaxScaler` — fails either way you
fit it. 2330 traded roughly 70–630 during the training period and reached 2510 by the end
of the test period. Fit the scaler on the training rows only (the leak-free way) and test
inputs land between 0.89 and 4.37, far outside the [0, 1] the network has seen, so it
saturates and under-predicts. Fit it over the whole series and everything stays inside
[0, 1] — but only because the test period's high and low set the boundaries, which is the
leak you were trying to avoid.

Measured on 2330 alone (same 80/20 split as above, seeds 0–2), same architecture, only
the normalisation differing. On this one stock the model and the baseline both reach
R² = 0.9968, so the small positive skill in the last row is a near-tie on 2330, not a
contradiction of the five-stock result above:

| Normalisation | RMSE | R² | vs baseline |
|---|---|---|---|
| MinMax fitted on training rows only | 310.96 | 0.6038 | −10.00 |
| MinMax fitted over the whole series (leaks) | 134.98 | 0.9203 | −3.77 |
| Window-relative returns (used here) | **28.15** | **0.9968** | **+0.0045** |

Note the ordering: leaking the test range makes the model look *better*, because it papers
over the range problem. Both are far worse than the baseline's RMSE of 28.28.

Features are computed so they only use past data. Two easy mistakes I had to avoid:
`scipy`'s `gaussian_filter1d` centres its kernel and so mixes in future prices (I use a
one-sided kernel), and US market data for day *t* only arrives after the Taiwan close, so
macro series are lagged one day.

## 2. Do extra features help?

Three feature sets, everything else identical: close only (1 input), plus 6 technical
indicators (7 inputs), plus 3 macro series (10 inputs). Five stocks × five seeds.

| Features | Skill vs baseline | 95% CI | Direction accuracy |
|---|---|---|---|
| close | −0.0073 | [−0.0130, −0.0016] | 0.5184 |
| technical | −0.0079 | [−0.0134, −0.0023] | 0.5158 |
| full | −0.0084 | [−0.0139, −0.0028] | 0.5172 |

Comparing each against the close-only control on the same stock and seed:

| Task | Arm | Difference | 95% CI | p |
|---|---|---|---|---|
| regression | technical | −0.00055 | [−0.00770, +0.00661] | 0.876 |
| regression | full | −0.00107 | [−0.00827, +0.00614] | 0.763 |
| classification | technical | −0.00266 | [−0.00613, +0.00082] | 0.127 |
| classification | full | −0.00127 | [−0.00511, +0.00257] | 0.501 |

![Paired ablation contrasts](reports/figures/ablation_contrast.png)

Every interval contains zero: the extra features made no measurable difference.

Before accepting that, I checked the obvious alternative explanation. The feature channels
are on wildly different scales — MACD histogram has σ ≈ 0.005, Bollinger position σ ≈ 0.63,
while the price-return channel the model is predicting sits at σ ≈ 0.02. Maybe the extra
features were never given a fair chance rather than carrying no signal. So I re-ran all
150 runs with every channel standardised, using training-set statistics only:

| Task | Arm | vs close control | 95% CI | p |
|---|---|---|---|---|
| regression | technical | −0.00967 | [−0.01637, −0.00297] | 0.007 |
| regression | full | −0.01034 | [−0.01720, −0.00347] | 0.005 |
| classification | technical | −0.00405 | [−0.00962, +0.00153] | 0.147 |
| classification | full | −0.00835 | [−0.01906, +0.00237] | 0.121 |

Putting the channels on equal footing made the multi-feature arms *worse*, not better.
So the answer isn't "the features were badly scaled": on this data and under this
evaluation, these particular indicators did not help.

## 3. LSTM, GRU or CNN?

I ran the same study for all three. Skill vs the baseline, close-only:

| | close | technical | full |
|---|---|---|---|
| LSTM | −0.0073 | −0.0079 | −0.0084 |
| GRU | −0.0049 | −0.0116 | −0.0135 |
| CNN | −0.0077 | −0.0582 | −0.0548 |

All 18 cells negative. Comparing the three head to head on the same stocks and seeds, all
the differences have confidence intervals containing zero — I can't tell them apart on
this data, and the LSTM comes out best in only one of the five stocks. I stayed with the
LSTM for the rest of the project to keep it aligned with the paper I started from, not
because it won here.

## 4. Out-of-sample results

A single 80/20 split only tests one stretch of market. Instead I retrain at five points in
time and only ever score on data the model hasn't seen — 1,524 trading days from 2020 to
2026, covering the pandemic rebound, the 2022 bear market and the 2023–2026 bull run.

| Fold | Test window | Skill | Direction accuracy |
|---|---|---|---|
| 0 | 2020-04 → 2021-07 | −0.0025 | 0.4919 |
| 1 | 2021-07 → 2022-10 | −0.0102 | 0.5250 |
| 2 | 2022-10 → 2024-01 | −0.0144 | 0.4660 |
| 3 | 2024-01 → 2025-04 | −0.0021 | 0.4932 |
| 4 | 2025-04 → 2026-07 | −0.0019 | 0.5095 |
| **All** | **2020-04 → 2026-07** | **−0.0062** | **0.4971** |

![Skill by fold](reports/figures/skill_by_fold.png)

No market regime produces positive skill. Of 75 stock-fold-seed cells, 15 are positive.

### Trading performance

Long/flat: hold a stock when the model predicts it will rise, otherwise sit in cash.
Costs are Taiwan retail rates — 0.1425% brokerage each side plus 0.3% transaction tax on
sells.

| Strategy | Total return | CAGR | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|---|
| LSTM long/flat (close) | 111.61% | 12.87% | 12.92% | 1.00 | −19.19% |
| LSTM long/flat (technical) | 176.88% | 17.67% | 14.59% | 1.17 | −21.92% |
| Buy & hold the same 5 stocks | 565.17% | 36.80% | 21.56% | **1.56** | −29.22% |
| Market index ^TWII | 312.36% | 26.40% | 20.19% | 1.26 | −31.63% |

The strategy is less volatile and has smaller drawdowns, but it still loses on a
risk-adjusted basis. It is only in the market 52.9% of the time, and in a market that
rises over the long run, the time in cash is where the return goes.

(^TWII excludes dividends while the five stocks are dividend-adjusted, so the index row
understates the market somewhat. The like-for-like comparison is against buy-and-hold of
the same five stocks.)

## 5. Two things that fooled me

**Exposure looks like skill.** On a single 2024–2026 split:

| Arm | Time in market | Total return | Sharpe |
|---|---|---|---|
| close | 49.5% | 23.40% | 0.79 |
| technical | 80.1% | 80.04% | 1.44 |

That looks like a huge win for feature engineering. It isn't — §2 shows the two are
statistically indistinguishable as predictors. The multi-feature model just says "up" more
often, so it stays invested longer in a rising market.

**One run tells you nothing.** Same model, same data, only the random seed changes:

| Features | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| close | 78.6% | 175.1% | 81.1% |
| technical | 119.1% | 105.4% | **306.1%** |

105% to 306% from weight initialisation alone. This is why every number above is an
average over multiple seeds with a confidence interval.

I also checked that the result isn't just an undertrained model: 8 configurations from
3.7k to 201k parameters, up to 60 epochs and 30-day windows. All negative, none beats the
baseline.

![Capacity sweep](reports/figures/tuning_capacity.png)

## 6. Limitations

* Daily closing prices only — no volume, fundamentals or news.
* Long/flat with daily rebalancing: no position sizing, stops or shorting.
* Costs cover fees and tax but not slippage; assumes execution at the close.
* Five stocks, one market. This says the approach doesn't work *here*, not that stock
  prediction is impossible.
* Seeds: 5 for the feature study, 3 for walk-forward, limited by compute.

## 7. Running it

Python 3.12.4, CPU only. Versions are pinned in `requirements.txt` because TensorFlow's
minor releases change the numbers.

```bash
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

51 tests, no network needed (`yfinance` is monkeypatched with synthetic data), ~15s. They
cover the things most likely to silently break the results: that features can't see the
future, that nothing is fitted across the train/test split, that walk-forward folds don't
overlap, that the backtest has no look-ahead and actually pays costs.

```bash
python -m src.train --ticker 2330.TW --model-type LSTM
python -m src.predict --ticker 2330.TW --days 5
```

Train one model and predict with it.

```bash
uvicorn src.api:app --port 8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"ticker":"2330.TW","model_type":"LSTM","prices":[2400,2410,2395,2420,2430,2415,2440,2450,2405,2350]}'
```

Serve predictions over HTTP.

```bash
python -m src.backtest --feature-set close      # single-split backtest
python -m src.walkforward --seeds 0 1 2         # walk-forward (§4)
python -m src.experiment --epochs 20            # feature study (§2)
python -m src.tuning                            # capacity check (§5)
python -m src.sensitivity --ticker 2330.TW      # normalisation check (§1)
python -m src.figures                           # rebuild all charts
```

A single 20-epoch training run takes ~5 seconds; the full feature study is ~100 minutes.
Every chart is rebuilt from the `.xlsx` artefacts in `reports/`; the tables above
summarise those same experiment outputs.

The reports hold more than this README shows: per-ticker breakdowns, rolling versus
expanding training windows, per-fold significance tests, and every individual run behind
each averaged figure.

```bash
docker build -t stockai . && docker run -p 8000:8000 stockai
```

## Project structure

```
src/
├── config.py       # all parameters in one place
├── data.py         # download and windowing
├── scaling.py      # window-relative normalisation
├── features.py     # technical indicators and macro factors
├── model.py        # LSTM / GRU / CNN
├── train.py        # train one model
├── predict.py      # load a model and forecast
├── api.py          # FastAPI service
├── backtest.py     # signal → position → costs → performance
├── walkforward.py  # retrain-and-roll evaluation
├── experiment.py   # feature study
├── tuning.py       # capacity sweep
├── sensitivity.py  # normalisation comparison
├── metrics.py      # metrics and baselines
└── figures.py      # rebuild charts from reports/

tests/    offline test suite
models/   trained weights + their preprocessing specs
reports/  results (.xlsx) and figures
```

## Reference

1. <a id="reference"></a>Wang, H., Rajakumar, V. S., Golec, M., Gill, S. S., and Uhlig, S.
   (2025). StockAICloud: AI-based sustainable and scalable stock price prediction
   framework using serverless cloud computing. *The Journal of Supercomputing*, 81(4), 527.
   https://doi.org/10.1007/s11227-025-06984-7

## Licence

MIT — see [LICENSE](LICENSE).
