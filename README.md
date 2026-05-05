# QuantTrade AI 📈

Production-grade, end-to-end **AI trading + backtesting platform** built in
Python. Vectorized indicators, rule-based and ML strategies, walk-forward
optimization, a Streamlit dashboard, and a FastAPI service — all wired up to
work fully offline (synthetic data fallback) and online (`ccxt` exchanges).

> Built as a complete reference quant stack: data → indicators → strategies →
> backtesting → AI predictions → risk management → optimization → deployment.

---

## ✨ Features

| Module | Capabilities |
| --- | --- |
| **Market data** | `ccxt` fetcher with synthetic-data fallback; Parquet storage; multi-timeframe |
| **Indicators** | SMA / EMA / MACD / RSI / Stochastic / Bollinger Bands / ATR / OBV / VWAP |
| **Strategies** | Moving-average crossover, RSI mean-reversion, Donchian breakout, custom registry |
| **Backtesting** | Vectorized engine, commission + slippage, look-ahead-safe, full trade log |
| **Metrics** | Total return, CAGR, Sharpe, Sortino, Max DD, Calmar, win rate, PF, expectancy, exposure |
| **AI prediction** | XGBoost classifier with `TimeSeriesSplit` CV; optional Keras LSTM |
| **Risk** | Stop-loss / take-profit (fixed-pct or ATR), position sizing, VaR / CVaR, risk-parity |
| **Optimization** | Grid search, walk-forward analysis, multi-strategy comparison |
| **Dashboard** | Streamlit with 6 pages (Plotly candles, indicator overlays, trade markers) |
| **API** | FastAPI service: `/strategies`, `/backtest`, `/compare`, `/predict` |
| **Deployment** | Dockerfile + docker-compose; GitHub Actions CI (lint, types, multi-version pytest) |

---

## 📂 Project layout

```
QuantTrade-AI/
├── app/
│   ├── api/              # FastAPI service
│   ├── backtesting/      # Engine, metrics, Trade dataclass
│   ├── dashboard/        # Reusable Plotly components
│   ├── data/             # ccxt fetcher, synthetic generator, Parquet store
│   ├── indicators/       # Vectorized technical indicators
│   ├── models/           # ML feature pipeline + XGBoost / LSTM
│   ├── optimization/     # Grid search, walk-forward, multi-strategy
│   ├── risk/             # SL/TP, sizing, portfolio risk
│   ├── strategies/       # Strategy ABC + MA / RSI / breakout
│   ├── utils/            # Logger, timeframes
│   └── config.py         # Pydantic settings
├── pages/                # Streamlit multi-page app (1..6)
├── tests/                # pytest suite (data, indicators, strategies,
│                         #   backtesting, risk, models, optimization, API)
├── scripts/              # Demo data seeder
├── main.py               # Typer CLI
├── streamlit_app.py      # Streamlit entrypoint
├── Dockerfile / docker-compose.yml
└── requirements/         # base.txt + dev.txt
```

---

## 🚀 Quickstart

### Local development (Python 3.11+)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt -r requirements/dev.txt
cp .env.example .env

# Run the test suite
pytest -ra

# Launch the Streamlit dashboard
streamlit run streamlit_app.py

# In another shell, run the FastAPI service
uvicorn app.api.main:app --reload --port 8000

# Use the CLI
python main.py strategies
python main.py backtest --strategy ma_crossover --symbol BTC/USDT --timeframe 1h
python main.py predict --symbol BTC/USDT --timeframe 1h
python main.py optimize --strategy ma_crossover
python main.py walkforward --strategy ma_crossover --folds 4
python main.py compare
```

### Docker

```bash
docker compose up --build
# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8000/docs
```

---

## 📐 Architecture

The platform is **strictly modular** and **vectorized end-to-end**:

```
┌─────────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────┐
│ Market Data     │ →  │ Indicators │ →  │ Strategies │ →  │ Backtesting  │
│ (ccxt + cache)  │    │ (numpy/pd) │    │ (Signals)  │    │ Engine       │
└─────────────────┘    └────────────┘    └────────────┘    └──────────────┘
                                              │                    │
                                              ▼                    ▼
                                        ┌────────────┐      ┌─────────────┐
                                        │ AI Models  │      │ Risk Mgr    │
                                        │ XGB / LSTM │      │ SL/TP/Size  │
                                        └────────────┘      └─────────────┘
                                              │                    │
                                              └─────────┬──────────┘
                                                        ▼
                                            ┌─────────────────────┐
                                            │ Optimization        │
                                            │ Grid / Walk-forward │
                                            └─────────────────────┘
                                                        │
                                                        ▼
                                            ┌─────────────────────┐
                                            │ Dashboard / API     │
                                            └─────────────────────┘
```

### Strategy contract

A `Strategy` consumes an OHLCV `DataFrame` and returns a `Signals` object with
an integer `position` in `{-1, 0, 1}` per bar. The backtester treats a position
change as an entry / exit / reversal and fills on the **next bar's open** to
avoid look-ahead bias.

```python
from app.data import fetch_ohlcv
from app.strategies import MovingAverageCrossover
from app.backtesting import BacktestEngine, BacktestConfig

df = fetch_ohlcv("BTC/USDT", "1h", 1_000)
strat = MovingAverageCrossover(fast=20, slow=50)
result = BacktestEngine(BacktestConfig(timeframe="1h")).run_strategy(df, strat)

print(result.metrics.to_dict())
print(f"{len(result.trades)} trades, final equity ${result.equity.iloc[-1]:,.2f}")
```

### AI prediction

```python
from app.data import fetch_ohlcv
from app.models import train_and_predict

df = fetch_ohlcv("BTC/USDT", "1h", 2_000)
model, report, pred = train_and_predict(df, horizon=1)
print(report)              # CV accuracy / AUC / log-loss
print(pred.signal.tail())  # In-sample {-1, 0, 1} signal
```

### Walk-forward analysis

```python
from app.data import fetch_ohlcv
from app.optimization import walk_forward

df = fetch_ohlcv("BTC/USDT", "1h", 2_000)
report = walk_forward(
    df,
    "ma_crossover",
    {"fast": [10, 20], "slow": [50, 100]},
    n_folds=4,
)
print(report.summary)
```

---

## 🧪 Quality gates

```bash
ruff check . && ruff format --check .
mypy app
pytest -ra --cov
```

CI runs:
- `ruff` lint + format on every push/PR.
- `mypy` (advisory).
- `pytest` on Python 3.11 and 3.12.

---

## 🔐 Configuration

All runtime knobs live in environment variables (or a `.env` file) prefixed
`QT_`. See [`.env.example`](./.env.example) for the full list. Never commit
`.env` files containing real exchange API keys.

---

## 🛣 Roadmap

- [ ] Live paper-trading loop via `ccxt` websocket
- [ ] Optuna-driven hyperparameter search
- [ ] Bayesian portfolio optimization
- [ ] Strategy attribution / factor decomposition
- [ ] Database persistence layer (PostgreSQL / TimescaleDB)

---

## 📄 License

MIT. Built by [Muhammaddiyor Tolibjonov](mailto:mtd.coder@gmail.com).
