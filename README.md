# indeng231-backtester

A modular trading strategy backtesting system built for INDENG 231 at UC Berkeley. The system simulates daily close trading strategies on five years of Nasdaq-100 historical price data (101 stocks, April 2021 to April 2026).

## What it does

The system lets you test any trading strategy on historical Nasdaq-100 data by implementing a single method. The backtesting engine, portfolio tracking, performance metrics, and reporting are all handled automatically. You just define the signal logic.

Nine strategies are implemented across three experiments:

- **Single-stock backtesting on NVDA:** momentum, mean reversion, dual momentum, breakout, and a trend plus mean reversion combo
- **Portfolio backtesting across all 101 stocks:** momentum and SMA crossover signals, each with equal and inverse volatility weighting
- **Benchmark comparison:** two spec-defined benchmarks vs two new strategies designed to beat them on Sharpe ratio (both succeed)

## Project structure

```
Project1/
├── backtester/          # Core engine: data loader, strategy base class,
│                        # engine, portfolio, metrics, reporter
├── strategies/          # One file per strategy family
├── notebooks/           # One notebook per deliverable for visualization
├── data/                # Place nasdaq100_daily_5y.csv here
├── results/             # Auto-generated plots and logs
├── main.py              # Single entry point
└── requirements.txt
```

## How to run

1. Clone the repo
2. Install dependencies with `pip install -r requirements.txt`
3. Download the dataset and place it at `data/nasdaq100_daily_5y.csv`
4. Run from the project root: `python main.py`

All NAV curves, metrics tables, and experiment logs are saved automatically to `results/`.

All commands must be run from the project root directory so relative paths resolve correctly.

## Key results

| Strategy | Sharpe | vs Benchmark |
|---|---|---|
| Benchmark 1 (SMA Crossover) | 0.66 | — |
| Benchmark 2 (Top-10 Momentum) | 1.03 | — |
| Risk-Adjusted Momentum | 1.07 | beats both |
| Dual Momentum + Pullback | 1.18 | beats both |

## Dependencies

Python 3.10 or later. Install with:

```bash
pip install -r requirements.txt
```
