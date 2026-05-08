"""
INDENG 231 Project 1 — Backtesting Simulation System
=====================================================
Single entry point. Loads the dataset and runs all experiments for
Deliverables 3, 4, and 5. All outputs are saved to results/.

Usage:
    python main.py

    All commands must be run from the project root directory (Project1/)
    so that relative paths resolve correctly.

Outputs:
    results/plots/  — combined and individual NAV curve PNGs
    results/logs/   — metrics tables (CSV + PNG) and JSON experiment logs
"""

from backtester import (
    load_prices,
    run_backtest,
    save_experiment_log,
    save_metrics_table,
    save_nav_plot,
)
from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum, MomentumPortfolio
from strategies.new_strategies import DualMomentumPullback, RiskAdjustedMomentum
from strategies.single_stock import Breakout, DualMomentum, TrendMeanReversionCombo
from strategies.sma_crossover import SMAPortfolio
from strategies.top_k_momentum import TopKMomentum

PLOTS_DIR = "results/plots"
LOGS_DIR  = "results/logs"
TC        = 0.0005          # 5 basis points — realistic one-way transaction cost


def _label(name: str, cost: float) -> str:
    """Append a frictionless/cost suffix to an experiment label."""
    suffix = "frictionless" if cost == 0.0 else "5bps cost"
    return f"{name} — {suffix}"


def _run_pair(prices, factory, label: str) -> tuple[dict, dict]:
    """Run a strategy twice (frictionless + with cost) using a fresh instance each time.

    Using a factory callable ensures stateful strategies (MeanReversion, Breakout)
    start with clean state for each independent run.
    """
    r_free = run_backtest(prices, factory(), transaction_cost=0.0,
                          name_override=_label(label, 0.0))
    r_cost = run_backtest(prices, factory(), transaction_cost=TC,
                          name_override=_label(label, TC))
    return r_free, r_cost


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("INDENG 231 Project 1 — Backtesting Simulation System")
    print("=" * 70)

    prices = load_prices("data/nasdaq100_daily_5y.csv")
    print(f"\nLoaded: {prices.shape[0]} trading days × {prices.shape[1]} tickers")
    print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}\n")

    # Collect (section_label, strategy_label, sharpe) for the final summary
    summary: list[tuple[str, str, float]] = []

    # ── DELIVERABLE 3: Single-stock NVDA strategy evaluation ─────────────────
    print("\n" + "─" * 70)
    print("DELIVERABLE 3 — Single-Stock Strategy Evaluation (NVDA)")
    print("─" * 70)

    d3_specs = [
        ("Momentum (20d)",
         lambda: Momentum(ticker="NVDA", lookback=20)),
        ("Mean Reversion (20d)",
         lambda: MeanReversion(ticker="NVDA", window=20)),
        ("Dual Momentum (20/60d)",
         lambda: DualMomentum(ticker="NVDA", short_window=20, long_window=60)),
        ("Breakout (252d high / 20d low)",
         lambda: Breakout(ticker="NVDA", breakout_window=252, breakdown_window=20)),
        ("Trend + MR Combo (60d trend, z<-0.5)",
         lambda: TrendMeanReversionCombo(ticker="NVDA")),
    ]

    d3_results = []
    for label, factory in d3_specs:
        r_free, r_cost = _run_pair(prices, factory, label)
        d3_results.extend([r_free, r_cost])
        summary.append(("D3 — NVDA Single Stock", label, r_free["metrics"]["sharpe_ratio"]))

    save_nav_plot(d3_results, PLOTS_DIR, "d3_single_stock_nav",
                  title="D3: NVDA Single-Stock Strategies — Frictionless vs 5bps Cost")
    save_metrics_table(d3_results, LOGS_DIR, "d3_single_stock_metrics")
    save_experiment_log(d3_results, LOGS_DIR, "d3_single_stock_log")

    # ── DELIVERABLE 4: Portfolio backtesting — 2×2 experiment grid ───────────
    print("\n" + "─" * 70)
    print("DELIVERABLE 4 — Portfolio Backtesting (2×2 Signal × Weighting Grid)")
    print("─" * 70)

    d4_specs = [
        ("Momentum Top 25% — Equal Wt",
         lambda: MomentumPortfolio(lookback=20, top_pct=0.25, weighting="equal")),
        ("Momentum Top 25% — Inv-Vol Wt",
         lambda: MomentumPortfolio(lookback=20, top_pct=0.25, weighting="inverse_vol")),
        ("SMA Crossover (20/50d) — Equal Wt",
         lambda: SMAPortfolio(short_window=20, long_window=50, weighting="equal")),
        ("SMA Crossover (20/50d) — Inv-Vol Wt",
         lambda: SMAPortfolio(short_window=20, long_window=50, weighting="inverse_vol")),
    ]

    d4_results = []
    for label, factory in d4_specs:
        r_free, r_cost = _run_pair(prices, factory, label)
        d4_results.extend([r_free, r_cost])
        summary.append(("D4 — Portfolio", label, r_free["metrics"]["sharpe_ratio"]))

    save_nav_plot(d4_results, PLOTS_DIR, "d4_portfolio_nav",
                  title="D4: Portfolio Backtesting — 2×2 Signal × Weighting Grid")
    save_metrics_table(d4_results, LOGS_DIR, "d4_portfolio_metrics")
    save_experiment_log(d4_results, LOGS_DIR, "d4_portfolio_log")

    # ── DELIVERABLE 5: Benchmarks vs new strategies ───────────────────────────
    print("\n" + "─" * 70)
    print("DELIVERABLE 5 — Beating the Benchmarks")
    print("─" * 70)

    d5_specs = [
        ("Benchmark 1 — SMA Crossover (20/50d)",
         lambda: SMAPortfolio(short_window=20, long_window=50, weighting="equal")),
        ("Benchmark 2 — Top-10 Momentum (30d)",
         lambda: TopKMomentum(k=10, lookback=30)),
        ("New Strategy 1 — Risk-Adj Momentum (Top 10)",
         lambda: RiskAdjustedMomentum(lookback=30, vol_window=20, k=10)),
        ("New Strategy 2 — Dual Mom + Pullback (20/40d, 3d)",
         lambda: DualMomentumPullback(short_window=20, long_window=40, pullback_window=3)),
    ]

    d5_results = []
    for label, factory in d5_specs:
        r_free, r_cost = _run_pair(prices, factory, label)
        d5_results.extend([r_free, r_cost])
        summary.append(("D5 — Benchmarks vs New", label, r_free["metrics"]["sharpe_ratio"]))

    save_nav_plot(d5_results, PLOTS_DIR, "d5_benchmarks_nav",
                  title="D5: New Strategies vs Benchmarks — Frictionless vs 5bps Cost")
    save_metrics_table(d5_results, LOGS_DIR, "d5_benchmarks_metrics")
    save_experiment_log(d5_results, LOGS_DIR, "d5_benchmarks_log")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("All experiments complete. Results saved to results/")
    print("=" * 70)

    current_section = ""
    for section, label, sharpe in summary:
        if section != current_section:
            print(f"\n{section} (frictionless Sharpe):")
            print(f"  {'Strategy':<48} {'Sharpe':>7}")
            print(f"  {'─' * 48} {'─' * 7}")
            current_section = section
        print(f"  {label:<48} {sharpe:>7.4f}")
    print()


if __name__ == "__main__":
    main()
