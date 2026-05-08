import numpy as np
import pandas as pd

from backtester.metrics import compute_metrics, format_metrics
from backtester.portfolio import Portfolio


def run_backtest(
    prices: pd.DataFrame,
    strategy,
    transaction_cost: float = 0.0,
    name_override: str | None = None,
) -> dict:
    """Run a full backtest for one strategy over the given price history.

    Walks forward day by day. During the warmup period the portfolio holds
    cash (zero weights). Once warmup is complete the strategy is called every
    day to produce target weights, which are executed at the closing price.

    Args:
        prices: Wide-format DataFrame of shape (trading_days, n_tickers),
            DatetimeIndex sorted ascending. As returned by load_prices().
        strategy: Any instance of a Strategy subclass. Must implement
            get_weights(prices_so_far) and expose a warmup_period property.
        transaction_cost: Proportional cost per unit of turnover, e.g. 0.0005
            for 5 basis points. Applied symmetrically to buys and sells.
            Defaults to 0.0 (frictionless).
        name_override: If provided, this label is used in logs and the returned
            dict instead of strategy.name. Useful for labeling the same strategy
            under different conditions (e.g. "Momentum — frictionless" vs
            "Momentum — 5bps cost").

    Returns:
        Dictionary with keys:
            "strategy_name"   : str, the label used for this experiment
            "transaction_cost": float, the cost parameter used
            "nav_series"      : pd.Series, NAV curve indexed by date
            "metrics"         : dict, output of compute_metrics()
            "prices"          : pd.DataFrame, the original prices (for reporter)
    """
    name = name_override if name_override is not None else strategy.name
    n_assets = prices.shape[1]
    warmup = strategy.warmup_period

    portfolio = Portfolio(n_assets=n_assets)
    zero_weights = np.zeros(n_assets)

    for i, date in enumerate(prices.index):
        prices_today = prices.iloc[i]
        prices_yesterday = prices.iloc[i - 1] if i > 0 else None

        if i < warmup:
            target_weights = zero_weights
        else:
            if i == warmup and warmup > 0:
                print(f"[{name}] warmup complete, backtesting from {date.date()} ...")
            target_weights = strategy.get_weights(prices.iloc[: i + 1])

        portfolio.update(
            date=date,
            target_weights=target_weights,
            prices_today=prices_today,
            prices_yesterday=prices_yesterday,
            transaction_cost=transaction_cost,
        )

    nav_series = portfolio.get_nav_series()
    metrics = compute_metrics(nav_series)

    cost_label = f"{transaction_cost:.4%}" if transaction_cost > 0 else "frictionless"
    print(f"\n[{name}] ({cost_label})")
    print(format_metrics(metrics))
    print()

    return {
        "strategy_name": name,
        "transaction_cost": transaction_cost,
        "nav_series": nav_series,
        "metrics": metrics,
        "prices": prices,
    }
