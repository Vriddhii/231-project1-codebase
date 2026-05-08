import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(nav_series: pd.Series) -> dict:
    """Compute standard backtesting performance metrics from a NAV curve.

    Args:
        nav_series: pd.Series of portfolio NAV values indexed by date,
            starting at 1.0. Must be sorted chronologically.

    Returns:
        Dictionary with keys:
            cumulative_return    : total return over the period (e.g. 2.50 = 250%)
            annualized_volatility: annualized std of daily returns
            sharpe_ratio         : annualized Sharpe (risk-free rate = 0)
            max_drawdown         : largest peak-to-trough decline (e.g. 0.33 = 33%)
            win_rate             : fraction of days with a positive return
        All values are 0.0 if the series has fewer than 2 data points.
    """
    empty = {
        "cumulative_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
    }

    if nav_series is None or len(nav_series) < 2:
        return empty

    nav = nav_series.values.astype(float)
    daily_returns = np.diff(nav) / nav[:-1]

    # Cumulative return
    cumulative_return = (nav[-1] / nav[0]) - 1.0

    # Annualized volatility
    vol_daily = float(np.std(daily_returns, ddof=1))
    annualized_volatility = vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe ratio (risk-free rate = 0, annualized)
    mean_daily = float(np.mean(daily_returns))
    if vol_daily == 0.0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (mean_daily * TRADING_DAYS_PER_YEAR) / (vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR))

    # Maximum drawdown
    peak = np.maximum.accumulate(nav)
    drawdowns = (peak - nav) / peak
    max_drawdown = float(np.max(drawdowns))

    # Win rate
    win_rate = float(np.sum(daily_returns > 0) / len(daily_returns))

    return {
        "cumulative_return": float(cumulative_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
    }


def format_metrics(metrics: dict) -> str:
    """Format a metrics dict into a human-readable string for logs and console.

    Args:
        metrics: Dictionary returned by compute_metrics().

    Returns:
        Multi-line string with aligned labels and formatted values.
    """
    lines = [
        f"  Cumulative Return : {metrics['cumulative_return']:+.2%}",
        f"  Annual Volatility : {metrics['annualized_volatility']:>8.2%}",
        f"  Sharpe Ratio      : {metrics['sharpe_ratio']:>8.2f}",
        f"  Max Drawdown      : {-metrics['max_drawdown']:+.2%}",
        f"  Win Rate          : {metrics['win_rate']:>8.2%}",
    ]
    return "\n".join(lines)
