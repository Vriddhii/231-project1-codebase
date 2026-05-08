import numpy as np
import pandas as pd

from backtester import Strategy


def _inv_vol_weights(prices_so_far: pd.DataFrame, selected: list[int], vol_window: int = 20) -> np.ndarray:
    """Return inverse-volatility normalized weights for the given column indices."""
    weights = np.zeros(prices_so_far.shape[1])
    inv_vols: dict[int, float] = {}
    for idx in selected:
        px = prices_so_far.iloc[:, idx].dropna().values
        if len(px) < vol_window + 1:
            continue
        daily_rets = np.diff(px[-(vol_window + 1):]) / px[-(vol_window + 1):-1]
        vol = float(np.std(daily_rets, ddof=1))
        if vol > 0:
            inv_vols[idx] = 1.0 / vol
    total = sum(inv_vols.values())
    if total > 0:
        for idx, iv in inv_vols.items():
            weights[idx] = iv / total
    elif selected:
        for idx in selected:
            weights[idx] = 1.0 / len(selected)
    return weights


class Momentum(Strategy):
    """Single-ticker momentum strategy.

    Allocates fully to the target ticker when its return over the lookback
    window is positive; holds cash otherwise. The simplest expression of
    trend-following: if it went up recently, keep riding it.

    Used for both single-stock (NVDA) and portfolio signal generation.
    In single-stock mode, pass ticker='NVDA'. In portfolio mode, subclass
    or override to produce a weight vector across multiple tickers.

    Args:
        ticker: Column name of the stock to trade. Default 'NVDA'.
        lookback: Number of days used to compute the trailing return. Default 20.
    """

    def __init__(self, ticker: str = "NVDA", lookback: int = 20) -> None:
        self.ticker = ticker
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"Momentum ({self.lookback}d) — {self.ticker}"

    @property
    def warmup_period(self) -> int:
        return self.lookback

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        weights = np.zeros(prices_so_far.shape[1])

        if self.ticker not in prices_so_far.columns:
            return weights

        prices = prices_so_far[self.ticker].values
        today = prices[-1]
        past = prices[-self.lookback - 1]  # price lookback days ago

        if np.isnan(today) or np.isnan(past) or past == 0:
            return weights

        ret = today / past - 1.0
        if ret > 0:
            idx = prices_so_far.columns.get_loc(self.ticker)
            weights[idx] = 1.0

        return weights


class MomentumPortfolio(Strategy):
    """Cross-sectional momentum portfolio across the full Nasdaq-100 universe.

    Ranks all tickers by trailing return and selects the top fraction. Supports
    two weighting schemes: equal weight (all selected tickers get 1/N) and
    inverse-volatility weight (allocation proportional to 1/rolling_vol,
    normalised to sum to 1).

    Args:
        lookback: Days used to compute trailing return. Default 20.
        top_pct: Fraction of tickers to select (e.g. 0.25 = top 25%). Default 0.25.
        weighting: 'equal' or 'inverse_vol'. Default 'equal'.
        vol_window: Days of returns used for volatility estimate. Default 20.
    """

    def __init__(
        self,
        lookback: int = 20,
        top_pct: float = 0.25,
        weighting: str = "equal",
        vol_window: int = 20,
    ) -> None:
        self.lookback = lookback
        self.top_pct = top_pct
        self.weighting = weighting
        self.vol_window = vol_window

    @property
    def name(self) -> str:
        w_label = "Equal Wt" if self.weighting == "equal" else "Inv-Vol Wt"
        return f"Momentum Top {int(self.top_pct * 100)}% ({self.lookback}d) — {w_label}"

    @property
    def warmup_period(self) -> int:
        return self.lookback

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        returns: dict[int, float] = {}

        for idx, col in enumerate(prices_so_far.columns):
            px = prices_so_far[col].dropna().values
            if len(px) < self.lookback + 1:
                continue
            if np.isnan(prices_so_far[col].iloc[-1]):
                continue
            today = px[-1]
            past = px[-self.lookback - 1]
            if past == 0 or np.isnan(past):
                continue
            returns[idx] = today / past - 1.0

        n_select = max(1, round(len(returns) * self.top_pct))
        top = sorted(returns, key=returns.__getitem__, reverse=True)[:n_select]

        if not top:
            return np.zeros(prices_so_far.shape[1])

        if self.weighting == "inverse_vol":
            return _inv_vol_weights(prices_so_far, top, self.vol_window)

        weights = np.zeros(prices_so_far.shape[1])
        w = 1.0 / len(top)
        for idx in top:
            weights[idx] = w
        return weights
