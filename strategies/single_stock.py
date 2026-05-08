import numpy as np
import pandas as pd

from backtester import Strategy


class DualMomentum(Strategy):
    """Single-ticker dual momentum strategy.

    Requires confirmation across two timeframes before investing. Only buys
    when BOTH the short-window and long-window trailing returns are positive.
    This filters out choppy short-term noise while still following genuine
    trends, at the cost of slightly delayed entries.

    Args:
        ticker: Column name of the stock to trade. Default 'NVDA'.
        short_window: Short lookback in days. Default 20.
        long_window: Long lookback in days. Default 60.
    """

    def __init__(
        self,
        ticker: str = "NVDA",
        short_window: int = 20,
        long_window: int = 60,
    ) -> None:
        self.ticker = ticker
        self.short_window = short_window
        self.long_window = long_window

    @property
    def name(self) -> str:
        return f"Dual Momentum ({self.short_window}d/{self.long_window}d) — {self.ticker}"

    @property
    def warmup_period(self) -> int:
        return self.long_window

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        weights = np.zeros(prices_so_far.shape[1])

        if self.ticker not in prices_so_far.columns:
            return weights

        prices = prices_so_far[self.ticker].values
        today = prices[-1]
        short_past = prices[-self.short_window - 1]
        long_past = prices[-self.long_window - 1]

        if any(np.isnan(v) or v == 0 for v in [today, short_past, long_past]):
            return weights

        short_ret = today / short_past - 1.0
        long_ret = today / long_past - 1.0

        if short_ret > 0 and long_ret > 0:
            idx = prices_so_far.columns.get_loc(self.ticker)
            weights[idx] = 1.0

        return weights


class Breakout(Strategy):
    """Single-ticker 52-week channel breakout strategy.

    Enters a position when the stock closes at a new 52-week high (interpreted
    as a structural breakout above resistance). Exits when the price falls to a
    new 20-day low (interpreted as a breakdown of the near-term floor). The
    strategy re-enters on the next 52-week high signal.

    State is maintained across calls: once invested, the strategy stays
    invested until a 20-day low is breached, regardless of the breakout
    signal resetting.

    Args:
        ticker: Column name of the stock to trade. Default 'NVDA'.
        breakout_window: Days used for the high breakout signal. Default 252.
        breakdown_window: Days used for the low exit signal. Default 20.
    """

    def __init__(
        self,
        ticker: str = "NVDA",
        breakout_window: int = 252,
        breakdown_window: int = 20,
    ) -> None:
        self.ticker = ticker
        self.breakout_window = breakout_window
        self.breakdown_window = breakdown_window
        self._in_position: bool = False

    @property
    def name(self) -> str:
        return f"Breakout ({self.breakout_window}d high / {self.breakdown_window}d low) — {self.ticker}"

    @property
    def warmup_period(self) -> int:
        return self.breakout_window

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        weights = np.zeros(prices_so_far.shape[1])

        if self.ticker not in prices_so_far.columns:
            return weights

        prices = prices_so_far[self.ticker].dropna().values
        if len(prices) < self.breakout_window:
            return weights

        today = prices[-1]
        if np.isnan(today):
            return weights

        high_252 = float(np.max(prices[-self.breakout_window:]))
        low_20 = float(np.min(prices[-self.breakdown_window:]))

        if self._in_position:
            if today <= low_20:
                self._in_position = False
        else:
            if today >= high_252:
                self._in_position = True

        if self._in_position:
            idx = prices_so_far.columns.get_loc(self.ticker)
            weights[idx] = 1.0

        return weights


class TrendMeanReversionCombo(Strategy):
    """Single-ticker hybrid: trend filter with mean reversion entry timing.

    First checks whether the stock is in a confirmed uptrend (positive return
    over the long window). Only if the uptrend condition is met does it look for
    a mean reversion dip (z-score below the entry threshold) as the buy signal.
    This buys pullbacks within bull markets rather than fighting the trend.

    If no uptrend is confirmed, the strategy holds cash regardless of the
    z-score. This prevents buying dips in downtrends, which is a classic
    mean reversion failure mode.

    Args:
        ticker: Column name of the stock to trade. Default 'NVDA'.
        trend_window: Lookback for trend confirmation. Default 60.
        zscore_window: Rolling window for z-score calculation. Default 20.
        entry_threshold: Z-score below which a dip entry is triggered. Default -0.5.
    """

    def __init__(
        self,
        ticker: str = "NVDA",
        trend_window: int = 60,
        zscore_window: int = 20,
        entry_threshold: float = -0.5,
    ) -> None:
        self.ticker = ticker
        self.trend_window = trend_window
        self.zscore_window = zscore_window
        self.entry_threshold = entry_threshold

    @property
    def name(self) -> str:
        return f"Trend+MR Combo ({self.trend_window}d trend, z<{self.entry_threshold}) — {self.ticker}"

    @property
    def warmup_period(self) -> int:
        return self.trend_window

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        weights = np.zeros(prices_so_far.shape[1])

        if self.ticker not in prices_so_far.columns:
            return weights

        prices = prices_so_far[self.ticker].dropna().values
        if len(prices) < self.trend_window:
            return weights

        today = prices[-1]
        trend_past = prices[-self.trend_window - 1]

        if np.isnan(today) or np.isnan(trend_past) or trend_past == 0:
            return weights

        # Gate 1: uptrend required
        trend_ret = today / trend_past - 1.0
        if trend_ret <= 0:
            return weights

        # Gate 2: mean reversion dip entry within the trend
        window_prices = prices[-self.zscore_window:]
        mean = float(np.mean(window_prices))
        std = float(np.std(window_prices, ddof=1))

        if std == 0.0:
            return weights

        z = (today - mean) / std
        if z < self.entry_threshold:
            idx = prices_so_far.columns.get_loc(self.ticker)
            weights[idx] = 1.0

        return weights
