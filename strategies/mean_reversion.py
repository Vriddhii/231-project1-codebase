import numpy as np
import pandas as pd

from backtester import Strategy


class MeanReversion(Strategy):
    """Single-ticker mean reversion strategy using a rolling z-score.

    Computes the z-score of today's price relative to the rolling window mean
    and standard deviation. Buys when the price is significantly below its
    recent average (oversold), sells to cash when significantly above (overbought).
    In the neutral zone (-1 <= z <= 1) the previous position is maintained to
    avoid excessive whipsawing.

    Args:
        ticker: Column name of the stock to trade. Default 'NVDA'.
        window: Rolling window length for mean and std. Default 20.
        buy_threshold: Z-score below which a buy signal is triggered. Default -1.0.
        sell_threshold: Z-score above which a sell signal is triggered. Default 1.0.
    """

    def __init__(
        self,
        ticker: str = "NVDA",
        window: int = 20,
        buy_threshold: float = -1.0,
        sell_threshold: float = 1.0,
    ) -> None:
        self.ticker = ticker
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self._position: float = 0.0  # current holding: 0 (cash) or 1 (invested)

    @property
    def name(self) -> str:
        return f"Mean Reversion ({self.window}d) — {self.ticker}"

    @property
    def warmup_period(self) -> int:
        return self.window

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        weights = np.zeros(prices_so_far.shape[1])

        if self.ticker not in prices_so_far.columns:
            return weights

        prices = prices_so_far[self.ticker].dropna().values
        if len(prices) < self.window:
            return weights

        today = prices[-1]
        if np.isnan(today):
            return weights

        window_prices = prices[-self.window:]
        mean = float(np.mean(window_prices))
        std = float(np.std(window_prices, ddof=1))

        if std == 0.0:
            return weights

        z = (today - mean) / std

        if z < self.buy_threshold:
            self._position = 1.0
        elif z > self.sell_threshold:
            self._position = 0.0
        # else: z in neutral zone — maintain self._position unchanged

        idx = prices_so_far.columns.get_loc(self.ticker)
        weights[idx] = self._position
        return weights
