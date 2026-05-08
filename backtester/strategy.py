from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Abstract base class for all daily-close trading strategies.

    Contract for implementors
    -------------------------
    - Override `name` to return a human-readable label for this strategy.
    - Override `warmup_period` to return the minimum number of historical
      rows required before the strategy can produce a meaningful signal.
      The engine holds cash for this many days before calling get_weights.
    - Implement `get_weights(prices_so_far)` to produce a target allocation.

    Contract for get_weights
    ------------------------
    Input:
        prices_so_far : pd.DataFrame, shape (t, n_tickers)
            All closing prices from day 0 up to and including the current day t.
            Columns are ticker symbols. Some entries may be NaN for tickers that
            were not yet trading on certain days.

    Output:
        np.ndarray of length n_tickers (same order as prices_so_far.columns)
            - Each element is the target fraction of total portfolio value to
              allocate to that ticker.
            - All values must be >= 0  (no short selling).
            - Values must sum to <= 1  (no leverage; remainder is held as cash).
            - Any ticker whose price is NaN on the current day must receive
              weight 0 — it cannot be traded.
            - The engine does NOT validate these constraints; it is the
              strategy's responsibility to satisfy them.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this strategy, used in logs and plot titles."""

    @property
    def warmup_period(self) -> int:
        """Minimum rows of history required before get_weights is called.

        The engine holds the portfolio in cash for this many days. Strategies
        that use a lookback window of N days should return N here so that
        the first call to get_weights always has a full window available.
        Defaults to 0 (no warmup required).
        """
        return 0

    @abstractmethod
    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        """Compute target portfolio weights given all available price history.

        Args:
            prices_so_far: DataFrame of shape (t, n_tickers) containing closing
                prices from the start of the dataset up to and including today.
                The last row (prices_so_far.iloc[-1]) is today's closing price.

        Returns:
            np.ndarray of length n_tickers with target allocation weights.
            All values >= 0, sum <= 1, weight = 0 for any NaN-priced ticker.
        """

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
