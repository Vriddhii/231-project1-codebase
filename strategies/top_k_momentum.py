import numpy as np
import pandas as pd

from backtester import Strategy


class TopKMomentum(Strategy):
    """Top-K cross-sectional momentum portfolio. Benchmark strategy 2.

    Computes the trailing return for every ticker over the lookback window,
    ranks them, and selects the top K performers. Allocates equally across
    the selected tickers. Tickers with NaN prices on the current day or
    insufficient history are excluded from ranking.

    This is the benchmark strategy defined in the project specification:
    lookback=30 days, K=10, equal weighting.

    Args:
        k: Number of top tickers to select. Default 10.
        lookback: Days used to compute trailing return. Default 30.
    """

    def __init__(self, k: int = 10, lookback: int = 30) -> None:
        self.k = k
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"Top-{self.k} Momentum ({self.lookback}d)"

    @property
    def warmup_period(self) -> int:
        return self.lookback

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        returns: dict[int, float] = {}

        for idx, col in enumerate(prices_so_far.columns):
            if np.isnan(prices_so_far[col].iloc[-1]):
                continue
            px = prices_so_far[col].dropna().values
            if len(px) < self.lookback + 1:
                continue
            today = px[-1]
            past = px[-self.lookback - 1]
            if past == 0 or np.isnan(past):
                continue
            returns[idx] = today / past - 1.0

        top_k = sorted(returns, key=returns.__getitem__, reverse=True)[: self.k]

        if not top_k:
            return np.zeros(prices_so_far.shape[1])

        weights = np.zeros(prices_so_far.shape[1])
        w = 1.0 / len(top_k)
        for idx in top_k:
            weights[idx] = w
        return weights
