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


class SMAPortfolio(Strategy):
    """Cross-sectional SMA crossover portfolio across the full Nasdaq-100 universe.

    Selects tickers where the short simple moving average is above the long
    simple moving average — a signal that short-term price momentum is positive
    relative to the medium-term trend. Allocates to the selected set using
    either equal weighting or inverse-volatility weighting.

    If no tickers meet the crossover criterion, the portfolio holds cash.

    Args:
        short_window: Short MA window in days. Default 20.
        long_window: Long MA window in days. Default 50.
        weighting: 'equal' or 'inverse_vol'. Default 'equal'.
        vol_window: Days of returns for volatility estimate when
            weighting='inverse_vol'. Default 20.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        weighting: str = "equal",
        vol_window: int = 20,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.weighting = weighting
        self.vol_window = vol_window

    @property
    def name(self) -> str:
        w_label = "Equal Wt" if self.weighting == "equal" else "Inv-Vol Wt"
        return f"SMA Crossover ({self.short_window}/{self.long_window}d) — {w_label}"

    @property
    def warmup_period(self) -> int:
        return self.long_window

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        selected: list[int] = []

        for idx, col in enumerate(prices_so_far.columns):
            # Skip if price is NaN today
            if np.isnan(prices_so_far[col].iloc[-1]):
                continue
            px = prices_so_far[col].dropna().values
            if len(px) < self.long_window:
                continue
            sma_short = float(np.mean(px[-self.short_window:]))
            sma_long = float(np.mean(px[-self.long_window:]))
            if sma_short > sma_long:
                selected.append(idx)

        if not selected:
            return np.zeros(prices_so_far.shape[1])

        if self.weighting == "inverse_vol":
            return _inv_vol_weights(prices_so_far, selected, self.vol_window)

        weights = np.zeros(prices_so_far.shape[1])
        w = 1.0 / len(selected)
        for idx in selected:
            weights[idx] = w
        return weights
