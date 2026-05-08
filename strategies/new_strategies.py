import numpy as np
import pandas as pd

from backtester import Strategy


def _inv_vol_weights(prices_so_far: pd.DataFrame, selected: list[int], vol_window: int = 20) -> np.ndarray:
    """Return inverse-volatility normalized weights for the given column indices.

    Falls back to equal weighting for any ticker whose volatility cannot be
    computed or is zero, ensuring the portfolio is never left empty.
    """
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


class RiskAdjustedMomentum(Strategy):
    """New Strategy 1: Risk-adjusted momentum — rank by return-per-unit-risk.

    Scores every ticker by a Sharpe-like signal: trailing return divided by
    rolling volatility over the same window. This directly rewards stocks that
    moved up smoothly (high return, low vol) over those with noisy, volatile
    gains. The top K tickers by this score are held in an equal-weight portfolio.

    Rationale: pure momentum rankings (Benchmark 2) favour raw returnees
    regardless of risk. Ranking by return/vol retains the momentum character
    but selects the stocks whose gains were most consistent — improving the
    Sharpe of the resulting portfolio without sacrificing return as aggressively
    as a vol-filter approach.

    Addresses both benchmark weaknesses:
    - Benchmark 1 (SMA): replaces a slow lagging signal with a responsive
      risk-adjusted ranking that reacts to price moves within days.
    - Benchmark 2 (Top-K momentum): improves selection quality so the
      chosen stocks have higher return-per-unit-risk than a raw return sort.

    Args:
        lookback: Days used for trailing return in the score numerator. Default 30.
        vol_window: Days of daily returns used for the score denominator. Default 20.
        k: Number of top tickers to hold. Default 10.
    """

    def __init__(
        self,
        lookback: int = 30,
        vol_window: int = 20,
        k: int = 10,
    ) -> None:
        self.lookback = lookback
        self.vol_window = vol_window
        self.k = k

    @property
    def name(self) -> str:
        return "Risk-Adjusted Momentum (Top 10)"

    @property
    def warmup_period(self) -> int:
        return max(self.lookback, self.vol_window)

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        scores: dict[int, float] = {}

        for idx, col in enumerate(prices_so_far.columns):
            if np.isnan(prices_so_far[col].iloc[-1]):
                continue
            px = prices_so_far[col].dropna().values
            if len(px) < max(self.lookback, self.vol_window) + 1:
                continue
            today = px[-1]
            past = px[-self.lookback - 1]
            if past == 0 or np.isnan(past):
                continue
            ret = today / past - 1.0
            daily_rets = np.diff(px[-(self.vol_window + 1):]) / px[-(self.vol_window + 1):-1]
            vol = float(np.std(daily_rets, ddof=1))
            if vol == 0:
                continue
            scores[idx] = ret / vol

        top = sorted(scores, key=scores.__getitem__, reverse=True)[: self.k]

        if not top:
            return np.zeros(prices_so_far.shape[1])

        weights = np.zeros(prices_so_far.shape[1])
        w = 1.0 / len(top)
        for idx in top:
            weights[idx] = w
        return weights


class DualMomentumPullback(Strategy):
    """New Strategy 2: Dual momentum with pullback entry and inverse-vol sizing.

    Pipeline:
        1. Dual momentum filter: keep only tickers with positive return on BOTH
           the short and long lookback windows. This confirms a genuine sustained
           uptrend across two timeframes, filtering out short-term noise.
        2. Pullback entry: from the dual-momentum set, prefer tickers that have
           experienced a slight pullback over the past pullback_window days
           (negative return). Buying a dip within a confirmed trend improves
           average entry price and reduces the chance of buying at a local peak.
        3. Fallback: if no tickers show a recent pullback, invest in the full
           dual-momentum set with inverse-vol weights. Never hold cash just
           because no pullbacks exist — missing a bull run is also a cost.
        4. Weight by inverse volatility.

    Rationale: dual momentum (short=20d, long=40d) reduces the false signals
    that a single 30-day window generates in choppy periods, directly addressing
    Benchmark 2's noise problem. The pullback layer buys at a discount within
    established trends. Combined with inverse-vol weighting, three independent
    layers of risk control are applied.

    Args:
        short_window: Short lookback for dual momentum. Default 20.
        long_window: Long lookback for dual momentum. Default 40.
        pullback_window: Days used to detect a pullback entry. Default 3.
        vol_window: Days of returns used for rolling volatility. Default 20.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 40,
        pullback_window: int = 3,
        vol_window: int = 20,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.pullback_window = pullback_window
        self.vol_window = vol_window

    @property
    def name(self) -> str:
        return f"Dual Momentum + Pullback ({self.short_window}/{self.long_window}d, {self.pullback_window}d)"

    @property
    def warmup_period(self) -> int:
        return self.long_window

    def get_weights(self, prices_so_far: pd.DataFrame) -> np.ndarray:
        dual_selected: list[int] = []

        for idx, col in enumerate(prices_so_far.columns):
            if np.isnan(prices_so_far[col].iloc[-1]):
                continue
            px = prices_so_far[col].dropna().values
            if len(px) < self.long_window + 1:
                continue
            today = px[-1]
            short_past = px[-self.short_window - 1]
            long_past = px[-self.long_window - 1]
            if short_past == 0 or long_past == 0 or np.isnan(short_past) or np.isnan(long_past):
                continue
            if (today / short_past - 1.0) > 0 and (today / long_past - 1.0) > 0:
                dual_selected.append(idx)

        if not dual_selected:
            return np.zeros(prices_so_far.shape[1])

        # Pullback filter: prefer tickers with a negative return over pullback_window
        pullback_selected: list[int] = []
        for idx in dual_selected:
            px = prices_so_far.iloc[:, idx].dropna().values
            if len(px) < self.pullback_window + 1:
                continue
            if px[-1] / px[-self.pullback_window - 1] - 1.0 < 0:
                pullback_selected.append(idx)

        # Fallback: no pullbacks → use full dual-momentum set
        final = pullback_selected if pullback_selected else dual_selected

        return _inv_vol_weights(prices_so_far, final, self.vol_window)
