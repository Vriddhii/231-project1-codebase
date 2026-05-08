import numpy as np
import pandas as pd


class Portfolio:
    """Tracks holdings, NAV, and transaction costs for one backtest run.

    NAV is normalized to 1.0 at the start. Each call to update() advances
    the portfolio by one trading day.

    Update sequence each day:
        1. Compute the return earned on yesterday's weights using today's prices.
        2. Grow NAV by that return.
        3. Compute turnover (sum of absolute weight changes) and deduct
           transaction costs from NAV.
        4. Store the new target weights as current weights.
        5. Record (date, nav) in history.
    """

    def __init__(self, n_assets: int) -> None:
        self.n_assets = n_assets
        self.nav: float = 1.0
        self.current_weights: np.ndarray = np.zeros(n_assets)
        self.nav_history: list[tuple] = []

    def update(
        self,
        date,
        target_weights: np.ndarray,
        prices_today: pd.Series,
        prices_yesterday: pd.Series | None,
        transaction_cost: float = 0.0,
    ) -> None:
        """Advance the portfolio by one day.

        Args:
            date: The current trading date (used as the NAV history index key).
            target_weights: Desired allocation after today's close. Length must
                equal n_assets. Values >= 0, sum <= 1.
            prices_today: Closing prices for today, indexed by ticker. May
                contain NaN for tickers not yet trading.
            prices_yesterday: Closing prices for the previous day, indexed by
                ticker. Pass None on the very first day (no return to compute).
            transaction_cost: Proportional cost applied to turnover, e.g. 0.0005
                for 5 basis points. Defaults to 0.0 (frictionless).
        """
        # Step 1 & 2: grow NAV by today's portfolio return
        if prices_yesterday is not None:
            daily_returns = prices_today / prices_yesterday - 1
            # NaN return (suspended / newly listed stock) contributes 0 return
            daily_returns = daily_returns.fillna(0.0)
            portfolio_return = float(np.dot(self.current_weights, daily_returns.values))
            self.nav *= (1.0 + portfolio_return)

        # Step 3: deduct transaction costs for trades executed at today's close
        turnover = float(np.sum(np.abs(target_weights - self.current_weights)))
        self.nav -= transaction_cost * turnover * self.nav

        # Step 4: record new weights
        self.current_weights = target_weights.copy()

        # Step 5: record NAV
        self.nav_history.append((date, self.nav))

    def get_nav_series(self) -> pd.Series:
        """Return the full NAV history as a pd.Series indexed by date."""
        if not self.nav_history:
            return pd.Series(dtype=float)
        dates, navs = zip(*self.nav_history)
        return pd.Series(navs, index=pd.DatetimeIndex(dates), name="NAV")
