import pandas as pd
from pathlib import Path


def load_prices(csv_path: str | Path) -> pd.DataFrame:
    """Load the Nasdaq-100 daily price CSV and return a wide-format close price DataFrame.

    Reads the long-format CSV (one row per ticker per day), pivots to wide format,
    and returns a DataFrame of shape (trading_days, n_tickers) indexed by date.

    Args:
        csv_path: Path to nasdaq100_daily_5y.csv.

    Returns:
        DataFrame with DatetimeIndex (ascending), one column per ticker, values are
        daily closing prices. Any ticker with all-NaN closes is dropped.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
        ValueError: If required columns are missing from the CSV.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Download nasdaq100_daily_5y.csv and place it in the data/ directory."
        )

    raw = pd.read_csv(path)

    required = {"ticker", "date", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    prices = (
        raw[["ticker", "date", "close"]]
        .pivot(index="date", columns="ticker", values="close")
    )

    prices.index = pd.to_datetime(prices.index)
    prices.columns.name = None
    prices.sort_index(inplace=True)

    # Drop tickers that have no close data at all
    prices.dropna(axis=1, how="all", inplace=True)

    return prices
