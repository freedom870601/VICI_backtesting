"""yfinance price data fetching with polars output."""

from __future__ import annotations

import logging

import polars as pl
import yfinance as yf

__all__ = ["fetch_prices"]

logger = logging.getLogger(__name__)


def fetch_prices(ticker: str, start: str, end: str) -> pl.DataFrame:
    """Fetch daily adjusted closing prices from yfinance.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        start: Start date string in 'YYYY-MM-DD' format (inclusive).
        end: End date string in 'YYYY-MM-DD' format (exclusive).

    Returns:
        Polars DataFrame with columns:
            - date (pl.Date): trading date, sorted ascending
            - close (pl.Float64): adjusted closing price, nulls dropped

    Raises:
        ValueError: If yfinance returns no data for the given ticker/range.
    """
    logger.info("Fetching %s from %s to %s", ticker, start, end)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' between {start} and {end}")

    # Flatten MultiIndex columns produced by yfinance (e.g. ('Close', 'AAPL') → 'Close')
    if isinstance(raw.columns, __import__("pandas").MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Normalize to lowercase and keep only 'close'
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.reset_index()
    raw.columns = [c.lower() for c in raw.columns]

    # Select date + close, convert to polars
    df = pl.from_pandas(raw[["date", "close"]])

    # Ensure correct dtypes, drop nulls/NaNs, sort ascending
    df = (
        df.with_columns(pl.col("close").cast(pl.Float64))
        .drop_nulls(subset=["close"])
        .sort("date")
    )

    return df
