"""yfinance price data fetching with polars output."""

from __future__ import annotations

import logging

import polars as pl
import yfinance as yf

__all__ = ["fetch_prices"]

logger = logging.getLogger(__name__)


def fetch_prices(ticker: str, start: str, end: str) -> pl.DataFrame:
    """Fetch daily adjusted open and closing prices from yfinance.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        start: Start date string in 'YYYY-MM-DD' format (inclusive).
        end: End date string in 'YYYY-MM-DD' format (exclusive).

    Returns:
        Polars DataFrame with columns:
            - date (pl.Date): trading date, sorted ascending
            - open (pl.Float64): adjusted opening price (null filled with close)
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

    # Normalize to lowercase and keep only 'open' and 'close'
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.reset_index()
    raw.columns = [c.lower() for c in raw.columns]

    # Select date + open + close, convert to polars
    df = pl.from_pandas(raw[["date", "open", "close"]])

    # Ensure correct dtypes, drop nulls on close, fill open nulls with close, sort ascending
    df = (
        df.with_columns(
            pl.col("open").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
        )
        .drop_nulls(subset=["close"])
        .with_columns(pl.col("open").fill_null(pl.col("close")))
        .sort("date")
    )

    return df
