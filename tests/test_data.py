"""Tests for backtest/data.py — written before implementation (TDD Red phase).

All yfinance calls are monkeypatched to avoid network requests.
"""

import datetime

import pandas as pd
import polars as pl
import pytest

from backtest.data import fetch_prices


def _make_yf_response(
    dates: list[datetime.date],
    closes: list[float],
) -> pd.DataFrame:
    """Build a minimal yfinance-style pandas DataFrame response."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    df = pd.DataFrame({"Close": closes}, index=idx)
    df.index.name = "Date"
    return df


class TestFetchPrices:
    def test_returns_polars_dataframe(self, monkeypatch):
        """Result must be a polars DataFrame, not pandas."""
        dates = [datetime.date(2023, 1, i) for i in range(1, 6)]
        mock_df = _make_yf_response(dates, [100.0, 101.0, 102.0, 103.0, 104.0])

        monkeypatch.setattr("yfinance.download", lambda *a, **kw: mock_df)
        result = fetch_prices("AAPL", "2023-01-01", "2023-01-05")
        assert isinstance(result, pl.DataFrame)

    def test_has_date_and_close_columns(self, monkeypatch):
        """Output must have 'date' and 'close' columns."""
        dates = [datetime.date(2023, 1, i) for i in range(1, 6)]
        mock_df = _make_yf_response(dates, [100.0, 101.0, 102.0, 103.0, 104.0])

        monkeypatch.setattr("yfinance.download", lambda *a, **kw: mock_df)
        result = fetch_prices("AAPL", "2023-01-01", "2023-01-05")
        assert "date" in result.columns
        assert "close" in result.columns

    def test_close_column_is_float64(self, monkeypatch):
        """The 'close' column must be pl.Float64 dtype."""
        dates = [datetime.date(2023, 1, i) for i in range(1, 6)]
        mock_df = _make_yf_response(dates, [100.0, 101.0, 102.0, 103.0, 104.0])

        monkeypatch.setattr("yfinance.download", lambda *a, **kw: mock_df)
        result = fetch_prices("AAPL", "2023-01-01", "2023-01-05")
        assert result["close"].dtype == pl.Float64

    def test_output_sorted_ascending_by_date(self, monkeypatch):
        """Even if yfinance returns data reversed, output must be sorted ascending."""
        dates = [datetime.date(2023, 1, i) for i in range(5, 0, -1)]  # reversed
        mock_df = _make_yf_response(dates, [104.0, 103.0, 102.0, 101.0, 100.0])

        monkeypatch.setattr("yfinance.download", lambda *a, **kw: mock_df)
        result = fetch_prices("AAPL", "2023-01-01", "2023-01-05")
        dates_out = result["date"].to_list()
        assert dates_out == sorted(dates_out)

    def test_empty_response_raises_value_error(self, monkeypatch):
        """Empty yfinance response must raise ValueError."""
        empty_df = pd.DataFrame({"Close": []})
        monkeypatch.setattr("yfinance.download", lambda *a, **kw: empty_df)
        with pytest.raises(ValueError, match="No data returned"):
            fetch_prices("INVALID", "2023-01-01", "2023-01-05")

    def test_nan_close_values_are_dropped(self, monkeypatch):
        """NaN values in close column must be dropped (no nulls in output)."""
        import math

        dates = [datetime.date(2023, 1, i) for i in range(1, 6)]
        mock_df = _make_yf_response(dates, [100.0, float("nan"), 102.0, float("nan"), 104.0])

        monkeypatch.setattr("yfinance.download", lambda *a, **kw: mock_df)
        result = fetch_prices("AAPL", "2023-01-01", "2023-01-05")
        assert result["close"].null_count() == 0
        assert len(result) == 3  # only 3 non-NaN rows
