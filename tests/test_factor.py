"""Tests for backtest/factor.py — TDD coverage for factor analysis module."""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest

from backtest.factor import (
    CAPMResult,
    LongShortResult,
    compute_momentum_scores,
    run_capm_regression,
    run_long_short_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices_df(prices: list[float], start: datetime.date | None = None) -> pl.DataFrame:
    """Create a minimal date+close DataFrame."""
    if start is None:
        start = datetime.date(2021, 1, 4)
    dates: list[datetime.date] = []
    d = start
    while len(dates) < len(prices):
        if d.weekday() < 5:
            dates.append(d)
        d += datetime.timedelta(days=1)
    return pl.DataFrame({
        "date": pl.Series(dates, dtype=pl.Date),
        "close": pl.Series(prices, dtype=pl.Float64),
    })


# ---------------------------------------------------------------------------
# TestComputeMomentumScores
# ---------------------------------------------------------------------------

class TestComputeMomentumScores:
    def test_output_has_date_column(self, multi_ticker_prices_dict):
        """Output DataFrame must include a 'date' column."""
        result = compute_momentum_scores(multi_ticker_prices_dict, lookback=20)
        assert "date" in result.columns

    def test_one_column_per_ticker(self, multi_ticker_prices_dict):
        """Output must have one score column per ticker plus the date column."""
        tickers = list(multi_ticker_prices_dict.keys())
        result = compute_momentum_scores(multi_ticker_prices_dict, lookback=20)
        for ticker in tickers:
            assert ticker in result.columns

    def test_warmup_rows_dropped(self, multi_ticker_prices_dict):
        """Output row count must equal total days minus lookback (warmup removed)."""
        lookback = 20
        # All tickers have the same dates in the fixture
        first_ticker = list(multi_ticker_prices_dict.keys())[0]
        total_days = len(multi_ticker_prices_dict[first_ticker])
        result = compute_momentum_scores(multi_ticker_prices_dict, lookback=lookback)
        # After inner join and drop_nulls we expect total_days - lookback rows
        assert len(result) == total_days - lookback

    def test_score_correct_arithmetic(self):
        """Score = close[d] / close[d - lookback] - 1 for known prices."""
        prices_a = [100.0, 100.0, 110.0]  # score at d=2 with lookback=1: 110/100 - 1 = 0.1
        prices_b = [50.0, 50.0, 40.0]    # score at d=2 with lookback=1: 40/50 - 1 = -0.2
        # Use 3-day data with lookback=1 → 2 scored rows
        prices_dict = {
            "A": _make_prices_df([100.0, 110.0], start=datetime.date(2021, 1, 4)),
            "B": _make_prices_df([50.0, 40.0], start=datetime.date(2021, 1, 4)),
        }
        result = compute_momentum_scores(prices_dict, lookback=1)
        assert len(result) == 1  # one non-warmup row
        a_score = result["A"][0]
        b_score = result["B"][0]
        assert abs(a_score - 0.1) < 1e-6
        assert abs(b_score - (-0.2)) < 1e-6


# ---------------------------------------------------------------------------
# TestRunCAPMRegression
# ---------------------------------------------------------------------------

class TestRunCAPMRegression:
    def _make_series(self, values: list[float], name: str = "r") -> pl.Series:
        return pl.Series(name, values)

    def test_known_beta_one(self):
        """y = 0.001 + 1.0 * x should yield beta ≈ 1.0 and R² ≈ 1.0."""
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.01, 500)
        y = 0.001 + 1.0 * x
        result = run_capm_regression(
            self._make_series(y.tolist(), "port"),
            self._make_series(x.tolist(), "bench"),
        )
        assert abs(result["beta"] - 1.0) < 1e-4
        assert result["r_squared"] > 0.999

    def test_alpha_is_annualized(self):
        """Returned alpha must equal daily_alpha * 252."""
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.01, 300)
        daily_alpha = 0.0005
        y = daily_alpha + 0.8 * x + rng.normal(0, 0.002, 300)
        result = run_capm_regression(
            self._make_series(y.tolist(), "port"),
            self._make_series(x.tolist(), "bench"),
        )
        # annualized alpha ≈ daily_alpha * 252 (rough check)
        expected_annual = daily_alpha * 252
        assert abs(result["alpha"] - expected_annual) < 0.05

    def test_zero_variance_raises(self):
        """Constant benchmark returns (zero variance) must raise ValueError."""
        port = self._make_series([0.01, 0.02, 0.03, 0.01], "port")
        bench = self._make_series([0.005, 0.005, 0.005, 0.005], "bench")
        with pytest.raises(ValueError, match="zero variance|degenerate"):
            run_capm_regression(port, bench)

    def test_too_few_obs_raises(self):
        """Fewer than 3 observations must raise ValueError."""
        port = self._make_series([0.01, 0.02], "port")
        bench = self._make_series([0.005, 0.003], "bench")
        with pytest.raises(ValueError, match="3"):
            run_capm_regression(port, bench)

    def test_result_keys_complete(self):
        """CAPMResult must contain alpha, beta, t_alpha, t_beta, r_squared."""
        rng = np.random.default_rng(2)
        x = rng.normal(0, 0.01, 200)
        y = 0.0002 + 0.9 * x + rng.normal(0, 0.003, 200)
        result = run_capm_regression(
            self._make_series(y.tolist(), "port"),
            self._make_series(x.tolist(), "bench"),
        )
        for key in ("alpha", "beta", "t_alpha", "t_beta", "r_squared"):
            assert key in result


# ---------------------------------------------------------------------------
# TestRunLongShortBacktest
# ---------------------------------------------------------------------------

class TestRunLongShortBacktest:
    def _rising_dict(self, n_tickers: int = 4, n_days: int = 120) -> dict[str, pl.DataFrame]:
        """All tickers trend upward (different slopes)."""
        rng = np.random.default_rng(seed=10)
        start = datetime.date(2021, 1, 4)
        dates: list[datetime.date] = []
        d = start
        while len(dates) < n_days:
            if d.weekday() < 5:
                dates.append(d)
            d += datetime.timedelta(days=1)
        result = {}
        for i in range(n_tickers):
            label = chr(ord("A") + i)
            prices = np.linspace(100.0, 100.0 + (i + 1) * 20, n_days).tolist()
            result[label] = pl.DataFrame({
                "date": pl.Series(dates, dtype=pl.Date),
                "close": pl.Series(prices, dtype=pl.Float64),
            })
        return result

    def _falling_dict(self, n_tickers: int = 4, n_days: int = 120) -> dict[str, pl.DataFrame]:
        """All tickers trend downward."""
        rng = np.random.default_rng(seed=20)
        start = datetime.date(2021, 1, 4)
        dates: list[datetime.date] = []
        d = start
        while len(dates) < n_days:
            if d.weekday() < 5:
                dates.append(d)
            d += datetime.timedelta(days=1)
        result = {}
        for i in range(n_tickers):
            label = chr(ord("A") + i)
            prices = np.linspace(100.0, 100.0 - (i + 1) * 20, n_days).tolist()
            result[label] = pl.DataFrame({
                "date": pl.Series(dates, dtype=pl.Date),
                "close": pl.Series(prices, dtype=pl.Float64),
            })
        return result

    def test_equity_starts_at_initial_capital(self, multi_ticker_prices_dict):
        """First equity value must equal initial_capital."""
        capital = 10_000.0
        result = run_long_short_backtest(
            multi_ticker_prices_dict, top_n=1, bottom_n=1, lookback=20,
            initial_capital=capital,
        )
        assert abs(result["equity"][0] - capital) < 1e-6

    def test_equity_length_equals_trading_days(self, multi_ticker_prices_dict):
        """Equity series length must equal the number of overlapping trading days."""
        result = run_long_short_backtest(
            multi_ticker_prices_dict, top_n=1, bottom_n=1, lookback=20,
        )
        # All tickers in fixture share the same dates (300 biz days)
        first_ticker = list(multi_ticker_prices_dict.keys())[0]
        n_days = len(multi_ticker_prices_dict[first_ticker])
        assert len(result["equity"]) == n_days

    def test_longs_gain_when_prices_rise(self):
        """If longs consistently outperform shorts, equity should grow."""
        prices = self._rising_dict()
        result = run_long_short_backtest(
            prices, top_n=1, bottom_n=1, lookback=10, initial_capital=10_000.0,
        )
        # Portfolio equity at end should exceed initial (longs rise more than shorts)
        assert result["equity"][-1] > result["equity"][0]

    def test_shorts_gain_when_prices_fall(self):
        """Short leg gains when prices fall faster than the long leg loses.

        _falling_dict slopes: A falls 20, B falls 40, C falls 60, D falls 80.
        Strategy: long A (least negative momentum), short D (most negative).
        Short gain on D (+80) dwarfs long loss on A (-20) → net positive.
        """
        prices = self._falling_dict()
        result = run_long_short_backtest(
            prices, top_n=1, bottom_n=1, lookback=10, initial_capital=10_000.0,
        )
        assert result["equity"][-1] > result["equity"][0]

    def test_commission_reduces_equity(self, multi_ticker_prices_dict):
        """With commission, final equity must be <= no-commission run."""
        result_no_cost = run_long_short_backtest(
            multi_ticker_prices_dict, top_n=1, bottom_n=1, lookback=20,
        )
        result_with_cost = run_long_short_backtest(
            multi_ticker_prices_dict, top_n=1, bottom_n=1, lookback=20,
            commission_rate=0.005,
        )
        assert result_with_cost["equity"][-1] <= result_no_cost["equity"][-1]

    def test_spread_reduces_equity(self, multi_ticker_prices_dict):
        """With spread, final equity must be <= no-spread run."""
        result_no_spread = run_long_short_backtest(
            multi_ticker_prices_dict, top_n=1, bottom_n=1, lookback=20,
        )
        result_with_spread = run_long_short_backtest(
            multi_ticker_prices_dict, top_n=1, bottom_n=1, lookback=20,
            spread_bps=20.0,
        )
        assert result_with_spread["equity"][-1] <= result_no_spread["equity"][-1]

    def test_monthly_holdings_columns(self, multi_ticker_prices_dict):
        """monthly_holdings must have rebal_date, long_tickers, short_tickers columns."""
        result = run_long_short_backtest(
            multi_ticker_prices_dict, top_n=1, bottom_n=1, lookback=20,
        )
        holdings = result["monthly_holdings"]
        assert "rebal_date" in holdings.columns
        assert "long_tickers" in holdings.columns
        assert "short_tickers" in holdings.columns

    def test_rebalances_occur_with_sufficient_data(self):
        """With >2 months of data, monthly_holdings must have at least one row (rebalances > 0)."""
        prices = self._rising_dict(n_tickers=4, n_days=120)
        result = run_long_short_backtest(
            prices, top_n=1, bottom_n=1, lookback=10, initial_capital=10_000.0,
        )
        assert len(result["monthly_holdings"]) > 0

    def test_smaller_n_weeks_more_rebalances(self):
        """Rebalancing every 1 week must produce more events than every 4 weeks."""
        prices = self._rising_dict(n_tickers=4, n_days=120)
        result_4w = run_long_short_backtest(
            prices, top_n=1, bottom_n=1, lookback=10, rebal_every_n_weeks=4,
        )
        result_1w = run_long_short_backtest(
            prices, top_n=1, bottom_n=1, lookback=10, rebal_every_n_weeks=1,
        )
        assert result_1w["monthly_holdings"].height > result_4w["monthly_holdings"].height

    def test_larger_n_weeks_fewer_rebalances(self):
        """Rebalancing every 13 weeks must produce fewer events than every 4 weeks."""
        prices = self._rising_dict(n_tickers=4, n_days=300)
        result_4w = run_long_short_backtest(
            prices, top_n=1, bottom_n=1, lookback=10, rebal_every_n_weeks=4,
        )
        result_13w = run_long_short_backtest(
            prices, top_n=1, bottom_n=1, lookback=10, rebal_every_n_weeks=13,
        )
        assert result_13w["monthly_holdings"].height < result_4w["monthly_holdings"].height

    def test_insufficient_tickers_raises(self, multi_ticker_prices_dict):
        """top_n + bottom_n > len(universe) must raise ValueError."""
        n_tickers = len(multi_ticker_prices_dict)
        with pytest.raises(ValueError, match="exceeds available tickers"):
            run_long_short_backtest(
                multi_ticker_prices_dict,
                top_n=n_tickers,
                bottom_n=1,
                lookback=20,
            )


class TestComputeMomentumScoresEdgeCases:
    def test_empty_dict_returns_empty_dataframe(self):
        """compute_momentum_scores({}) must return an empty DataFrame with 'date' column."""
        result = compute_momentum_scores({}, lookback=20)
        assert result.is_empty()
        assert "date" in result.columns


class TestLongShortBacktestEdgeCases:
    def test_zero_overlap_days_returns_empty_result(self):
        """Non-overlapping date ranges → LongShortResult with empty equity/dates/daily_returns."""
        # Ticker A: Jan 2020; Ticker B: Jan 2021 — no date overlap
        def _biz_dates(year: int, month: int, n: int) -> list[datetime.date]:
            out: list[datetime.date] = []
            d = datetime.date(year, month, 1)
            while len(out) < n:
                if d.weekday() < 5:
                    out.append(d)
                d += datetime.timedelta(days=1)
            return out
        dates_a = _biz_dates(2020, 1, 20)
        dates_b = _biz_dates(2021, 1, 20)
        prices_dict = {
            "A": pl.DataFrame({"date": pl.Series(dates_a[:20], dtype=pl.Date), "close": pl.Series([100.0] * 20)}),
            "B": pl.DataFrame({"date": pl.Series(dates_b[:20], dtype=pl.Date), "close": pl.Series([100.0] * 20)}),
        }
        result = run_long_short_backtest(prices_dict, top_n=1, bottom_n=1, lookback=5)
        assert len(result["equity"]) == 0
        assert len(result["dates"]) == 0
        assert len(result["daily_returns"]) == 0

    def test_empty_holdings_when_lookback_exceeds_data(self):
        """lookback > available rows → no rebalance fires → monthly_holdings is empty with correct schema."""
        dates = [datetime.date(2020, 1, i) for i in range(2, 12) if datetime.date(2020, 1, i).weekday() < 5]
        prices_dict = {
            "A": pl.DataFrame({"date": pl.Series(dates, dtype=pl.Date), "close": pl.Series([100.0] * len(dates))}),
            "B": pl.DataFrame({"date": pl.Series(dates, dtype=pl.Date), "close": pl.Series([100.0] * len(dates))}),
        }
        # lookback larger than available data so compute_momentum_scores returns all NaN → no scores → no rebalance
        result = run_long_short_backtest(prices_dict, top_n=1, bottom_n=1, lookback=500)
        holdings = result["monthly_holdings"]
        assert "rebal_date" in holdings.columns
        assert "long_tickers" in holdings.columns
        assert "short_tickers" in holdings.columns
        assert holdings.is_empty()
