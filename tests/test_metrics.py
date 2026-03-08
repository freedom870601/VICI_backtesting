"""Tests for backtest/metrics.py — written before implementation (TDD Red phase)."""

import datetime
import math

import polars as pl
import pytest

from backtest.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
    drawdown_series,
    holding_period_stats,
    monthly_returns,
    rolling_sharpe,
    rolling_volatility,
)


class TestCAGR:
    def test_equity_doubles_in_252_days_yields_cagr_100pct(self):
        """Equity doubles in exactly 252 trading days → CAGR = 100%."""
        import numpy as np
        # 253 data points = 252 periods = exactly 1 trading year
        equity = pl.Series("equity", np.linspace(100.0, 200.0, 253))
        result = calculate_cagr(equity, periods_per_year=252)
        assert math.isclose(result, 1.0, rel_tol=1e-6)

    def test_two_year_10pct_cagr(self):
        """Equity grows from 100 to 121 over exactly 2 years (504 periods) → CAGR = 10%."""
        # 505 points = 504 periods = 2 * 252 → CAGR = (121/100)^(1/2) - 1 = 0.10 exactly
        equity = pl.Series("equity", [100.0] + [None] * 503 + [121.0]).fill_null(strategy="forward")
        result = calculate_cagr(equity, periods_per_year=252)
        assert math.isclose(result, 0.10, rel_tol=1e-6)

    def test_single_element_raises_value_error(self):
        """Single element equity series should raise ValueError."""
        equity = pl.Series("equity", [100.0])
        with pytest.raises(ValueError, match="at least 2"):
            calculate_cagr(equity)

    def test_empty_raises_value_error(self):
        """Empty equity series should raise ValueError."""
        equity = pl.Series("equity", [], dtype=pl.Float64)
        with pytest.raises(ValueError, match="at least 2"):
            calculate_cagr(equity)

    def test_flat_equity_yields_cagr_zero(self, flat_prices):
        """Flat equity (no change) → CAGR = 0%."""
        result = calculate_cagr(flat_prices)
        assert math.isclose(result, 0.0, abs_tol=1e-9)


class TestAnnualizedVolatility:
    def test_flat_returns_yield_zero_vol(self):
        """Constant returns have zero variance → vol = 0."""
        returns = pl.Series("returns", [0.001] * 252)
        result = calculate_annualized_volatility(returns)
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_vol_formula_std_times_sqrt252(self, known_returns):
        """Annualized vol = daily std * sqrt(252)."""
        expected = known_returns.std() * math.sqrt(252)
        result = calculate_annualized_volatility(known_returns)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_empty_raises_value_error(self):
        """Empty returns raise ValueError."""
        returns = pl.Series("returns", [], dtype=pl.Float64)
        with pytest.raises(ValueError, match="empty"):
            calculate_annualized_volatility(returns)

    def test_known_series(self):
        """Known 2-element series: std([0.01, -0.01]) * sqrt(252)."""
        returns = pl.Series("returns", [0.01, -0.01])
        expected = returns.std() * math.sqrt(252)
        result = calculate_annualized_volatility(returns)
        assert math.isclose(result, expected, rel_tol=1e-9)


class TestSharpeRatio:
    def test_known_sharpe_value(self, known_returns):
        """Sharpe from seeded returns: (mean*252 - rfr) / (std*sqrt(252))."""
        rfr = 0.02
        mean_r = known_returns.mean()
        std_r = known_returns.std()
        expected = (mean_r * 252 - rfr) / (std_r * math.sqrt(252))
        result = calculate_sharpe_ratio(known_returns, risk_free_rate=rfr)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_zero_vol_returns_zero(self):
        """Zero volatility → Sharpe = 0.0 (no division by zero)."""
        returns = pl.Series("returns", [0.001] * 252)
        result = calculate_sharpe_ratio(returns)
        assert result == 0.0

    def test_empty_raises_value_error(self):
        """Empty returns raise ValueError."""
        returns = pl.Series("returns", [], dtype=pl.Float64)
        with pytest.raises(ValueError, match="empty"):
            calculate_sharpe_ratio(returns)

    def test_positive_sharpe_for_good_returns(self):
        """Returns with positive mean and non-zero variance → Sharpe > 0."""
        # alternating 2%/1% daily: mean=1.5%/day >> rfr/252, has variance
        returns = pl.Series("returns", [0.02, 0.01] * 126)
        result = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert result > 0.0


class TestMaxDrawdown:
    def test_known_drawdown_50pct(self):
        """100 → 150 → 75 → 120: MDD = (150-75)/150 = 50%."""
        equity = pl.Series("equity", [100.0, 150.0, 75.0, 120.0])
        result = calculate_max_drawdown(equity)
        assert math.isclose(result, 0.50, rel_tol=1e-6)

    def test_monotonic_rise_zero_drawdown(self):
        """Monotonically rising equity → MDD = 0."""
        equity = pl.Series("equity", [100.0, 110.0, 120.0, 130.0])
        result = calculate_max_drawdown(equity)
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_total_loss_100pct_drawdown(self):
        """Equity goes to zero → MDD = 100%."""
        equity = pl.Series("equity", [100.0, 50.0, 0.0])
        result = calculate_max_drawdown(equity)
        assert math.isclose(result, 1.0, rel_tol=1e-6)

    def test_empty_raises_value_error(self):
        """Empty equity raises ValueError."""
        equity = pl.Series("equity", [], dtype=pl.Float64)
        with pytest.raises(ValueError, match="empty"):
            calculate_max_drawdown(equity)

    def test_flat_equity_zero_drawdown(self, flat_prices):
        """Flat equity → MDD = 0."""
        result = calculate_max_drawdown(flat_prices)
        assert math.isclose(result, 0.0, abs_tol=1e-9)


class TestWinRate:
    def test_three_wins_five_trades(self):
        """3 winning / 5 total trades → win rate = 0.60."""
        trades = pl.DataFrame({"pnl": [100.0, -50.0, 200.0, -30.0, 150.0]})
        result = calculate_win_rate(trades)
        assert math.isclose(result, 0.60, rel_tol=1e-6)

    def test_all_wins(self):
        """All trades positive → win rate = 1.0."""
        trades = pl.DataFrame({"pnl": [10.0, 20.0, 30.0]})
        result = calculate_win_rate(trades)
        assert math.isclose(result, 1.0, rel_tol=1e-6)

    def test_all_losses(self):
        """All trades negative → win rate = 0.0."""
        trades = pl.DataFrame({"pnl": [-10.0, -20.0, -30.0]})
        result = calculate_win_rate(trades)
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_empty_trades_returns_zero(self):
        """Empty trade log → win rate = 0.0."""
        trades = pl.DataFrame({"pnl": pl.Series([], dtype=pl.Float64)})
        result = calculate_win_rate(trades)
        assert result == 0.0

    def test_breakeven_not_a_win(self):
        """pnl == 0 is not counted as a win."""
        trades = pl.DataFrame({"pnl": [0.0, 100.0, -50.0]})
        result = calculate_win_rate(trades)
        assert math.isclose(result, 1 / 3, rel_tol=1e-6)


class TestSortinoRatio:
    def test_sortino_positive_with_negative_returns(self):
        """Mixed returns with negatives → finite positive Sortino."""
        returns = pl.Series("returns", [0.02, -0.01, 0.03, -0.005, 0.01] * 50)
        result = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        assert isinstance(result, float)
        neg = returns.filter(returns < 0)
        downside_std = float(neg.std()) * math.sqrt(252)
        expected = (float(returns.mean()) * 252 - 0.02) / downside_std
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_sortino_zero_downside_returns_zero(self):
        """All positive returns → downside std = 0 → returns 0.0."""
        returns = pl.Series("returns", [0.01] * 100)
        result = calculate_sortino_ratio(returns)
        assert result == 0.0

    def test_sortino_empty_raises(self):
        """Empty returns raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_sortino_ratio(pl.Series("returns", [], dtype=pl.Float64))


class TestCalmarRatio:
    def test_calmar_known_values(self):
        """Equity: 100 → 200 → 150 over exactly 1 trading year (252 periods).

        CAGR = (150/100)^1 - 1 = 50%
        MDD  = (200 - 150) / 200 = 25%
        Calmar = 0.50 / 0.25 = 2.0
        """
        n = 253   # 252 periods = 1 trading year
        mid = 127  # peak at index 126
        rising  = [100.0 + 100.0 * i / (mid - 1) for i in range(mid)]   # 100 → 200
        falling = [200.0 -  50.0 * i / (mid - 1) for i in range(mid)]   # 200 → 150
        equity = rising + falling[1:]  # 253 points total
        result = calculate_calmar_ratio(equity, periods_per_year=252)
        assert math.isclose(result, 2.0, rel_tol=1e-4)

    def test_calmar_zero_drawdown(self):
        """Monotone rising equity → MDD = 0 → Calmar = 0.0."""
        equity = [100.0 + i for i in range(253)]
        result = calculate_calmar_ratio(equity)
        assert result == 0.0


class TestProfitFactor:
    def test_profit_factor_known(self):
        """Gross profit = 300, gross loss = 80 → PF = 3.75."""
        trades = pl.DataFrame({"pnl": [100.0, 200.0, -50.0, -30.0]})
        result = calculate_profit_factor(trades)
        assert math.isclose(result, 300.0 / 80.0, rel_tol=1e-6)

    def test_profit_factor_no_losers(self):
        """No losing trades → returns 0.0."""
        trades = pl.DataFrame({"pnl": [100.0, 200.0, 50.0]})
        result = calculate_profit_factor(trades)
        assert result == 0.0

    def test_profit_factor_empty(self):
        """Empty trades → returns 0.0."""
        trades = pl.DataFrame({"pnl": pl.Series([], dtype=pl.Float64)})
        result = calculate_profit_factor(trades)
        assert result == 0.0


class TestDrawdownSeries:
    def test_monotone_rising_all_zeros(self):
        """Monotone rising equity → all drawdowns are 0."""
        equity = [100.0, 110.0, 120.0, 130.0]
        result = drawdown_series(equity)
        assert all(d == 0.0 for d in result)

    def test_drawdown_known_values(self):
        """100 → 200 → 100: peak after idx 1 = 200 → dd at idx 2 = -50%."""
        equity = [100.0, 200.0, 100.0]
        result = drawdown_series(equity)
        assert math.isclose(result[0], 0.0, abs_tol=1e-9)
        assert math.isclose(result[1], 0.0, abs_tol=1e-9)
        assert math.isclose(result[2], -50.0, rel_tol=1e-6)

    def test_drawdown_empty(self):
        """Empty equity → empty list."""
        assert drawdown_series([]) == []


class TestMonthlyReturns:
    def _make_equity_and_dates(self) -> tuple[pl.Series, pl.Series]:
        """Two full months of constant 1% daily growth (Jan + Feb 2020)."""
        start = datetime.date(2020, 1, 2)
        dates: list[datetime.date] = []
        d = start
        while len(dates) < 42:  # ~21 biz days × 2 months
            if d.weekday() < 5:
                dates.append(d)
            d += datetime.timedelta(days=1)
        equity = [100.0 * (1.01 ** i) for i in range(len(dates))]
        return pl.Series("equity", equity), pl.Series("date", dates)

    def test_monthly_returns_shape(self):
        """Result has year, month, return_pct columns."""
        equity, dates = self._make_equity_and_dates()
        result = monthly_returns(equity, dates)
        assert "year" in result.columns
        assert "month" in result.columns
        assert "return_pct" in result.columns

    def test_monthly_returns_positive_growth(self):
        """With 1%/day compounding equity, monthly returns must be positive."""
        equity, dates = self._make_equity_and_dates()
        result = monthly_returns(equity, dates)
        assert result.height > 0
        assert all(r > 0 for r in result["return_pct"].to_list())

    def test_monthly_returns_empty_raises(self):
        """Empty series must raise ValueError."""
        with pytest.raises(ValueError):
            monthly_returns(
                pl.Series("equity", [], dtype=pl.Float64),
                pl.Series("date", [], dtype=pl.Date),
            )


class TestRollingSharpe:
    def test_rolling_sharpe_length_matches_input(self, known_returns):
        """Output length == len(returns)."""
        result = rolling_sharpe(known_returns, window=63)
        assert len(result) == len(known_returns)

    def test_rolling_sharpe_leading_nulls(self, known_returns):
        """First (window-1) values must be null (insufficient data)."""
        window = 63
        result = rolling_sharpe(known_returns, window=window)
        assert result[:window - 1].null_count() == window - 1

    def test_rolling_sharpe_non_null_tail(self, known_returns):
        """Values from index window onward must be non-null."""
        window = 63
        result = rolling_sharpe(known_returns, window=window)
        assert result[window:].null_count() == 0

    def test_rolling_sharpe_correct_value(self):
        """rolling_sharpe should use std(excess_returns), not std(returns)."""
        window = 5
        risk_free_rate = 0.02
        periods_per_year = 252
        daily_rf = risk_free_rate / periods_per_year
        returns = pl.Series("returns", [0.01, 0.02, 0.01, 0.02, 0.01, 0.02])
        result = rolling_sharpe(returns, window=window, risk_free_rate=risk_free_rate)

        excess = returns - daily_rf
        expected_mean = float(excess[-window:].mean())
        expected_std = float(excess[-window:].std())
        expected = (expected_mean / expected_std) * math.sqrt(periods_per_year)

        assert math.isclose(float(result[-1]), expected, rel_tol=1e-6)

    def test_rolling_sharpe_flat_returns_gives_null(self):
        """When all returns in window are identical, std≈0 → result should be null."""
        returns = pl.Series("returns", [0.001] * 10)
        result = rolling_sharpe(returns, window=5)
        assert result[-1] is None


class TestRollingVolatility:
    def test_rolling_volatility_length_matches_input(self, known_returns):
        """Output length == len(returns)."""
        result = rolling_volatility(known_returns, window=63)
        assert len(result) == len(known_returns)

    def test_rolling_volatility_non_negative(self, known_returns):
        """All non-null values must be >= 0."""
        result = rolling_volatility(known_returns, window=63)
        non_null = result.drop_nulls()
        assert (non_null >= 0).all()

    def test_rolling_volatility_flat_is_zero(self):
        """Flat returns → rolling vol = 0."""
        flat = pl.Series("returns", [0.0] * 100)
        result = rolling_volatility(flat, window=10)
        non_null = result.drop_nulls()
        assert (non_null.abs() < 1e-9).all()


class TestHoldingPeriodStats:
    def _make_trades(self) -> pl.DataFrame:
        return pl.DataFrame({
            "entry_date": [0, 5, 20],
            "exit_date": [4, 14, 30],
            "pnl": [100.0, -50.0, 200.0],
        })

    def test_holding_period_stats_keys(self):
        """Result dict must contain mean, median, min, max keys."""
        trades = self._make_trades()
        result = holding_period_stats(trades)
        assert "mean" in result
        assert "median" in result
        assert "min" in result
        assert "max" in result

    def test_holding_period_stats_values(self):
        """Known trade durations: 4, 9, 10 days."""
        trades = self._make_trades()
        result = holding_period_stats(trades)
        assert result["min"] == 4
        assert result["max"] == 10
        assert math.isclose(result["mean"], (4 + 9 + 10) / 3, rel_tol=1e-6)

    def test_holding_period_stats_empty_returns_none_values(self):
        """Empty trade log returns dict with None/NaN for all keys."""
        trades = pl.DataFrame({"entry_date": pl.Series([], dtype=pl.Int64),
                               "exit_date": pl.Series([], dtype=pl.Int64),
                               "pnl": pl.Series([], dtype=pl.Float64)})
        result = holding_period_stats(trades)
        assert result["mean"] is None or (result["mean"] != result["mean"])  # None or NaN
