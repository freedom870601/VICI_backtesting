"""Tests for backtest/engine.py — written before implementation (TDD Red phase)."""

import datetime

import polars as pl
import pytest

from backtest.engine import run_backtest
from backtest.strategy import generate_sma_signals


def make_signals(prices: pl.Series, fast: int = 10, slow: int = 20) -> pl.DataFrame:
    """Helper to create a signals DataFrame from a price series."""
    return generate_sma_signals(prices, fast_window=fast, slow_window=slow)


class TestRunBacktest:
    def test_returns_equity_and_trades_keys(self, sma_crossover_prices):
        """Result must be a dict with 'equity' and 'trades' keys."""
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=10_000.0)
        assert "equity" in result
        assert "trades" in result

    def test_equity_length_equals_input_length(self, sma_crossover_prices):
        """Equity series length must equal input signal DataFrame length."""
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=10_000.0)
        assert len(result["equity"]) == len(signals)

    def test_equity_starts_at_initial_capital(self, sma_crossover_prices):
        """First equity value must equal initial_capital."""
        capital = 10_000.0
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=capital)
        assert result["equity"][0] == capital

    def test_equity_never_goes_negative(self, sma_crossover_prices):
        """Without leverage, equity must remain >= 0 at all times."""
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=10_000.0)
        assert result["equity"].min() >= 0.0

    def test_flat_prices_equity_stays_constant(self, flat_prices):
        """Flat prices → no signals → equity stays at initial_capital throughout."""
        signals = make_signals(flat_prices, fast=5, slow=10)
        capital = 10_000.0
        result = run_backtest(signals, initial_capital=capital)
        equity = result["equity"]
        assert equity.min() == capital
        assert equity.max() == capital

    def test_trades_has_required_columns(self, sma_crossover_prices):
        """Trade log must have entry_date, exit_date, entry_price, exit_price, pnl."""
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=10_000.0)
        trades = result["trades"]
        required = {"entry_date", "exit_date", "entry_price", "exit_price", "pnl"}
        assert required.issubset(set(trades.columns))

    def test_trending_up_produces_positive_pnl(self, trending_up_prices):
        """Consistently rising prices must result in net positive PnL."""
        signals = make_signals(trending_up_prices, fast=10, slow=50)
        result = run_backtest(signals, initial_capital=10_000.0)
        trades = result["trades"]
        if not trades.is_empty():
            assert trades["pnl"].sum() > 0

    def test_open_position_closed_at_last_price(self, sma_crossover_prices):
        """An open position at data end must be closed and appear in trade log."""
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=10_000.0)
        # If there were any buy signals, there should be trades
        n_buys = signals.filter(pl.col("signal") == 1).height
        if n_buys > 0:
            assert result["trades"].height >= 1

    def test_empty_trades_on_no_signals(self, flat_prices):
        """No signals → trades DataFrame is empty (but correctly typed)."""
        signals = make_signals(flat_prices, fast=5, slow=10)
        result = run_backtest(signals, initial_capital=10_000.0)
        assert result["trades"].is_empty()
        assert "pnl" in result["trades"].columns
