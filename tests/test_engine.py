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


class TestRunBacktestWithCosts:
    def test_zero_costs_matches_no_cost_result(self, sma_crossover_prices):
        """commission_rate=0, slippage_rate=0 must produce identical results to default."""
        signals = make_signals(sma_crossover_prices)
        result_default = run_backtest(signals, initial_capital=10_000.0)
        result_zero = run_backtest(signals, initial_capital=10_000.0, commission_rate=0.0, slippage_rate=0.0)
        assert result_default["equity"].to_list() == result_zero["equity"].to_list()

    def test_commission_reduces_final_equity(self, sma_crossover_prices):
        """Positive commission_rate must reduce final equity vs no-cost run."""
        signals = make_signals(sma_crossover_prices)
        result_no_cost = run_backtest(signals, initial_capital=10_000.0)
        result_with_cost = run_backtest(signals, initial_capital=10_000.0, commission_rate=0.01)
        # Only meaningful if there were trades
        if not result_with_cost["trades"].is_empty():
            assert result_with_cost["equity"][-1] <= result_no_cost["equity"][-1]

    def test_slippage_reduces_final_equity(self, sma_crossover_prices):
        """Positive slippage_rate must reduce final equity vs no-cost run."""
        signals = make_signals(sma_crossover_prices)
        result_no_cost = run_backtest(signals, initial_capital=10_000.0)
        result_with_slip = run_backtest(signals, initial_capital=10_000.0, slippage_rate=0.005)
        if not result_with_slip["trades"].is_empty():
            assert result_with_slip["equity"][-1] <= result_no_cost["equity"][-1]

    def test_combined_costs_reduce_equity_further(self, sma_crossover_prices):
        """Combined commission + slippage must reduce equity more than either alone."""
        signals = make_signals(sma_crossover_prices)
        result_commission = run_backtest(signals, initial_capital=10_000.0, commission_rate=0.01)
        result_both = run_backtest(signals, initial_capital=10_000.0, commission_rate=0.01, slippage_rate=0.005)
        if not result_both["trades"].is_empty():
            assert result_both["equity"][-1] <= result_commission["equity"][-1]

    def test_commission_exact_arithmetic(self):
        """Verify commission arithmetic: buy 100 shares @$100, sell @$110, 1% commission."""
        # Craft a signals DataFrame: buy at bar 0, sell at bar 1
        signals = pl.DataFrame({
            "close": [100.0, 110.0],
            "signal": pl.Series([1, -1], dtype=pl.Int8),
        })
        result = run_backtest(signals, initial_capital=10_000.0, commission_rate=0.01)
        trades = result["trades"]
        assert not trades.is_empty()
        # commission_on_buy = 10000 * 0.01 = 100 → available 9900
        # shares = 9900 / 100 = 99
        # gross_proceeds = 99 * 110 = 10890
        # commission_on_sell = 10890 * 0.01 = 108.90
        # net_proceeds = 10890 - 108.90 = 10781.10
        # pnl = 10781.10 - (99 * 100) = 10781.10 - 9900 = 881.10
        expected_pnl = 881.10
        assert abs(trades["pnl"][0] - expected_pnl) < 0.01

    def test_slippage_adjusts_fill_prices_in_trade_log(self):
        """Slippage should cause entry_price > raw close (buy) and exit_price < raw close (sell)."""
        signals = pl.DataFrame({
            "close": [100.0, 110.0],
            "signal": pl.Series([1, -1], dtype=pl.Int8),
        })
        slippage = 0.01
        result = run_backtest(signals, initial_capital=10_000.0, slippage_rate=slippage)
        trades = result["trades"]
        assert not trades.is_empty()
        # entry fill = 100 * 1.01 = 101.0
        assert abs(trades["entry_price"][0] - 101.0) < 1e-9
        # exit fill = 110 * 0.99 = 108.90
        assert abs(trades["exit_price"][0] - 108.90) < 1e-9

    def test_equity_never_negative_with_extreme_costs(self, sma_crossover_prices):
        """Even with very high costs, equity must never go below 0."""
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=10_000.0, commission_rate=0.5, slippage_rate=0.5)
        assert result["equity"].min() >= 0.0

    def test_no_cost_trade_log_schema_unchanged(self, sma_crossover_prices):
        """Trade log schema must include the same columns regardless of cost params."""
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=10_000.0, commission_rate=0.01, slippage_rate=0.005)
        required = {"entry_date", "exit_date", "entry_price", "exit_price", "pnl"}
        assert required.issubset(set(result["trades"].columns))


class TestRunBacktestSpreadAndEntryType:
    """Tests for spread_bps and entry_price_type parameters."""

    def _two_bar_signals(self, buy_close: float = 100.0, sell_close: float = 110.0) -> pl.DataFrame:
        """Minimal signals: BUY at bar 0, SELL at bar 1."""
        return pl.DataFrame({
            "close": [buy_close, sell_close],
            "signal": pl.Series([1, -1], dtype=pl.Int8),
        })

    def test_spread_zero_matches_default(self, sma_crossover_prices):
        """spread_bps=0 must produce identical equity to the no-spread default."""
        signals = make_signals(sma_crossover_prices)
        result_default = run_backtest(signals, initial_capital=10_000.0)
        result_zero_spread = run_backtest(signals, initial_capital=10_000.0, spread_bps=0.0)
        assert result_default["equity"].to_list() == result_zero_spread["equity"].to_list()

    def test_buy_fill_includes_spread(self):
        """BUY fill = price * (1 + slippage) * (1 + spread_bps/20000)."""
        signals = self._two_bar_signals(100.0, 200.0)
        spread_bps = 10.0
        slippage = 0.005
        result = run_backtest(
            signals, initial_capital=10_000.0,
            slippage_rate=slippage, spread_bps=spread_bps,
        )
        trades = result["trades"]
        assert not trades.is_empty()
        expected_fill = 100.0 * (1.0 + slippage) * (1.0 + spread_bps / 20_000.0)
        assert abs(trades["entry_price"][0] - expected_fill) < 1e-6

    def test_sell_fill_includes_spread(self):
        """SELL fill = price * (1 - slippage) * (1 - spread_bps/20000)."""
        signals = self._two_bar_signals(100.0, 110.0)
        spread_bps = 10.0
        slippage = 0.005
        result = run_backtest(
            signals, initial_capital=10_000.0,
            slippage_rate=slippage, spread_bps=spread_bps,
        )
        trades = result["trades"]
        assert not trades.is_empty()
        expected_fill = 110.0 * (1.0 - slippage) * (1.0 - spread_bps / 20_000.0)
        assert abs(trades["exit_price"][0] - expected_fill) < 1e-6

    def test_spread_reduces_final_equity(self, sma_crossover_prices):
        """Positive spread_bps must reduce final equity vs zero-spread run."""
        signals = make_signals(sma_crossover_prices)
        result_no_spread = run_backtest(signals, initial_capital=10_000.0)
        result_with_spread = run_backtest(signals, initial_capital=10_000.0, spread_bps=20.0)
        if not result_with_spread["trades"].is_empty():
            assert result_with_spread["equity"][-1] <= result_no_spread["equity"][-1]

    def test_entry_type_open_uses_open_price(self):
        """When entry_price_type='open', BUY fill must use next bar's open price."""
        # signal fires at i=0 (close=100), fills at opens[1]=110 (next bar open)
        signals = pl.DataFrame({
            "open": [95.0, 110.0, 105.0],
            "close": [100.0, 110.0, 120.0],
            "signal": pl.Series([1, 0, -1], dtype=pl.Int8),
        })
        result = run_backtest(signals, initial_capital=10_000.0, entry_price_type="open")
        trades = result["trades"]
        assert not trades.is_empty()
        # fill should be opens[1]=110.0, not closes[0]=100.0 or opens[0]=95.0
        assert abs(trades["entry_price"][0] - 110.0) < 1e-6

    def test_entry_type_close_unchanged(self):
        """entry_price_type='close' (default) must use close price for BUY fill."""
        open_price = 95.0
        close_price = 100.0
        signals = pl.DataFrame({
            "open": [open_price, 110.0],
            "close": [close_price, 110.0],
            "signal": pl.Series([1, -1], dtype=pl.Int8),
        })
        result = run_backtest(signals, initial_capital=10_000.0, entry_price_type="close")
        trades = result["trades"]
        assert not trades.is_empty()
        # fill should be close_price
        assert abs(trades["entry_price"][0] - close_price) < 1e-6

    def test_trade_has_entry_type_column(self, sma_crossover_prices):
        """Trade log must contain 'entry_type' column."""
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=10_000.0, entry_price_type="close")
        trades = result["trades"]
        # Column should be present even in empty trades (schema check)
        assert "entry_type" in trades.columns

    def test_entry_type_value_stored_in_trade(self, sma_crossover_prices):
        """Each trade record's entry_type matches the parameter passed."""
        signals = make_signals(sma_crossover_prices)
        result = run_backtest(signals, initial_capital=10_000.0, entry_price_type="open")
        trades = result["trades"]
        if not trades.is_empty():
            assert all(v == "open" for v in trades["entry_type"].to_list())


class TestForceClose:
    """Tests covering the force-close-at-end-of-data branch (engine.py:114-129)."""

    def test_force_close_when_no_sell_signal(self):
        """BUY at bar 0, no SELL ever → force-closed trade appears in log at last bar."""
        signals = pl.DataFrame({
            "open": [100.0, 105.0, 110.0],
            "close": [100.0, 105.0, 110.0],
            "signal": pl.Series([1, 0, 0], dtype=pl.Int8),
        })
        result = run_backtest(signals, initial_capital=10_000.0)
        trades = result["trades"]
        assert not trades.is_empty()
        # Exit index should be the last bar (index 2)
        assert trades["exit_date"][0] == 2

    def test_force_close_pnl_computed_correctly(self):
        """Force-closed trade PnL reflects slippage and commission at last bar price."""
        buy_price = 100.0
        last_price = 120.0
        commission = 0.001
        slippage = 0.002
        signals = pl.DataFrame({
            "open": [buy_price, 110.0, last_price],
            "close": [buy_price, 110.0, last_price],
            "signal": pl.Series([1, 0, 0], dtype=pl.Int8),
        })
        result = run_backtest(
            signals,
            initial_capital=10_000.0,
            commission_rate=commission,
            slippage_rate=slippage,
        )
        trades = result["trades"]
        assert not trades.is_empty()
        # exit_price should include slippage
        expected_exit = last_price * (1.0 - slippage)
        assert abs(trades["exit_price"][0] - expected_exit) < 1e-6
        # PnL should be positive (price rose)
        assert trades["pnl"][0] > 0
