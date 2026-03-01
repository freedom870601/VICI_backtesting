"""Tests for backtest/strategy.py — written before implementation (TDD Red phase)."""

import polars as pl
import pytest

from backtest.strategy import generate_momentum_signals, generate_sma_signals


class TestGenerateSmaSignals:
    def test_output_length_equals_input_length(self, sma_crossover_prices):
        """Output DataFrame row count must equal input price Series length."""
        result = generate_sma_signals(sma_crossover_prices, fast_window=10, slow_window=20)
        assert len(result) == len(sma_crossover_prices)

    def test_signal_column_values_are_valid(self, sma_crossover_prices):
        """Signal column values must only be -1, 0, or 1."""
        result = generate_sma_signals(sma_crossover_prices, fast_window=10, slow_window=20)
        valid = {-1, 0, 1}
        unique_signals = set(result["signal"].to_list())
        assert unique_signals.issubset(valid)

    def test_at_least_one_buy_signal(self, sma_crossover_prices):
        """Engineered price series must produce at least one BUY (signal=1)."""
        result = generate_sma_signals(sma_crossover_prices, fast_window=10, slow_window=20)
        assert result.filter(pl.col("signal") == 1).height >= 1

    def test_at_least_one_sell_signal(self, sma_crossover_prices):
        """Engineered price series must produce at least one SELL (signal=-1)."""
        result = generate_sma_signals(sma_crossover_prices, fast_window=10, slow_window=20)
        assert result.filter(pl.col("signal") == -1).height >= 1

    def test_warmup_rows_have_zero_signal(self, sma_crossover_prices):
        """First (slow_window - 1) rows must have signal == 0 (warmup period)."""
        slow_window = 20
        result = generate_sma_signals(
            sma_crossover_prices, fast_window=10, slow_window=slow_window
        )
        warmup = result["signal"].head(slow_window - 1).to_list()
        assert all(s == 0 for s in warmup)

    def test_flat_prices_produce_no_buy_sell(self, flat_prices):
        """Constant prices → SMAs are equal → no crossover → no buy or sell."""
        result = generate_sma_signals(flat_prices, fast_window=5, slow_window=10)
        assert result.filter(pl.col("signal") != 0).height == 0

    def test_fast_window_gte_slow_raises_value_error(self, sma_crossover_prices):
        """fast_window >= slow_window must raise ValueError."""
        with pytest.raises(ValueError, match="fast_window"):
            generate_sma_signals(sma_crossover_prices, fast_window=20, slow_window=20)

        with pytest.raises(ValueError, match="fast_window"):
            generate_sma_signals(sma_crossover_prices, fast_window=30, slow_window=20)

    def test_fewer_rows_than_slow_window_raises_value_error(self):
        """Price series shorter than slow_window must raise ValueError."""
        short_prices = pl.Series("close", [100.0] * 5)
        with pytest.raises(ValueError, match="slow_window"):
            generate_sma_signals(short_prices, fast_window=3, slow_window=10)

    def test_output_has_sma_columns_for_visualization(self, sma_crossover_prices):
        """Output DataFrame must include fast_sma and slow_sma columns."""
        result = generate_sma_signals(sma_crossover_prices, fast_window=10, slow_window=20)
        assert "fast_sma" in result.columns
        assert "slow_sma" in result.columns

    def test_signal_column_exists(self, sma_crossover_prices):
        """Output DataFrame must have a 'signal' column."""
        result = generate_sma_signals(sma_crossover_prices, fast_window=10, slow_window=20)
        assert "signal" in result.columns


class TestGenerateMomentumSignals:
    def test_output_length_equals_input_length(self, momentum_prices):
        """Output DataFrame row count must equal input price Series length."""
        result = generate_momentum_signals(momentum_prices, lookback=252)
        assert len(result) == len(momentum_prices)

    def test_warmup_rows_have_zero_signal(self, momentum_prices):
        """First `lookback` rows must have signal == 0 (warmup period)."""
        lookback = 252
        result = generate_momentum_signals(momentum_prices, lookback=lookback)
        warmup = result["signal"].head(lookback).to_list()
        assert all(s == 0 for s in warmup)

    def test_signal_one_when_price_above_reference(self, momentum_prices):
        """signal == 1 when close[i] > close[i - lookback]."""
        lookback = 100
        result = generate_momentum_signals(momentum_prices, lookback=lookback)
        # In rising phase (indices 200+), close > reference → signal should be 1
        after_warmup = result.slice(200, 200)
        assert after_warmup.filter(pl.col("signal") == 1).height > 0

    def test_signal_minus_one_when_price_below_or_equal_reference(self, momentum_prices):
        """signal == -1 when close[i] <= close[i - lookback]."""
        lookback = 100
        result = generate_momentum_signals(momentum_prices, lookback=lookback)
        # In flat phase after warmup (indices 100-199), close == reference → signal -1
        flat_post_warmup = result.slice(100, 100)
        assert flat_post_warmup.filter(pl.col("signal") == -1).height > 0

    def test_output_has_required_columns(self, momentum_prices):
        """Output DataFrame must include 'close', 'momentum_ref', and 'signal' columns."""
        result = generate_momentum_signals(momentum_prices, lookback=252)
        assert "close" in result.columns
        assert "momentum_ref" in result.columns
        assert "signal" in result.columns

    def test_momentum_ref_is_null_during_warmup(self, momentum_prices):
        """momentum_ref must be null for the first `lookback` rows."""
        lookback = 252
        result = generate_momentum_signals(momentum_prices, lookback=lookback)
        null_count = result["momentum_ref"].head(lookback).null_count()
        assert null_count == lookback

    def test_insufficient_data_raises_value_error(self):
        """Price series shorter than lookback must raise ValueError."""
        short = pl.Series("close", [100.0] * 10)
        with pytest.raises(ValueError, match="lookback"):
            generate_momentum_signals(short, lookback=50)

    def test_signal_values_only_in_valid_set(self, momentum_prices):
        """Signal column values must only be -1, 0, or 1."""
        result = generate_momentum_signals(momentum_prices, lookback=100)
        unique = set(result["signal"].to_list())
        assert unique.issubset({-1, 0, 1})

    def test_trending_up_produces_long_signals_after_warmup(self, momentum_prices):
        """Rising price phase should produce signal == 1 after warmup."""
        lookback = 100
        result = generate_momentum_signals(momentum_prices, lookback=lookback)
        # Rising section starts at index 200; after warmup (100), check indices 200-399
        after_warmup = result.slice(200, 200)
        assert after_warmup.filter(pl.col("signal") == 1).height > 0
