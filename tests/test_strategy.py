"""Tests for backtest/strategy.py — written before implementation (TDD Red phase)."""

import polars as pl
import pytest

from backtest.strategy import generate_sma_signals


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
