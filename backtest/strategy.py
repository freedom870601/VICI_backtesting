"""SMA crossover strategy signal generation."""

from __future__ import annotations

import polars as pl

__all__ = ["generate_sma_signals"]


def generate_sma_signals(
    prices: pl.Series,
    fast_window: int,
    slow_window: int,
) -> pl.DataFrame:
    """Generate event-based SMA crossover buy/sell signals.

    Only the crossover bar receives a non-zero signal:
    - signal =  1 (BUY)  when fast_sma crosses above slow_sma
    - signal = -1 (SELL) when fast_sma crosses below slow_sma
    - signal =  0        for all other bars (including warmup)

    Args:
        prices: Daily closing price series.
        fast_window: Lookback period for the fast SMA (must be < slow_window).
        slow_window: Lookback period for the slow SMA.

    Returns:
        DataFrame with columns: close, fast_sma, slow_sma, signal.

    Raises:
        ValueError: If fast_window >= slow_window.
        ValueError: If len(prices) < slow_window.
    """
    if fast_window >= slow_window:
        raise ValueError(
            f"fast_window ({fast_window}) must be less than slow_window ({slow_window})"
        )
    if len(prices) < slow_window:
        raise ValueError(
            f"Price series length ({len(prices)}) must be >= slow_window ({slow_window})"
        )

    df = pl.DataFrame({"close": prices})

    df = df.with_columns(
        pl.col("close").rolling_mean(window_size=fast_window).alias("fast_sma"),
        pl.col("close").rolling_mean(window_size=slow_window).alias("slow_sma"),
    )

    # Detect crossovers using lagged values — null comparisons fall through to 0
    df = df.with_columns(
        pl.col("fast_sma").shift(1).alias("prev_fast"),
        pl.col("slow_sma").shift(1).alias("prev_slow"),
    )

    df = df.with_columns(
        pl.when(
            (pl.col("prev_fast") < pl.col("prev_slow"))
            & (pl.col("fast_sma") >= pl.col("slow_sma"))
        )
        .then(1)
        .when(
            (pl.col("prev_fast") > pl.col("prev_slow"))
            & (pl.col("fast_sma") <= pl.col("slow_sma"))
        )
        .then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("signal")
    )

    return df.select(["close", "fast_sma", "slow_sma", "signal"])
