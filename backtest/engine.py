"""Core backtesting loop — all-in / all-out position management."""

from __future__ import annotations

from typing import TypedDict

import polars as pl

__all__ = ["run_backtest", "BacktestResult"]


class BacktestResult(TypedDict):
    """Return type of run_backtest."""

    equity: pl.Series
    trades: pl.DataFrame


_EMPTY_TRADES_SCHEMA = {
    "entry_date": pl.Int64,
    "exit_date": pl.Int64,
    "entry_price": pl.Float64,
    "exit_price": pl.Float64,
    "pnl": pl.Float64,
}


def run_backtest(
    signals: pl.DataFrame,
    initial_capital: float = 10_000.0,
    commission_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> BacktestResult:
    """Run an all-in / all-out backtest from a signals DataFrame.

    On a BUY signal (signal=1): deploy 100% of current cash into the asset.
    On a SELL signal (signal=-1): liquidate the full position back to cash.
    Open positions at end-of-data are closed at the last available price.

    Args:
        signals: DataFrame with columns 'close' and 'signal'.
        initial_capital: Starting cash in currency units (default 10,000).
        commission_rate: Fractional commission applied to trade value (default 0.0).
        slippage_rate: Fractional slippage applied to fill price (default 0.0).

    Returns:
        BacktestResult with 'equity' (daily mark-to-market) and 'trades' (closed trade log).
    """
    # Convert to Python lists once — avoids repeated polars element access overhead
    closes = signals["close"].to_list()
    signal_list = signals["signal"].to_list()
    n = len(closes)

    equity_values: list[float] = []
    trade_records: list[dict] = []

    cash = initial_capital
    shares = 0.0
    entry_idx: int | None = None
    entry_price: float = 0.0

    for i in range(n):
        price = closes[i]
        sig = signal_list[i]

        # Mark-to-market equity at start of bar
        current_equity = cash + shares * price
        equity_values.append(current_equity)

        if sig == 1 and shares == 0.0:
            # BUY: go all-in with slippage and commission
            fill_price = price * (1.0 + slippage_rate)
            commission = cash * commission_rate
            available_cash = cash - commission
            shares = available_cash / fill_price
            cash = 0.0
            entry_idx = i
            entry_price = fill_price

        elif sig == -1 and shares > 0.0:
            # SELL: liquidate with slippage and commission
            fill_price = price * (1.0 - slippage_rate)
            gross_proceeds = shares * fill_price
            commission = gross_proceeds * commission_rate
            net_proceeds = gross_proceeds - commission
            pnl = net_proceeds - (shares * entry_price)
            trade_records.append(
                {
                    "entry_date": entry_idx,
                    "exit_date": i,
                    "entry_price": entry_price,
                    "exit_price": fill_price,
                    "pnl": pnl,
                }
            )
            cash = net_proceeds
            shares = 0.0
            entry_idx = None
            entry_price = 0.0

    # Close any open position at the last bar price
    if shares > 0.0:
        fill_price = closes[-1] * (1.0 - slippage_rate)
        gross_proceeds = shares * fill_price
        commission = gross_proceeds * commission_rate
        net_proceeds = gross_proceeds - commission
        pnl = net_proceeds - (shares * entry_price)
        trade_records.append(
            {
                "entry_date": entry_idx,
                "exit_date": n - 1,
                "entry_price": entry_price,
                "exit_price": fill_price,
                "pnl": pnl,
            }
        )

    if trade_records:
        trades = pl.DataFrame(trade_records)
    else:
        trades = pl.DataFrame(schema=_EMPTY_TRADES_SCHEMA)

    return BacktestResult(
        equity=pl.Series("equity", equity_values),
        trades=trades,
    )
