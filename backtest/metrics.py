"""Performance metric calculations for the backtesting system."""

from __future__ import annotations

import math

import polars as pl

__all__ = [
    "calculate_cagr",
    "calculate_annualized_volatility",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_win_rate",
]


def _validate_non_empty(series: pl.Series, name: str = "series") -> None:
    """Raise ValueError if series is empty."""
    if len(series) == 0:
        raise ValueError(f"{name} must not be empty")


def calculate_cagr(equity: pl.Series, periods_per_year: int = 252) -> float:
    """Calculate Compound Annual Growth Rate from an equity curve.

    Args:
        equity: Daily equity values (must have at least 2 elements).
        periods_per_year: Trading days per year (default 252).

    Returns:
        CAGR as a decimal (e.g. 0.10 for 10%).

    Raises:
        ValueError: If equity has fewer than 2 elements.
    """
    if len(equity) < 2:
        raise ValueError("equity must have at least 2 elements to calculate CAGR")

    start_value = equity[0]
    end_value = equity[-1]
    n_years = (len(equity) - 1) / periods_per_year

    return (end_value / start_value) ** (1.0 / n_years) - 1.0


def calculate_annualized_volatility(
    returns: pl.Series, periods_per_year: int = 252
) -> float:
    """Calculate annualized volatility from a daily return series.

    Args:
        returns: Daily return values.
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualized volatility as a decimal.

    Raises:
        ValueError: If returns is empty.
    """
    _validate_non_empty(returns, "returns")
    return returns.std() * math.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pl.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe Ratio from a daily return series.

    Args:
        returns: Daily return values.
        risk_free_rate: Annual risk-free rate (default 2%).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualized Sharpe Ratio. Returns 0.0 on zero volatility.

    Raises:
        ValueError: If returns is empty.
    """
    _validate_non_empty(returns, "returns")
    annualized_vol = calculate_annualized_volatility(returns, periods_per_year)
    if annualized_vol < 1e-10:
        return 0.0
    annualized_return = returns.mean() * periods_per_year
    return (annualized_return - risk_free_rate) / annualized_vol


def calculate_max_drawdown(equity: pl.Series) -> float:
    """Calculate Maximum Drawdown from an equity curve.

    Uses vectorized cumulative-max computation — no Python-level loops.

    Args:
        equity: Daily equity values.

    Returns:
        Maximum drawdown as a positive decimal (e.g. 0.50 for 50% drawdown).

    Raises:
        ValueError: If equity is empty.
    """
    _validate_non_empty(equity, "equity")
    rolling_max = equity.cum_max()
    drawdown = (rolling_max - equity) / rolling_max
    return drawdown.fill_nan(0.0).max()


def calculate_win_rate(trades: pl.DataFrame) -> float:
    """Calculate the fraction of trades with positive PnL.

    Args:
        trades: DataFrame with a 'pnl' column. Breakeven (pnl == 0) is not a win.

    Returns:
        Win rate as a decimal in [0.0, 1.0]. Returns 0.0 for empty DataFrames.
    """
    if trades.is_empty():
        return 0.0
    n_wins = trades.filter(pl.col("pnl") > 0).height
    return n_wins / trades.height
