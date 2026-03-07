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
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_profit_factor",
    "drawdown_series",
    "monthly_returns",
    "rolling_sharpe",
    "rolling_volatility",
    "holding_period_stats",
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


def monthly_returns(equity: pl.Series, dates: pl.Series) -> pl.DataFrame:
    """Aggregate daily equity into monthly % returns table.

    Args:
        equity: Daily equity values.
        dates: Corresponding date series (pl.Date dtype).

    Returns:
        DataFrame with columns: year (Int32), month (Int32), return_pct (Float64).

    Raises:
        ValueError: If equity or dates is empty.
    """
    _validate_non_empty(equity, "equity")
    df = pl.DataFrame({"date": dates, "equity": equity}).with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )
    monthly = (
        df.group_by(["year", "month"])
        .agg([
            pl.col("equity").first().alias("equity_start"),
            pl.col("equity").last().alias("equity_end"),
        ])
        .sort(["year", "month"])
        .with_columns(
            ((pl.col("equity_end") / pl.col("equity_start") - 1.0)).alias("return_pct")
        )
        .select(["year", "month", "return_pct"])
    )
    return monthly


def rolling_sharpe(
    returns: pl.Series,
    window: int = 63,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> pl.Series:
    """63-day rolling annualized Sharpe ratio.

    Args:
        returns: Daily return series.
        window: Rolling window size in days (default 63 ≈ 1 quarter).
        risk_free_rate: Annual risk-free rate (default 2%).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Series of same length as returns; first (window-1) values are null.
    """
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    roll_mean = excess.rolling_mean(window_size=window)
    roll_std = excess.rolling_std(window_size=window)
    # annualize
    df = pl.DataFrame({"roll_mean": roll_mean, "roll_std": roll_std})
    sharpe = df.select(
        pl.when(pl.col("roll_std").abs() > 1e-10)
        .then((pl.col("roll_mean") / pl.col("roll_std")) * math.sqrt(periods_per_year))
        .otherwise(None)
        .alias("sharpe")
    )["sharpe"]
    return sharpe


def rolling_volatility(
    returns: pl.Series,
    window: int = 63,
    periods_per_year: int = 252,
) -> pl.Series:
    """63-day rolling annualized volatility.

    Args:
        returns: Daily return series.
        window: Rolling window size in days (default 63).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Series of same length as returns; first (window-1) values are null.
    """
    roll_std = returns.rolling_std(window_size=window)
    return roll_std * math.sqrt(periods_per_year)


def holding_period_stats(trades: pl.DataFrame) -> dict:
    """Compute mean, median, min, max holding days from a trade log.

    Args:
        trades: DataFrame with 'entry_date' and 'exit_date' integer index columns.

    Returns:
        Dict with keys: mean, median, min, max. Values are None if trades is empty.
    """
    if trades.is_empty():
        return {"mean": None, "median": None, "min": None, "max": None}
    durations = (trades["exit_date"] - trades["entry_date"]).cast(pl.Float64)
    return {
        "mean": durations.mean(),
        "median": durations.median(),
        "min": int(durations.min()),
        "max": int(durations.max()),
    }


def calculate_sortino_ratio(
    returns: pl.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino Ratio using downside deviation.

    Args:
        returns: Daily return values.
        risk_free_rate: Annual risk-free rate (default 2%).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualized Sortino Ratio. Returns 0.0 if downside deviation is zero.

    Raises:
        ValueError: If returns is empty.
    """
    _validate_non_empty(returns, "returns")
    negative_returns = returns.filter(returns < 0)
    if len(negative_returns) == 0:
        downside_std = 0.0
    else:
        downside_std = float(negative_returns.std()) * math.sqrt(periods_per_year)
    if downside_std < 1e-10:
        return 0.0
    annualized_return = float(returns.mean()) * periods_per_year
    return (annualized_return - risk_free_rate) / downside_std


def calculate_calmar_ratio(
    equity: list[float],
    periods_per_year: int = 252,
) -> float:
    """Calculate Calmar Ratio: CAGR / abs(Max Drawdown).

    Args:
        equity: Daily equity values as a list.
        periods_per_year: Trading days per year (default 252).

    Returns:
        Calmar Ratio. Returns 0.0 if max drawdown is zero.
    """
    equity_s = pl.Series("equity", equity)
    cagr = calculate_cagr(equity_s, periods_per_year)
    mdd = calculate_max_drawdown(equity_s)
    if mdd < 1e-10:
        return 0.0
    return cagr / mdd


def calculate_profit_factor(trades: pl.DataFrame) -> float:
    """Calculate Profit Factor: gross profit / abs(gross loss).

    Args:
        trades: DataFrame with a 'pnl' column.

    Returns:
        Profit Factor. Returns 0.0 if there are no losing trades.
    """
    if trades.is_empty():
        return 0.0
    gross_profit = float(trades.filter(pl.col("pnl") > 0)["pnl"].sum())
    gross_loss = float(trades.filter(pl.col("pnl") < 0)["pnl"].sum())
    if abs(gross_loss) < 1e-10:
        return 0.0
    return gross_profit / abs(gross_loss)


def drawdown_series(equity: list[float]) -> list[float]:
    """Compute running drawdown percentage at each point.

    Args:
        equity: Daily equity values as a list.

    Returns:
        List of drawdown percentages (0 to -100).
    """
    if not equity:
        return []
    result: list[float] = []
    peak = equity[0]
    for v in equity:
        if v > peak:
            peak = v
        dd = (v / peak - 1.0) * 100.0 if peak > 0 else 0.0
        result.append(dd)
    return result


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
