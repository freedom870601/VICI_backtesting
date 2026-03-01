"""Factor analysis module: momentum scores, CAPM regression, long-short backtesting."""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import polars as pl

__all__ = [
    "compute_momentum_scores",
    "run_capm_regression",
    "run_long_short_backtest",
    "CAPMResult",
    "LongShortResult",
]

logger = logging.getLogger(__name__)


class CAPMResult(TypedDict):
    """Return type of run_capm_regression."""

    alpha: float        # annualized alpha (daily_alpha * 252)
    beta: float
    t_alpha: float
    t_beta: float
    r_squared: float


class LongShortResult(TypedDict):
    """Return type of run_long_short_backtest."""

    equity: pl.Series
    dates: pl.Series
    daily_returns: pl.Series
    monthly_holdings: pl.DataFrame


def compute_momentum_scores(
    prices_dict: dict[str, pl.DataFrame],
    lookback: int,
) -> pl.DataFrame:
    """Compute cross-sectional momentum scores for each ticker at each date.

    Score for ticker i at date d = close[d] / close[d - lookback] - 1.

    Args:
        prices_dict: Mapping of ticker → DataFrame with 'date' and 'close' columns.
        lookback: Number of trading days for the momentum lookback window.

    Returns:
        Wide DataFrame: 'date' column + one column per ticker with momentum scores.
        Warmup rows (first `lookback` rows) are removed via drop_nulls.
    """
    score_frames: list[pl.DataFrame] = []

    for ticker, df in prices_dict.items():
        scored = df.select([
            pl.col("date"),
            (pl.col("close") / pl.col("close").shift(lookback) - 1).alias(ticker),
        ])
        score_frames.append(scored)

    if not score_frames:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})

    # Join all tickers on date (inner join keeps only dates where all tickers have data)
    result = score_frames[0]
    for frame in score_frames[1:]:
        result = result.join(frame, on="date", how="inner")

    # Drop warmup rows (where any score is null)
    ticker_cols = [c for c in result.columns if c != "date"]
    result = result.drop_nulls(subset=ticker_cols)

    return result


def run_capm_regression(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series,
) -> CAPMResult:
    """Estimate CAPM parameters via OLS regression using pure numpy.

    Model: R_p = alpha + beta * R_b + epsilon

    Args:
        portfolio_returns: Daily portfolio returns series.
        benchmark_returns: Daily benchmark (e.g. SPY) returns series.

    Returns:
        CAPMResult with annualized alpha, beta, t-statistics, and R².

    Raises:
        ValueError: If fewer than 3 observations or benchmark has zero variance.
    """
    y = portfolio_returns.to_numpy().astype(float)
    x = benchmark_returns.to_numpy().astype(float)

    n = len(y)
    if n < 3:
        raise ValueError(f"Need at least 3 observations for CAPM regression, got {n}.")

    # Design matrix: [intercept, benchmark]
    X = np.column_stack([np.ones(n), x])

    # Check for near-singular matrix (zero variance in benchmark)
    XtX = X.T @ X
    if abs(np.linalg.det(XtX)) < 1e-14:
        raise ValueError("Benchmark returns have near-zero variance; CAPM regression is degenerate.")

    # OLS solution
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily, beta = theta[0], theta[1]

    # Residuals and standard errors
    residuals = y - X @ theta
    sse = float(residuals @ residuals)
    sigma_sq = sse / (n - 2)

    XtX_inv = np.linalg.inv(XtX)
    se = np.sqrt(np.diag(XtX_inv) * sigma_sq)

    t_alpha = alpha_daily / se[0] if se[0] > 0 else 0.0
    t_beta = beta / se[1] if se[1] > 0 else 0.0

    # R²
    sst = float(np.var(y) * n)
    r_squared = 1.0 - sse / sst if sst > 0 else 0.0

    return CAPMResult(
        alpha=alpha_daily * 252,
        beta=float(beta),
        t_alpha=float(t_alpha),
        t_beta=float(t_beta),
        r_squared=float(r_squared),
    )


def run_long_short_backtest(
    prices_dict: dict[str, pl.DataFrame],
    top_n: int,
    bottom_n: int,
    lookback: int,
    initial_capital: float = 10_000.0,
    commission_rate: float = 0.0,
    spread_bps: float = 0.0,
) -> LongShortResult:
    """Run a monthly-rebalanced long-short momentum backtest.

    Each month-start, rank tickers by momentum score:
    - Go long top_n (equal-weight, 50% of equity)
    - Go short bottom_n (equal-weight, 50% of equity)

    Args:
        prices_dict: Mapping of ticker → DataFrame with 'date' and 'close' columns.
        top_n: Number of tickers to hold long.
        bottom_n: Number of tickers to sell short.
        lookback: Momentum lookback window in trading days.
        initial_capital: Starting portfolio value.
        commission_rate: Fractional commission per trade.
        spread_bps: Bid-ask spread in basis points (half-spread per side).

    Returns:
        LongShortResult with equity, dates, daily_returns, and monthly_holdings.

    Raises:
        ValueError: If top_n + bottom_n exceeds the number of available tickers.
    """
    universe = list(prices_dict.keys())
    if top_n + bottom_n > len(universe):
        raise ValueError(
            f"top_n ({top_n}) + bottom_n ({bottom_n}) = {top_n + bottom_n} "
            f"exceeds available tickers ({len(universe)})."
        )

    # Build wide close price matrix via inner join
    close_frames = []
    for ticker, df in prices_dict.items():
        renamed = df.select([pl.col("date"), pl.col("close").alias(ticker)])
        close_frames.append(renamed)

    price_matrix = close_frames[0]
    for frame in close_frames[1:]:
        price_matrix = price_matrix.join(frame, on="date", how="inner")
    price_matrix = price_matrix.sort("date")

    dates_list = price_matrix["date"].to_list()
    n_days = len(dates_list)

    if n_days == 0:
        empty_holdings = pl.DataFrame({
            "rebal_date": pl.Series([], dtype=pl.Date),
            "long_tickers": pl.Series([], dtype=pl.Utf8),
            "short_tickers": pl.Series([], dtype=pl.Utf8),
        })
        return LongShortResult(
            equity=pl.Series("equity", [], dtype=pl.Float64),
            dates=pl.Series("date", [], dtype=pl.Date),
            daily_returns=pl.Series("daily_returns", [], dtype=pl.Float64),
            monthly_holdings=empty_holdings,
        )

    # Compute momentum scores (wide DataFrame: date + ticker columns)
    scores_df = compute_momentum_scores(prices_dict, lookback)
    scores_dict: dict = {}
    if not scores_df.is_empty():
        for row in scores_df.iter_rows(named=True):
            d = row["date"]
            scores_dict[d] = {k: v for k, v in row.items() if k != "date"}

    spread_factor = spread_bps / 20_000.0

    # State
    cash = initial_capital
    # long_positions: dict[ticker, shares]
    long_positions: dict[str, float] = {}
    # short_positions: dict[ticker, (shares_short, entry_price)]
    short_positions: dict[str, tuple[float, float]] = {}

    equity_values: list[float] = []
    holdings_records: list[dict] = []

    prev_month: tuple[int, int] | None = None

    for i in range(n_days):
        d = dates_list[i]
        cur_month = (d.year, d.month)

        # Current prices for all tickers
        row_prices: dict[str, float] = {
            ticker: price_matrix[ticker][i] for ticker in universe
        }

        # Mark-to-market: cash + long MtM + short PnL
        long_mtm = sum(
            row_prices[t] * shares for t, shares in long_positions.items()
        )
        short_pnl = sum(
            (entry_px - row_prices[t]) * shares
            for t, (shares, entry_px) in short_positions.items()
        )
        current_equity = cash + long_mtm + short_pnl
        equity_values.append(current_equity)

        # Month-start rebalance (only if we have scores for this date)
        is_month_start = cur_month != prev_month
        if is_month_start and d in scores_dict:
            day_scores = scores_dict[d]

            # 1. Close all long positions
            for ticker, shares in list(long_positions.items()):
                sell_fill = row_prices[ticker] * (1.0 - spread_factor)
                gross = shares * sell_fill
                commission = gross * commission_rate
                cash += gross - commission
            long_positions.clear()

            # 2. Cover all short positions
            for ticker, (shares, _entry_px) in list(short_positions.items()):
                cover_fill = row_prices[ticker] * (1.0 + spread_factor)
                cost = shares * cover_fill
                commission = cost * commission_rate
                cash -= cost + commission
            short_positions.clear()

            # 3. Re-rank tickers by momentum score
            ranked = sorted(
                [(t, s) for t, s in day_scores.items() if s is not None],
                key=lambda x: x[1],
                reverse=True,
            )
            long_tickers = [t for t, _ in ranked[:top_n]]
            short_tickers = [t for t, _ in ranked[-bottom_n:]]

            # Re-compute equity after closing (before opening new positions)
            long_alloc = current_equity * 0.5
            short_alloc = current_equity * 0.5

            # 4. Open long positions (equal-weight on long_alloc)
            if long_tickers:
                per_long = long_alloc / len(long_tickers)
                for ticker in long_tickers:
                    buy_fill = row_prices[ticker] * (1.0 + spread_factor)
                    shares_bought = per_long / buy_fill
                    commission = per_long * commission_rate
                    cash -= per_long + commission
                    long_positions[ticker] = shares_bought

            # 5. Open short positions (equal-weight on short_alloc)
            if short_tickers:
                per_short = short_alloc / len(short_tickers)
                for ticker in short_tickers:
                    short_fill = row_prices[ticker] * (1.0 - spread_factor)
                    shares_shorted = per_short / short_fill
                    proceeds = per_short
                    commission = proceeds * commission_rate
                    cash += proceeds - commission
                    short_positions[ticker] = (shares_shorted, short_fill)

            holdings_records.append({
                "rebal_date": d,
                "long_tickers": ", ".join(long_tickers),
                "short_tickers": ", ".join(short_tickers),
            })

        prev_month = cur_month

    # Close remaining positions at last price
    if long_positions or short_positions:
        last_prices = {ticker: price_matrix[ticker][-1] for ticker in universe}
        for ticker, shares in long_positions.items():
            sell_fill = last_prices[ticker] * (1.0 - spread_factor)
            gross = shares * sell_fill
            commission = gross * commission_rate
            cash += gross - commission
        for ticker, (shares, _entry_px) in short_positions.items():
            cover_fill = last_prices[ticker] * (1.0 + spread_factor)
            cost = shares * cover_fill
            commission = cost * commission_rate
            cash -= cost + commission

    equity_series = pl.Series("equity", equity_values)
    dates_series = pl.Series("date", dates_list)
    daily_returns = equity_series.pct_change().drop_nulls()

    if holdings_records:
        monthly_holdings = pl.DataFrame(holdings_records)
    else:
        monthly_holdings = pl.DataFrame({
            "rebal_date": pl.Series([], dtype=pl.Date),
            "long_tickers": pl.Series([], dtype=pl.Utf8),
            "short_tickers": pl.Series([], dtype=pl.Utf8),
        })

    return LongShortResult(
        equity=equity_series,
        dates=dates_series,
        daily_returns=daily_returns,
        monthly_holdings=monthly_holdings,
    )
