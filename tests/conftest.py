"""Shared pytest fixtures for the backtesting test suite."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def flat_prices() -> pl.Series:
    """100-row constant price series for zero-return edge cases."""
    return pl.Series("close", [100.0] * 100)


@pytest.fixture
def trending_up_prices() -> pl.Series:
    """252-day series that doubles in price (CAGR ~ 100%)."""
    start = 100.0
    end = 200.0
    prices = np.linspace(start, end, 252)
    return pl.Series("close", prices)


@pytest.fixture
def sma_crossover_prices() -> pl.Series:
    """
    Engineered 120-day series with known SMA crossover positions.

    Structure:
    - Days 0-49:  Declining phase (fast SMA below slow SMA)
    - Days 50-79: Rising phase that causes fast to cross above slow (BUY signal)
    - Days 80-119: Another decline causing fast to cross below slow (SELL signal)
    """
    rng = np.random.default_rng(42)
    prices = []

    # Phase 1: Declining — starts at 120, trends down to 90
    phase1 = np.linspace(120, 90, 50) + rng.normal(0, 0.5, 50)
    prices.extend(phase1.tolist())

    # Phase 2: Strong rise — from 90 to 140, triggers bullish crossover
    phase2 = np.linspace(90, 140, 30) + rng.normal(0, 0.5, 30)
    prices.extend(phase2.tolist())

    # Phase 3: Decline — from 140 back to 100, triggers bearish crossover
    phase3 = np.linspace(140, 100, 40) + rng.normal(0, 0.5, 40)
    prices.extend(phase3.tolist())

    return pl.Series("close", prices)


@pytest.fixture
def known_returns() -> pl.Series:
    """
    Seeded RNG normal return series for deterministic Sharpe/vol tests.
    Mean ≈ 0.001, std ≈ 0.02, 252 observations.
    """
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.001, scale=0.02, size=252)
    return pl.Series("returns", returns)


@pytest.fixture
def flat_prices_df() -> pl.DataFrame:
    """100-row constant price DataFrame for engine/strategy tests."""
    import datetime

    start = datetime.date(2020, 1, 1)
    dates = [start + datetime.timedelta(days=i) for i in range(100)]
    return pl.DataFrame(
        {
            "date": dates,
            "close": [100.0] * 100,
        }
    )


@pytest.fixture
def trending_up_df() -> pl.DataFrame:
    """252-day doubling price DataFrame for engine tests."""
    import datetime

    start = datetime.date(2020, 1, 1)
    dates = [start + datetime.timedelta(days=i) for i in range(252)]
    prices = np.linspace(100.0, 200.0, 252).tolist()
    return pl.DataFrame({"date": dates, "close": prices})
