# VICI Backtesting System

A verifiable US stock backtesting system built with an AI-only workflow (Claude Code). Features SMA crossover single-stock backtesting, factor analysis long-short momentum, transaction costs, multi-stock comparison, and an interactive Streamlit dashboard.

**Live Demo:** https://vici-backtesting.zeabur.app

---

## How to Run Locally

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd VICI_backtesting

# Create virtual environment and install dependencies
uv venv --python 3.11 .venv
uv sync
```

### Run Tests

```bash
# All tests (verbose)
uv run pytest -v

# With coverage report (target ≥ 80%)
uv run pytest --cov=backtest --cov-report=term-missing
```

### Start the Dashboard

```bash
uv run streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Features

### Single Stock — SMA Crossover
- **Signal**: buys when fast SMA crosses above slow SMA; sells on reverse crossover.
- Fast/slow windows configurable via sidebar sliders.
- All-in / all-out position model: 100% of capital deployed on entry, full liquidation on exit.
- Buy/sell markers shown on equity curve for single-ticker runs.

### Factor Analysis — Long-Short Momentum
- Ranks a user-defined stock universe by momentum score (`close[d] / close[d - lookback] - 1`).
- Monthly rebalance: go long top-N, go short bottom-N (equal-weight, 50/50 allocation).
- CAPM regression (OLS) against SPY benchmark: reports annualized alpha, beta, t-stats, R².

### Transaction Costs
Three independent cost parameters, all reflected in equity curve and trade PnL:

| Parameter | Range | Effect |
|---|---|---|
| **Commission (%)** | 0–2% | Deducted from cash on buy; from gross proceeds on sell |
| **Slippage (%)** | 0–2% | Buy fills at `price × (1 + s)`; sell at `price × (1 - s)` |
| **Bid-Ask Spread (bps)** | 0–50 bps | Half-spread per side: `spread_factor = bps / 20000` |

Slippage and spread model different costs (market impact vs. bid-ask crossing) and are applied multiplicatively. Defaults are 0.

**Entry Price** selectbox: choose `close` (default) or `open` — determines the price used for BUY fills. Useful for next-bar execution simulation.

### Hover Tooltips
All sidebar inputs and metric cards include a `(?)` help icon with plain-English explanations of financial terms (CAGR, Sharpe Ratio, Beta, SMA, etc.), making the dashboard accessible to non-expert users.

### Multi-Stock Comparison
- Enter multiple tickers as a comma-separated list (e.g. `AAPL, MSFT, GOOG`).
- Invalid or unavailable tickers show a warning and are skipped; valid ones still run.
- **Metrics table**: side-by-side CAGR, Volatility, Sharpe, Max Drawdown, Win Rate for all tickers.
- **Normalized equity curves**: all tickers indexed to 100 at start, SPY benchmark overlaid as a dashed line.
- **Per-ticker trade logs**: each ticker has its own expandable trade log section.

---

## Key Assumptions

### Data Source
- **yfinance only** — public, no API key required. Data availability and accuracy depends on Yahoo Finance.
- `auto_adjust=True` is mandatory to avoid spurious signals from dividend/split price discontinuities.
- Minimum backtest period: enough daily data to cover the strategy's warmup window.

### Strategy
- All-in / all-out position management; no partial sizing or leverage.
- SMA Crossover: only crossover bars generate signals (`signal ∈ {-1, 0, 1}`).
- Factor Analysis: monthly rebalance long-short using cross-sectional momentum scores.

### Metrics
| Metric | Formula |
|---|---|
| CAGR | `(end/start)^(1/years) - 1` |
| Annualized Volatility | `daily_returns.std() × √252` |
| Sharpe Ratio | `(mean_return × 252 - rfr) / annualized_vol` (rfr = 2%) |
| Max Drawdown | `max((rolling_max - equity) / rolling_max)` |
| Win Rate | `n_trades_with_pnl > 0 / total_trades` |

### Benchmark
- SPY (Buy & Hold) is normalized to the same initial capital for fair visual comparison.

### Time Zone
- All prices are in US market local time (NYSE) as provided by yfinance.

---

## AI Workflow Documentation

This project was built entirely using **Claude Code** (Sonnet 4.6) following a strict TDD workflow.

### Skills Invoked
- `test-driven-development` — Red → Green → Refactor cycle for every module
- `python-testing-patterns` — pytest fixtures, monkeypatching, known-value assertions
- `writing-plans` — structured implementation plan before writing code

### Example Prompts Used

```
"Implement the following plan: [full implementation plan with phases 0-7]"

"Continue implementing the plan, use uv to create the virtual environment"
```

### Development Process

| Phase | Commit | Description |
|---|---|---|
| 0 | `chore(deps)` | requirements.txt, requirements-dev.txt |
| 0 | `chore(structure)` | Package `__init__.py` files |
| 0 | `test(conftest)` | Shared pytest fixtures |
| 1 | `test(metrics)` | RED: 23 failing metric tests |
| 1 | `feat(metrics)` | GREEN: all 23 tests passing |
| 1 | `refactor(metrics)` | Extract `_validate_non_empty`, add `__all__` |
| 2 | `test(strategy)` | RED: 10 failing SMA strategy tests |
| 2 | `feat(strategy)` | GREEN: SMA crossover signal generation |
| 3 | `test(engine)` | RED: 9 failing engine tests |
| 3 | `feat(engine)` | GREEN: all-in/all-out backtesting loop |
| 4 | `test(data)` | RED: 6 failing data tests (monkeypatched) |
| 4 | `feat(data)` | GREEN: yfinance fetching with polars output |
| 5 | `feat(app)` | Streamlit dashboard (single stock, SMA) |
| 6 | `chore(docker)` | Dockerfile for Zeabur |
| 7 | `docs` | README |
| 8a | `feat(data)` | Add open price column; null-fill with close |
| 8b | `feat(engine)` | Add spread_bps, entry_price_type, entry_type column in trade log |
| 8c | `feat(app)` | Transaction costs UI, multi-stock comparison, entry price selectbox |
| 9 | `feat(factor)` | Factor analysis: long-short backtest, CAPM regression, monthly holdings |
| 10 | `refactor(strategy)` | Remove single-stock momentum; SMA Crossover is the only single-stock strategy |
| 11 | `feat(metrics)` | Add monthly returns, rolling Sharpe/vol, holding period stats |
| 12 | `feat(app)` | Reorganize single-stock results into tabs; add dark theme |
| 13 | `feat(app)` | Add hover tooltips (`help=`) on all sidebar inputs and metric cards |

### Manual Corrections Made
- **CAGR test**: Fixed test to use 253 data points (252 intervals = 1 year) rather than 2 points
- **Sharpe zero-vol guard**: Changed `== 0.0` to `< 1e-10` to handle floating-point precision in polars `.std()`
- **uv Python pin**: Ran `uv python pin 3.11` to resolve version conflict from conda environment

### Test Coverage Results
```
Name                   Stmts   Miss  Cover
------------------------------------------
backtest/__init__.py       0      0   100%
backtest/data.py          19      1    95%
backtest/engine.py        57      6    89%
backtest/factor.py       149      4    97%
backtest/metrics.py       34      0   100%
backtest/strategy.py      13      0   100%
------------------------------------------
TOTAL                    272     11    96%
```
101 tests passing.

---

## Project Structure

```
.
├── CLAUDE.md               # AI collaboration guidelines
├── README.md               # This file
├── Dockerfile              # Zeabur deployment
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Dev/test dependencies
├── .python-version         # Pins Python 3.11 for uv
├── app.py                  # Streamlit entrypoint (Single Stock + Factor Analysis tabs)
├── backtest/
│   ├── __init__.py
│   ├── data.py             # yfinance data fetching
│   ├── engine.py           # Core backtesting loop (commission, slippage, spread)
│   ├── metrics.py          # CAGR, Sharpe, MDD, etc.
│   ├── strategy.py         # SMA crossover signal generation
│   └── factor.py           # Momentum scores, CAPM regression, long-short backtest
└── tests/
    ├── conftest.py         # Shared fixtures
    ├── test_data.py
    ├── test_engine.py
    ├── test_metrics.py
    ├── test_strategy.py
    └── test_factor.py
```
