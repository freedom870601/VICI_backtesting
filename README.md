# VICI Backtesting System

A minimal but verifiable US stock backtesting system built with an AI-only workflow (Claude Code). Features an SMA crossover strategy, five performance metrics, and an interactive Streamlit dashboard.

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
uv pip install -r requirements-dev.txt
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

## Key Assumptions

### Data Source
- **yfinance only** — public, no API key required. Data availability and accuracy depends on Yahoo Finance.
- `auto_adjust=True` is mandatory to avoid spurious signals from dividend/split price discontinuities.
- Minimum backtest period: enough daily data to cover the slow SMA window (default 50 days).

### Strategy
- **SMA Crossover (all-in / all-out)**: 100% of capital is deployed on BUY; full position liquidated on SELL.
- Only crossover bars generate signals (`signal ∈ {-1, 0, 1}`); the engine tracks position state separately.
- No transaction costs, commissions, or slippage by default.

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
| 2 | `test(strategy)` | RED: 10 failing strategy tests |
| 2 | `feat(strategy)` | GREEN: SMA crossover signal generation |
| 3 | `test(engine)` | RED: 9 failing engine tests |
| 3 | `feat(engine)` | GREEN: all-in/all-out backtesting loop |
| 4 | `test(data)` | RED: 6 failing data tests (monkeypatched) |
| 4 | `feat(data)` | GREEN: yfinance fetching with polars output |
| 5 | `feat(app)` | Streamlit dashboard |
| 6 | `chore(docker)` | Dockerfile for Zeabur |
| 7 | `docs` | This README |

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
backtest/engine.py        46      4    91%
backtest/metrics.py       34      0   100%
backtest/strategy.py      13      0   100%
------------------------------------------
TOTAL                    112      5    96%
```

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
├── app.py                  # Streamlit entrypoint
├── backtest/
│   ├── __init__.py
│   ├── data.py             # yfinance data fetching
│   ├── engine.py           # Core backtesting loop
│   ├── metrics.py          # CAGR, Sharpe, MDD, etc.
│   └── strategy.py         # SMA crossover signals
└── tests/
    ├── conftest.py         # Shared fixtures
    ├── test_data.py
    ├── test_engine.py
    ├── test_metrics.py
    └── test_strategy.py
```
