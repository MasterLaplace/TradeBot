# 🤖 TradeBot v2.0

Algorithmic trading bot built with a SOLID architecture and a single unified CLI entrypoint.

Comprehensive toolkit for backtesting trading strategies, running simulated live (paper) trading, and executing parallel experiments across multiple datasets and strategies.

---

## Getting Started

### Installation

```bash
# Clone the repository and create a virtual environment
git clone <repo-url>
cd Hackaton
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirement.txt
```

### Quick test

```bash
source venv/bin/activate
python -m pytest -q
```

If import errors occur:
```bash
PYTHONPATH=$(pwd) python -m pytest -q
```

### Make `tradebot` a convenience command

**Option 1** — local symlink (ensure `~/.local/bin` is in PATH):
```bash
ln -s "$(pwd)/tradebot.py" "$HOME/.local/bin/tradebot"
```

**Option 2** — alias in zsh:
```bash
echo "alias tradebot='$(pwd)/venv/bin/python $(pwd)/tradebot.py'" >> ~/.zshrc
source ~/.zshrc
```

---

## Summary

This repository includes:
- **Multiple trading strategies** with different risk/reward profiles
- **Backtesting engine** for historical analysis
- **Paper trading engine** for simulated live sessions
- **Parallel runners** for efficient multi-experiment execution
- **Reporting & charts** in Markdown and PNG formats

The codebase follows SOLID principles for maintainability and extensibility.

---

## Architecture

```
tradebot (script: `tradebot.py`)  # CLI entrypoint
src/
├── cli/                 # CLI (argparse + command classes)
├── core/                # Domain models
├── data/                # Data sources (CSV, Binance REST, broadcaster)
├── engine/              # Engines: backtest and paper-trading
├── reporting/           # Reports and charts (Markdown + PNG)
└── strategies/          # Strategy implementations
```

---

## Usage

### Show help

```bash
tradebot --help  # or: python tradebot.py --help
tradebot <command> --help
```

### Available commands

| Command | Description |
|---|---|
| `backtest` | Run a backtest on historical data |
| `compare` | Compare several strategies on the same dataset |
| `paper` | Run a live-simulated (paper) session using live prices |
| `fetch` | Download data from Binance REST into CSV |
| `report` | Generate a report with charts (Markdown + PNG) |
| `list` | List available strategies |
| `test` | Test Binance API connectivity |
| `parallel` | Run parallel backtests or parallel paper sessions |

### Examples

```bash
# Test Binance connectivity
tradebot test

# List available strategies
tradebot list

# Fetch 30 days of BTC/ETH data and save as CSV
tradebot fetch --days 30 --output data/crypto_30d.csv

# Run a backtest on a strategy
tradebot backtest --data data/crypto_30d.csv --strategy safe_profit

# Compare all strategies on the same dataset
tradebot compare --data data/crypto_30d.csv

# Generate a report (Markdown + PNG charts)
tradebot report --data data/crypto_30d.csv --output reports/

# Run 1 hour of paper trading with safe_profit (live prices simulated)
tradebot paper --duration 3600 --strategy safe_profit
```

---

## Parallel Backtests

Use `parallel` to run many strategy/dataset combinations in parallel and generate a summary CSV.

### Key options

- `--datasets` / `-d`: one or more CSV paths or glob patterns (e.g., `data/*.csv`)
- `--strategies` / `-s`: pick a subset of strategies (default: all available)
- `--workers` / `-w`: number of parallel processes (default: cpu_count)
- `--capital` / `-c`: initial capital for each run (default: 10000)
- `--fee-rate`: simulated fee rate (default: 0.001)
- `--per-strategy-dir` / `-p`: optional dir to write one CSV per strategy
- `--append`: append to an existing results file instead of overwriting
- `--no-progress`: disable the progress bar (if `tqdm` is available)

### Examples (parallel)

```bash
# Run parallel backtests on all CSVs in data/
tradebot parallel --datasets data/*.csv

# Run a specific strategy on datasets and write per-strategy CSVs
tradebot parallel -d data/*.csv -s safe_profit composite -w 4 -p outputs/parallel_by_strategy

# Append results to existing CSV
tradebot parallel -d data/*.csv -s safe_profit --append -o outputs/parallel_summary.csv

# Run parallel paper sessions with 2 workers
tradebot parallel --mode paper --strategies safe_profit composite --symbols BTCUSDT ETHUSDT --duration 60 --interval 1 -w 2 --output outputs/parallel_paper_summary.csv
```

**Paper mode note**: In `--mode paper`, the CLI uses a centralized price broadcaster to reduce REST calls (or uses websockets when available). Avoid using a very small `--interval` across many workers to prevent hitting API rate limits.

---

## Strategies

Examples of strategies included in the repo:

| Strategy | Description |
|---|---|
| `safe_profit` | Conservative blend of indicators with reasonable drawdown control |
| `adaptive_trend` | Trend-following with volatility filter |
| `composite` | Multi-indicator strategy (SMA + stoploss + volatility scaling) |
| `sma` | Simple Moving Average crossover |
| `baseline` | A simple momentum baseline |

---

## Progress bar

The progress bar is displayed if `tqdm` is installed. Install it with:

```bash
pip install tqdm
```

---

## Docker

```bash
# Build the Docker image
docker build -t trading-bot:latest .

# Run a backtest inside the container
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs trading-bot:latest backtest --data data/crypto_btc_eth_4h_90d.csv --strategy safe_profit --output /app/outputs/docker_backtest

# Run a 1-hour paper trading session (simulated)
docker run --rm -v $(pwd)/experiments:/app/experiments trading-bot:latest paper --duration 3600 --strategy safe_profit --symbols BTCUSDT ETHUSDT
```

---

## Development

- This project uses a Python virtual environment (`venv`), so that required packages like `pandas`, `numpy`, `matplotlib` and others are available.
- Large dependencies (e.g., `pandas`, `matplotlib`) are imported at module level to avoid surprising runtime import errors. Optional packages (e.g., `websockets`) remain conditional to allow running only a subset of features.
- Run tests with the virtual environment activated.

---

## Data Structure

CSV format expected for historical files:

```csv
epoch,Asset A,Asset B,Cash
0,100000.0,3500.0,1.0
1,100500.0,3520.0,1.0
```

---

## Tests

Run the tests from the repository root with the venv activated.

```bash
source venv/bin/activate
python -m pytest -q
```

If import errors occur, run with PYTHONPATH:

```bash
PYTHONPATH=$(pwd) python -m pytest -q
```

---

## Contribution & Coding Standards

- Use `ruff` for linting and `black` for formatting (or the pre-commit hooks if enabled).
- Add tests for new features and run `pytest` locally before opening a PR.

---

## License

MIT License — Use at your own risk. This software is for educational and simulation purposes.
