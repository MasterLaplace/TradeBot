# Makefile for common dev tasks

.PHONY: install install-all test lint build-image run-backtest run-paper setup

# Install core dependencies with uv
install:
	uv pip install -e "."

# Install everything (analysis + notifications + dev)
install-all:
	uv pip install -e ".[all]"

# Setup: install + copy env template
setup: install-all
	@test -f .env || cp .env.example .env && echo "✅ .env created — fill in your API keys"

precommit-install:
	uv pip install pre-commit && pre-commit install

test:
	python -m pytest tests/ -v

lint:
	ruff check src tests

build-image:
	docker build -t trading-bot:latest .

run-backtest:
	python tradebot.py backtest --data data/asset_b_train.csv --strategy safe_profit

run-paper:
	python tradebot.py paper --duration 600 --interval 30 --symbols BTCUSDT ETHUSDT --strategy safe_profit
