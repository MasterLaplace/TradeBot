# Makefile for common dev tasks

.PHONY: venv install test lint build run-backtest run-paper build-image

venv:
	python3 -m venv venv

install: venv
	source venv/bin/activate && pip install -r requirement.txt

precommit-install:
	source venv/bin/activate && pip install pre-commit && pre-commit install

test:
	source venv/bin/activate && python -m pytest tests/ -v

lint:
	ruff check src tests

build-image:
	docker build -t trading-bot:latest .

run-backtest:
	python tradebot.py backtest --data data/crypto_btc_eth_4h_90d.csv --strategy safe_profit

run-paper:
	python tradebot.py paper --duration 600 --interval 30 --symbols BTCUSDT ETHUSDT --strategy safe_profit
