# 🤖 TradeBot

A **free, local-first personal stock-analysis copilot**. It runs as an
interactive console 24/7, watches a list of stocks, and produces clear
**BUY / SELL / HOLD** advice (with explanations) that *you* execute by hand on
Trade Republic. It never connects to a broker and never trades for you.

> Full product spec: [`SPEC.md`](SPEC.md).

## What it does

- **Interactive console (REPL)** — launch `tradebot`, then type commands. No
  argument-based CLI.
- **Signals** combining technical analysis (0.40), chart patterns (0.35) and
  news sentiment (0.25) → a composite score (BUY > +0.20, SELL < −0.20).
- **Two portfolios**: a **real** one (positions you enter by hand) and a
  **simulated** one the bot manages itself to prove it works.
- **Background surveillance** during an **active window** (weekdays 8h–23h
  Europe/Paris by default); it sleeps outside the window to save energy.
- **Market scan** for opportunities and a **weekly newsletter**.
- **Discord notifications** via webhook (Telegram optional).
- **Journal** of every signal/trade/equity snapshot in `data/journal.jsonl`.

## Privacy & safety

- Default market data is **yfinance** — no account, no ID.
- **No broker connection.** The bot only advises and simulates; you place real
  orders yourself and record them with `buy` / `sell`.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd TradeBot
uv sync                  # install dependencies
cp .env.example .env     # optional — all values are optional
```

## Run

```bash
uv run tradebot          # or: python tradebot.py
```

You get a prompt:

```
tradebot> help
tradebot> add NVDA
tradebot> analyze AAPL
tradebot> start          # background surveillance (in active window)
tradebot> buy AAPL 0.5 152.30   # record a real Trade Republic order
tradebot> portfolio
tradebot> sim
tradebot> report
tradebot> quit
```

### Commands

| Command | Description |
|---|---|
| `add / remove / list` | manage the watchlist |
| `analyze SYMBOL [days]` | one-shot analysis + signal |
| `scan` | scan the market for opportunities |
| `buy / sell SYMBOL QTY PRICE` | record a **real** position |
| `portfolio` / `sim` | real / simulated portfolio with live P&L |
| `report [n]` | recent journal activity |
| `status` | market window + background tasks + active user |
| `user [list/add/switch/remove]` | manage users (separate portfolios) |
| `users` | list users |
| `start` / `stop` | background surveillance |
| `quit` | exit cleanly |

### Multiple users

Several people (e.g. a family) can share one instance. Each user has their own
real & simulated portfolios, watchlist and journal under `data/users/<name>/`,
and their name appears in Discord notifications:

```
tradebot[default]> user add maman
tradebot[default]> user switch maman
tradebot[maman]> buy ASML 0.2 900
```

## Configuration

Everything is optional (see [`.env.example`](.env.example)): a Finnhub key
(history; yfinance is the default), a Discord webhook (alerts + newsletter),
Ollama (AI sentiment; rule-based fallback otherwise), the active window, and
the simulation parameters.

## Tests

```bash
uv run pytest -q
```

## Docker

```bash
docker compose run --rm tradebot   # interactive console (TTY)
docker compose run --rm test       # test suite
```

## Architecture

```
src/
  console.py     # interactive REPL — entry point
  engine.py      # application core (analysis + dual portfolios + actions)
  scheduler.py   # energy-sober background loop
  config.py      # .env-driven settings
  core/models.py # domain models (Candle, Position, Portfolio, TradingSignal…)
  data/          # quotes.py (live), finnhub_source.py (history), news_fetcher.py
  analysis/      # technical, pattern_detector, sentiment, signal_aggregator, scanner
  portfolio/     # manager.py, watchlist.py, journal.py
  market/        # schedule.py (active window)
  reporting/     # discord.py, report.py, telegram_bot.py
```
