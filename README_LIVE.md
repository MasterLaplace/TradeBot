# Live testing & paper trading

This guide shows simple steps to test `bot_trade.py` in real-time using public exchanges or paper accounts.

## Options

1. Binance (public REST polling)
   - Lightweight and no API key required for public prices.
   - Use `scripts/live_connector_binance.py` to poll prices and log decisions.
   - Example:
     ```bash
     source venv/bin/activate
     python scripts/live_connector_binance.py --symbols BTCUSDT ETHUSDT --interval 1 --duration 60 --out experiments/live/binance.csv --reset
     ```

2. Alpaca (paper trading for US stocks)
   - Requires API key & secret (paper account) and environment variables.
   - Use `scripts/live_connector_alpaca.py` to poll bar data and optionally submit paper orders.

   Example (set env vars first):
   ```bash
   export APCA_API_KEY_ID=your_key
   export APCA_API_SECRET_KEY=your_secret
   export APCA_API_BASE_URL=https://paper-api.alpaca.markets
   source venv/bin/activate
   python scripts/live_connector_alpaca.py --symbol AAPL --interval 60 --duration 3600 --out experiments/live/alpaca.json
   ```

3. Important Safety Guidelines
   - Start with paper/testnet; do not run with real funds until thoroughly validated.
   - Include logging for decisions and simulated executed orders before moving to paper orders.
   - Use small risk amounts and verify slippage & fees.
   - Watch deployments and set alerts.

4. Suggested workflow
   - Use `live_connector_binance.py` with `--duration` for a short test (60-300s) to validate decision stream.
   - Run `scripts/ensemble_simulator.py` or `scripts/prod_run_simulator.py` against the logged CSV to compute backtest metrics.
   - If results are promising, use Alpaca paper or Binance testnet to run a longer order-simulated test.

5. Optional improvements
   - Use `ccxt` with websockets support for lower-latency data.
   - Add an orders simulator and slippage model.
   - Add monitoring and alerting (Slack/Email) for thresholds.

If you want, I can: (A) create a Docker image for this setup, (B) add a GitHub Actions workflow to run a daily paper test, or (C) start a CCXT-based websocket connector. Tell me which you want next.
