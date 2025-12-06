#!/usr/bin/env python3
"""
Alpaca paper trading template for live testing.
This is a template and requires environment variables `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` and `APCA_API_BASE_URL` (paper).
It executes paper orders, but for safety it's implemented to just log decisions by default. Remove the safety flags only if you
fully understand the risk.

Usage:
  (create an Alpaca paper account and set env vars) then:
  venv/bin/python scripts/live_connector_alpaca.py --symbol AAPL --interval 1 --duration 60

Note: This script uses the `alpaca-trade-api` package (pip install alpaca-trade-api). It's included as a template; run at your own risk.
"""
import argparse
import os
import time
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bot_trade import make_decision, reset_history

try:
    from alpaca_trade_api.rest import REST, TimeFrame
except Exception:
    REST = None


def run_alpaca(symbol, api, out, interval=1.0, duration=60.0):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    start = time.time()
    hist = []
    rows = []

    while True:
        now = time.time()
        if duration > 0 and (now - start) >= duration:
            break
        # fetch latest price (bar) via REST
        try:
            bars = api.get_bars(symbol, TimeFrame.Minute, limit=1).df
            last_price = bars['close'].iloc[-1]
        except Exception as e:
            print('Error fetching bars', e)
            time.sleep(interval)
            continue
        # use bot to decide
        try:
            decision = make_decision(int(time.time()), float(last_price), history=hist)
        except Exception as e:
            print('Error running decision', e)
            decision = None
        # Save local log
        rows.append({'ts': int(time.time()), 'symbol': symbol, 'price': float(last_price), 'decision': json.dumps(decision)})
        hist.append({'epoch': int(time.time()), 'price': float(last_price)})
        time.sleep(interval)
    # save data
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print('Saved', out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='AAPL')
    parser.add_argument('--out', default='experiments/live/alpaca_live_log.json')
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--duration', type=float, default=300.0)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--paper', action='store_true', help='Actually submit paper orders')
    args = parser.parse_args()

    if args.reset:
        reset_history()

    key = os.getenv('APCA_API_KEY_ID')
    secret = os.getenv('APCA_API_SECRET_KEY')
    base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    if not key or not secret:
        print('APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set to use Alpaca API')
        sys.exit(1)
    api = REST(key, secret, base_url=base_url)

    run_alpaca(args.symbol, api, args.out, interval=args.interval, duration=args.duration)


if __name__ == '__main__':
    main()
