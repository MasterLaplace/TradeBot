#!/usr/bin/env python3
"""
Simple Binance public REST poller for live-price testing.
This script polls Binance public APIs for the last price of given symbols (e.g. BTCUSDT), calls `make_decision`
from `bot_trade` and logs decisions in a CSV in `experiments/live`.

Usage:
  venv/bin/python scripts/live_connector_binance.py --symbols BTCUSDT ETHUSDT --interval 1.0 --duration 60

Notes:
- This script doesn't make trades; it's a 'paper' simulation that logs decisions and positions.
- For real trading, use APIs with key/secret and a paper/testnet endpoint (Binance Testnet, `ccxt`, or Alpaca).
"""
import argparse
import time
import os
import csv
import requests
from importlib import import_module
import json

# Add project root to path and import bot
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bot_trade import make_decision, reset_history


def get_binance_price(symbol: str):
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    j = r.json()
    return float(j['price'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'], help='Symbols to monitor (Binance format)')
    parser.add_argument('--interval', type=float, default=1.0, help='Polling interval in seconds')
    parser.add_argument('--duration', type=float, default=60.0, help='Duration in seconds (0 for infinite)')
    parser.add_argument('--out', default='experiments/live/binance_live_log.csv')
    parser.add_argument('--reset', action='store_true', help='Reset module history before starting')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.reset:
        reset_history()

    headers = ['ts', 'symbol', 'price', 'decision', 'assetA', 'assetB', 'cash']
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)

        start = time.time()
        count = 0
        while True:
            now = time.time()
            if args.duration > 0 and (now - start) >= args.duration:
                break
            for symbol in args.symbols:
                try:
                    price = get_binance_price(symbol)
                except Exception as e:
                    print('Error fetching price for', symbol, e)
                    continue
                # feed price into the bot
                # if only one symbol, call make_decision with price; if 2, pass two for multi-asset
                try:
                    if len(args.symbols) >= 2:
                        # For simplicity, pass price for the first two symbols as price and priceB
                        # This only makes sense if there are two assets to consider
                        # If symbol is not among first two, still call make_decision as single asset using BTCUSDT
                        if len(args.symbols) >= 2:
                            # find two assets to pass: first two in list
                            pA = get_binance_price(args.symbols[0])
                            pB = get_binance_price(args.symbols[1])
                            dec = make_decision(int(time.time()), pA, pB)
                        else:
                            dec = make_decision(int(time.time()), price)
                    else:
                        dec = make_decision(int(time.time()), price)
                except Exception as e:
                    print('Bot error', e)
                    dec = None

                # write a row for this symbol
                if dec is None:
                    dec_str = ''
                    assetA = ''
                    assetB = ''
                    cash = ''
                else:
                    dec_str = json.dumps(dec)
                    assetA = dec.get('Asset A', '') if isinstance(dec, dict) else ''
                    assetB = dec.get('Asset B', '') if isinstance(dec, dict) else dec.get('Asset B', '') if isinstance(dec, dict) else ''
                    cash = dec.get('Cash', '') if isinstance(dec, dict) else ''
                w.writerow([int(time.time()), symbol, price, dec_str, assetA, assetB, cash])
                f.flush()
            count += 1
            time.sleep(args.interval)
    print('Finished live polling, rows written:', count)


if __name__ == '__main__':
    main()
