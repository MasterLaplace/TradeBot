"""
Price Broadcaster using Binance WebSocket or REST fallback.

This module provides a centralized broadcaster that publishes the most recent
prices for a pair of symbols into a shared manager dict so worker processes
can read them without each making REST requests.

Implementation details:
- WebSocket broadcaster uses Binance combined streams: wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade
- If `websockets` is not available or fails, it falls back to REST polling.

Usage:
  broadcaster = PriceBroadcaster(symbols=['BTCUSDT','ETHUSDT'], interval_seconds=1)
  broadcaster.start(shared_prices, key)
  # workers read shared_prices[key]

"""
from __future__ import annotations
from multiprocessing import Process
from typing import List
from datetime import datetime
import time
import json
import os

try:
    import websockets
except Exception:
    websockets = None  # type: ignore

import requests


def _safe_symbol(sym: str) -> str:
    return sym.lower()


class PriceBroadcaster:
    def __init__(self, symbols: List[str], interval_seconds: float = 1.0, use_ws: bool = True):
        assert len(symbols) == 2, "Symbols must be a two-item list"
        self.symbols = symbols
        self.interval_seconds = interval_seconds
        self.use_ws = use_ws and (websockets is not None)
        self._proc: Process | None = None

    def start(self, shared_prices, key: str):
        if self._proc and self._proc.is_alive():
            return
        self._proc = Process(target=self._run, args=(shared_prices, key))
        self._proc.daemon = True
        self._proc.start()

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None

    def _run(self, shared_prices, key: str) -> None:
        if self.use_ws:
            try:
                self._run_websocket(shared_prices, key)
                return
            except Exception:
                pass
        self._run_rest(shared_prices, key)

    def _run_rest(self, shared_prices, key: str) -> None:
        """Polling loop using REST endpoints to update the shared price cache.

        The method intentionally swallows exceptions to ensure the
        broadcaster keeps running, letting workers use the last known
        price snapshot.
        """
        base_url = 'https://api.binance.com/api/v3'
        a = _safe_symbol(self.symbols[0]).upper()
        b = _safe_symbol(self.symbols[1]).upper()
        while True:
            try:
                r_a = requests.get(f"{base_url}/ticker/price", params={'symbol': a}, timeout=10)
                r_b = requests.get(f"{base_url}/ticker/price", params={'symbol': b}, timeout=10)
                r_a.raise_for_status()
                r_b.raise_for_status()
                price_a = float(r_a.json().get('price', 0))
                price_b = float(r_b.json().get('price', 0))
                shared_prices[key] = {'asset_a': price_a, 'asset_b': price_b, 'timestamp': datetime.utcnow().isoformat()}
            except Exception:
                pass
            time.sleep(max(0.25, self.interval_seconds))

    def _run_websocket(self, shared_prices, key: str) -> None:
        """Subscribe to Binance combined trade streams and update the shared cache.

        The method runs a small async event loop that listens to trade
        messages and stores a combined snapshot for the two symbols.
        """
        a = _safe_symbol(self.symbols[0])
        b = _safe_symbol(self.symbols[1])
        url = f"wss://stream.binance.com:9443/stream?streams={a}@trade/{b}@trade"
        import asyncio

        async def ws_loop():
            async with websockets.connect(url, ping_interval=20) as ws:
                last_a = None
                last_b = None
                while True:
                    msg = await ws.recv()
                    if not msg:
                        continue
                    data = json.loads(msg)
                    if 'data' in data and 's' in data['data']:
                        sym = data['data']['s']
                        price_s = float(data['data'].get('p')) if 'p' in data['data'] else float(data['data'].get('c', 0))
                        ts = data['data'].get('T') or data['data'].get('E') or None
                        if sym.lower() == a:
                            last_a = (price_s, ts)
                        elif sym.lower() == b:
                            last_b = (price_s, ts)

                        if last_a and last_b:
                            timestamp = datetime.utcnow().isoformat()
                            shared_prices[key] = {'asset_a': float(last_a[0]), 'asset_b': float(last_b[0]), 'timestamp': timestamp}

        asyncio.run(ws_loop())
