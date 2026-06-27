"""
Live Quote Helper.

Lightweight current-price lookup used for portfolio mark-to-market.
Uses yfinance (no key, no account) with a tiny in-process cache to avoid
hammering the source within a single analysis cycle.
"""

import logging
import time
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}  # symbol -> (price, fetched_at)
_CACHE_TTL = 60.0  # seconds


def get_current_price(symbol: str) -> Optional[float]:
    """Return the latest price for a single symbol, or None if unavailable."""
    symbol = symbol.upper()
    now = time.time()
    cached = _CACHE.get(symbol)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0]

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        price = None
        try:
            price = float(ticker.fast_info.last_price)
        except Exception:
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        if price and price > 0:
            _CACHE[symbol] = (price, now)
            return price
    except Exception as e:
        logger.warning(f"Could not fetch current price for {symbol}: {e}")

    return None


def get_current_prices(symbols: Iterable[str]) -> Dict[str, float]:
    """Return latest prices for several symbols. Missing ones are omitted."""
    prices: Dict[str, float] = {}
    for sym in symbols:
        p = get_current_price(sym)
        if p is not None:
            prices[sym.upper()] = p
    return prices
