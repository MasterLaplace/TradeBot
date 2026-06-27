"""
Live Quote Helper.

Lightweight current-price lookup used for portfolio mark-to-market.
Uses yfinance (no key, no account) with a tiny in-process cache to avoid
hammering the source within a single analysis cycle.

Prices are normalized to a single **base currency** (default EUR) so the
portfolios — which mix US ($), Korean (₩), European (€) instruments — value
consistently. A US ticker quoted in USD and a Korean GDR quoted in KRW are
both converted to EUR before being returned.
"""

import logging
import time
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}  # symbol -> (price_in_base, fetched_at)
_CACHE_TTL = 60.0  # seconds

_FX_CACHE: Dict[tuple, tuple] = {}  # (from,to) -> (rate, fetched_at)
_FX_TTL = 3600.0  # FX moves slowly enough; refresh hourly

_NAME_CACHE: Dict[str, str] = {}  # symbol -> company name (names rarely change)


def get_company_name(symbol: str) -> str:
    """Return the company's display name for a ticker (falls back to the ticker)."""
    symbol = symbol.upper()
    if symbol in _NAME_CACHE:
        return _NAME_CACHE[symbol]
    name = symbol
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).get_info()
        name = info.get("longName") or info.get("shortName") or symbol
    except Exception as e:
        logger.debug(f"Could not fetch company name for {symbol}: {e}")
    _NAME_CACHE[symbol] = name
    return name


def quote_url(symbol: str) -> str:
    """Public quote page for a ticker (Yahoo Finance — supports US/EU/Asia)."""
    return f"https://finance.yahoo.com/quote/{symbol.upper()}"


def search_symbols(query: str, limit: int = 8) -> list:
    """Search tickers by company name. Returns [{symbol, name, exchange}, ...]."""
    out = []
    try:
        import yfinance as yf

        results = yf.Search(query, max_results=limit)
        for q in results.quotes:
            sym = q.get("symbol")
            if not sym:
                continue
            name = (q.get("shortname") or q.get("longname") or "").strip()
            out.append({
                "symbol": sym,
                "name": name,
                "exchange": q.get("exchDisp") or q.get("exchange") or "",
            })
    except Exception as e:
        logger.warning(f"Symbol search failed for '{query}': {e}")
    return out


def _base_currency() -> str:
    from ..config import get_settings
    return get_settings().base_currency.upper()


def _fx_rate(from_cur: str, to_cur: str) -> Optional[float]:
    """Return how many `to_cur` one `from_cur` buys (e.g. USD->EUR)."""
    from_cur, to_cur = from_cur.upper(), to_cur.upper()
    if from_cur == to_cur:
        return 1.0

    key = (from_cur, to_cur)
    now = time.time()
    cached = _FX_CACHE.get(key)
    if cached and (now - cached[1]) < _FX_TTL:
        return cached[0]

    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{from_cur}{to_cur}=X")
        rate = None
        try:
            rate = float(ticker.fast_info.last_price)
        except Exception:
            hist = ticker.history(period="5d")
            if not hist.empty:
                rate = float(hist["Close"].iloc[-1])
        if rate and rate > 0:
            _FX_CACHE[key] = (rate, now)
            return rate
    except Exception as e:
        logger.warning(f"Could not fetch FX rate {from_cur}->{to_cur}: {e}")
    return None


def get_current_price(symbol: str, to_currency: Optional[str] = None) -> Optional[float]:
    """Return the latest price for one symbol, converted to the base currency.

    Returns None if the price or the required FX conversion is unavailable
    (callers then fall back to the position's entry price, i.e. flat P&L,
    rather than showing a currency-mismatched figure).
    """
    symbol = symbol.upper()
    base = (to_currency or _base_currency()).upper()
    now = time.time()
    cached = _CACHE.get(symbol)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0]

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        price = None
        currency = None
        try:
            fi = ticker.fast_info
            price = float(fi.last_price)
            currency = (fi.currency or base).upper()
        except Exception:
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
            # currency unknown from history; assume base to avoid bad conversion
            currency = base

        if not price or price <= 0:
            return None

        if currency != base:
            rate = _fx_rate(currency, base)
            if rate is None:
                logger.warning(
                    f"Skipping {symbol}: no FX {currency}->{base} for valuation."
                )
                return None
            price *= rate

        _CACHE[symbol] = (price, now)
        return price
    except Exception as e:
        logger.warning(f"Could not fetch current price for {symbol}: {e}")

    return None


def get_current_prices(symbols: Iterable[str], to_currency: Optional[str] = None) -> Dict[str, float]:
    """Return base-currency prices for several symbols. Missing ones are omitted."""
    prices: Dict[str, float] = {}
    for sym in symbols:
        p = get_current_price(sym, to_currency=to_currency)
        if p is not None:
            prices[sym.upper()] = p
    return prices
