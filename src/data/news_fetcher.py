"""
Financial News Fetcher Module.

Collects financial news articles from Finnhub API and converts them
into structured NewsArticle objects for sentiment analysis.

Supports:
- Company-specific news (by ticker symbol)
- Market-wide general news
- Local caching to avoid duplicate processing and API waste

Usage:
    fetcher = NewsFetcher(api_key="your_finnhub_key")
    articles = await fetcher.fetch_company_news("AAPL", days=7)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging

import aiohttp

from ..core.models import NewsArticle

logger = logging.getLogger(__name__)


class NewsFetcher:
    """
    Asynchronous financial news collector using Finnhub API.

    Features:
    - Company-specific and market-wide news retrieval
    - Built-in deduplication via URL tracking
    - Rate limiting (respects Finnhub 30 req/s limit)
    - Async-first design for integration with the trading pipeline
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str = "", max_cache_size: int = 1000):
        self.api_key = api_key
        self._seen_urls: Set[str] = set()
        self._cache: Dict[str, List[NewsArticle]] = {}
        self._max_cache_size = max_cache_size

    async def fetch_company_news(
        self,
        symbol: str,
        days: int = 7,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[NewsArticle]:
        """Fetch news articles related to a specific company.

        Args:
            symbol: Ticker symbol (e.g. "AAPL", "MSFT").
            days: Number of days of history to fetch.
            session: Optional shared aiohttp session.

        Returns:
            List of NewsArticle objects, deduplicated and sorted by timestamp.
        """
        # No Finnhub key → go straight to the free yfinance news source.
        if not self.api_key:
            return await self._fallback_yfinance_news(symbol.upper())

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        params = {
            "symbol": symbol.upper(),
            "from": from_date,
            "to": to_date,
            "token": self.api_key,
        }

        return await self._fetch_and_parse(
            endpoint="/company-news",
            params=params,
            cache_key=f"company_{symbol.upper()}_{from_date}",
            session=session,
        )

    async def fetch_market_news(
        self,
        category: str = "general",
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[NewsArticle]:
        """Fetch general market news.

        Args:
            category: News category ("general", "forex", "crypto", "merger").
            session: Optional shared aiohttp session.

        Returns:
            List of NewsArticle objects.
        """
        params = {
            "category": category,
            "token": self.api_key,
        }

        return await self._fetch_and_parse(
            endpoint="/news",
            params=params,
            cache_key=f"market_{category}",
            session=session,
        )

    async def fetch_news_for_watchlist(
        self,
        symbols: List[str],
        days: int = 3,
    ) -> Dict[str, List[NewsArticle]]:
        """Fetch news for all symbols in the watchlist concurrently.

        Args:
            symbols: List of ticker symbols.
            days: Number of days of history.

        Returns:
            Dict mapping symbol to its list of NewsArticle objects.
        """
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.fetch_company_news(symbol, days=days, session=session)
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        news_by_symbol: Dict[str, List[NewsArticle]] = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch news for {symbol}: {result}")
                news_by_symbol[symbol] = []
            else:
                news_by_symbol[symbol] = result

        return news_by_symbol

    async def _fetch_and_parse(
        self,
        endpoint: str,
        params: dict,
        cache_key: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[NewsArticle]:
        """Fetch from Finnhub API and parse into NewsArticle objects.

        Handles session management, error handling, deduplication, and caching.
        """
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()

        try:
            url = f"{self.BASE_URL}{endpoint}"
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 429:
                    logger.warning("Finnhub rate limit hit, waiting 1s...")
                    await asyncio.sleep(1)
                    async with session.get(url, params=params) as retry_resp:
                        retry_resp.raise_for_status()
                        raw_articles = await retry_resp.json()
                else:
                    response.raise_for_status()
                    raw_articles = await response.json()

            if not isinstance(raw_articles, list):
                logger.warning(f"Unexpected Finnhub response for {endpoint}: {type(raw_articles)}")
                return []

            articles = self._parse_articles(raw_articles)

            # Cache results (with size limit)
            if len(self._cache) >= self._max_cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = articles

            return articles

        except aiohttp.ClientError as e:
            logger.error(f"Finnhub API error for {endpoint}: {e.status}, message='{getattr(e, 'message', '')}', url='{e.request_info.url if hasattr(e, 'request_info') else ''}'")
            symbol = params.get("symbol")
            if symbol and endpoint == "/company-news":
                return await self._fallback_yfinance_news(symbol)
            return []
        finally:
            if own_session and session:
                await session.close()

    async def _fallback_yfinance_news(self, symbol: str) -> List[NewsArticle]:
        """Fallback to fetching news using yfinance."""
        import yfinance as yf
        logger.info(f"Using yfinance news fallback for {symbol}")
        try:
            # yfinance news is synchronous, but fast enough for a fallback
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news
            if not raw_news:
                return []
                
            articles = []
            for item in raw_news:
                url = item.get("link", "")
                if url in self._seen_urls:
                    continue
                self._seen_urls.add(url)
                
                # yfinance timestamps are usually unix ints
                ts = item.get("providerPublishTime", 0)
                timestamp = datetime.fromtimestamp(ts) if ts else datetime.now()
                
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("summary", "") or item.get("title", ""),
                    source=item.get("publisher", "Yahoo Finance"),
                    url=url,
                    timestamp=timestamp,
                    related_symbols=[symbol],
                )
                articles.append(article)
            
            # Sort by newest first
            articles.sort(key=lambda a: a.timestamp, reverse=True)
            return articles
            
        except Exception as e:
            logger.error(f"yfinance news fallback failed for {symbol}: {e}")
            return []

    def _parse_articles(self, raw_articles: list) -> List[NewsArticle]:
        """Parse raw Finnhub JSON articles into NewsArticle objects.

        Deduplicates by URL and sorts by timestamp (newest first).
        """
        articles = []
        for raw in raw_articles:
            url = raw.get("url", "")

            # Deduplicate
            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            try:
                timestamp = datetime.fromtimestamp(raw.get("datetime", 0))
                related = raw.get("related", "")
                symbols = [s.strip() for s in related.split(",") if s.strip()] if related else []

                article = NewsArticle(
                    title=raw.get("headline", ""),
                    summary=raw.get("summary", ""),
                    source=raw.get("source", ""),
                    url=url,
                    timestamp=timestamp,
                    related_symbols=symbols,
                )
                articles.append(article)
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping malformed article: {e}")
                continue

        # Sort by newest first
        articles.sort(key=lambda a: a.timestamp, reverse=True)
        return articles

    def clear_cache(self) -> None:
        """Clear the internal cache and seen URLs set."""
        self._cache.clear()
        self._seen_urls.clear()
