#!/usr/bin/env python3
"""
📥 Historical Data Fetcher
===========================
Downloads historical OHLCV data from Binance for backtesting.

No API key required - uses public endpoints.
Respects rate limits.

Usage:
  source venv/bin/activate
  python scripts/fetch_historical_data.py --symbol BTCUSDT --interval 1h --days 30 --out data/btcusdt_1h_30d.csv
"""
import argparse
import time
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

# Binance public API endpoint
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Interval mappings
INTERVALS = {
    '1m': 60 * 1000,
    '3m': 3 * 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '2h': 2 * 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '6h': 6 * 60 * 60 * 1000,
    '8h': 8 * 60 * 60 * 1000,
    '12h': 12 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    '3d': 3 * 24 * 60 * 60 * 1000,
    '1w': 7 * 24 * 60 * 60 * 1000,
}


def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int, 
                 limit: int = 1000) -> List[List]:
    """
    Fetch klines (candlesticks) from Binance.
    
    Returns list of:
    [open_time, open, high, low, close, volume, close_time, 
     quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
    """
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': limit
    }
    
    try:
        r = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠️ Error fetching data: {e}")
        return []


def fetch_historical_data(symbol: str, interval: str, days: int, 
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Fetch historical data for a given period.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval (e.g., '1h', '4h', '1d')
        days: Number of days of history
        end_date: End date (default: now)
    
    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=days)
    
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    interval_ms = INTERVALS.get(interval, 60 * 60 * 1000)
    
    print(f"📥 Fetching {symbol} {interval} data from {start_date.date()} to {end_date.date()}...")
    
    all_klines = []
    current_start = start_ms
    
    while current_start < end_ms:
        klines = fetch_klines(symbol, interval, current_start, end_ms)
        
        if not klines:
            break
        
        all_klines.extend(klines)
        
        # Update start time for next batch
        current_start = klines[-1][0] + interval_ms
        
        # Progress
        progress = (current_start - start_ms) / (end_ms - start_ms) * 100
        print(f"  Progress: {min(100, progress):.1f}% ({len(all_klines)} candles)", end='\r')
        
        # Rate limit: 1200 requests per minute, be conservative
        time.sleep(0.1)
    
    print()
    
    if not all_klines:
        print("❌ No data fetched")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Convert types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)
    
    df['trades'] = df['trades'].astype(int)
    
    # Set index
    df.set_index('open_time', inplace=True)
    
    print(f"✅ Fetched {len(df)} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def convert_to_bot_format(df: pd.DataFrame, symbol: str = 'Asset B') -> pd.DataFrame:
    """
    Convert OHLCV data to the format expected by bot_trade.py.
    
    The bot expects: index (epoch), 'Asset B' (price column)
    """
    bot_df = pd.DataFrame()
    bot_df['epoch'] = range(len(df))
    bot_df[symbol] = df['close'].values
    bot_df['Cash'] = 1.0
    bot_df.set_index('epoch', inplace=True)
    return bot_df


def main():
    parser = argparse.ArgumentParser(description='Fetch historical data from Binance')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair')
    parser.add_argument('--interval', default='1h', 
                        choices=list(INTERVALS.keys()),
                        help='Candle interval')
    parser.add_argument('--days', type=int, default=30, help='Days of history')
    parser.add_argument('--out', default='data/historical_btcusdt.csv',
                        help='Output CSV file')
    parser.add_argument('--format', choices=['raw', 'bot'], default='bot',
                        help='Output format (raw=OHLCV, bot=compatible with bot_trade)')
    parser.add_argument('--second-symbol', default=None,
                        help='Second symbol for multi-asset mode (e.g., ETHUSDT)')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    
    # Fetch primary symbol
    df = fetch_historical_data(args.symbol, args.interval, args.days)
    
    if df.empty:
        print("❌ Failed to fetch data")
        return
    
    # If second symbol provided, fetch and merge
    if args.second_symbol:
        print()
        df2 = fetch_historical_data(args.second_symbol, args.interval, args.days)
        
        if not df2.empty:
            # Align by timestamp
            df = df.join(df2[['close']], rsuffix='_2', how='inner')
            df.rename(columns={'close_2': f'{args.second_symbol}_close'}, inplace=True)
    
    # Convert format
    if args.format == 'bot':
        if args.second_symbol and f'{args.second_symbol}_close' in df.columns:
            bot_df = pd.DataFrame()
            bot_df['epoch'] = range(len(df))
            bot_df['Asset A'] = df['close'].values
            bot_df['Asset B'] = df[f'{args.second_symbol}_close'].values
            bot_df['Cash'] = 1.0
            bot_df.set_index('epoch', inplace=True)
            df = bot_df
        else:
            df = convert_to_bot_format(df)
    
    # Save
    df.to_csv(args.out)
    print(f"\n📁 Saved to: {args.out}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Preview
    print(f"\n📊 Preview (first 5 rows):")
    print(df.head())
    print(f"\n📊 Preview (last 5 rows):")
    print(df.tail())


if __name__ == '__main__':
    main()
