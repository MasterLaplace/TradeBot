#!/usr/bin/env python3
"""
🚀 Binance WebSocket connector for real-time price feeds.
Uses websocket streams for lower latency than REST polling.

For EU users: Binance.com may be restricted in some countries.
This script can fallback to Kraken or use Binance EU endpoints.

Usage:
  source venv/bin/activate
  python scripts/live_connector_binance_ws.py --symbols BTCUSDT ETHUSDT --duration 300

Features:
- Real-time WebSocket price updates
- Automatic reconnection
- Paper trading simulation
- CSV logging with PnL tracking
- Dashboard output to terminal
"""
import argparse
import asyncio
import json
import time
import os
import csv
from datetime import datetime
from typing import Dict, Optional

try:
    import websockets
except ImportError:
    print("❌ Please install websockets: pip install websockets")
    exit(1)

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bot_trade import make_decision, reset_history


# ============= Configuration =============
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
# Fallback URLs for EU users (if Binance blocks your IP)
BINANCE_EU_WS_URL = "wss://stream.binance.com:443/ws"  # Port 443 sometimes works
KRAKEN_WS_URL = "wss://ws.kraken.com"


class PaperTrader:
    """Simple paper trading tracker."""
    def __init__(self, initial_cash: float = 10000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.entry_prices: Dict[str, float] = {}
        self.total_trades = 0
        self.winning_trades = 0
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        value = self.cash
        for symbol, qty in self.positions.items():
            if symbol in prices:
                value += qty * prices[symbol]
        return value
    
    def get_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate profit/loss."""
        return self.get_portfolio_value(prices) - self.initial_cash
    
    def get_pnl_pct(self, prices: Dict[str, float]) -> float:
        """Calculate profit/loss percentage."""
        return (self.get_pnl(prices) / self.initial_cash) * 100


class LiveDashboard:
    """Terminal dashboard for monitoring."""
    def __init__(self):
        self.last_update = time.time()
        self.prices: Dict[str, float] = {}
        self.decisions: list = []
        
    def update(self, symbol: str, price: float, decision: dict, trader: PaperTrader):
        self.prices[symbol] = price
        self.decisions.append(decision)
        
        # Clear screen and print dashboard
        if time.time() - self.last_update > 0.5:  # Update every 0.5s max
            self.print_dashboard(trader)
            self.last_update = time.time()
    
    def print_dashboard(self, trader: PaperTrader):
        # ANSI escape codes for formatting
        CLEAR = "\033[2J\033[H"
        BOLD = "\033[1m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        RESET = "\033[0m"
        
        print(CLEAR)
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}{CYAN}🤖 TRADING BOT - LIVE DASHBOARD{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Prices
        print(f"{BOLD}📊 PRICES:{RESET}")
        for symbol, price in self.prices.items():
            print(f"  {symbol}: ${price:,.2f}")
        print()
        
        # Portfolio
        pnl = trader.get_pnl(self.prices)
        pnl_pct = trader.get_pnl_pct(self.prices)
        pnl_color = GREEN if pnl >= 0 else RED
        
        print(f"{BOLD}💰 PORTFOLIO:{RESET}")
        print(f"  Cash: ${trader.cash:,.2f}")
        print(f"  Value: ${trader.get_portfolio_value(self.prices):,.2f}")
        print(f"  PnL: {pnl_color}${pnl:+,.2f} ({pnl_pct:+.2f}%){RESET}")
        print()
        
        # Last decision
        if self.decisions:
            dec = self.decisions[-1]
            print(f"{BOLD}🎯 LAST DECISION:{RESET}")
            if dec:
                asset_a = dec.get('Asset A', 0)
                asset_b = dec.get('Asset B', 0)
                cash = dec.get('Cash', 0)
                print(f"  Asset A: {asset_a:.4f}")
                print(f"  Asset B: {asset_b:.4f}")
                print(f"  Cash: {cash:.4f}")
        
        print()
        print(f"{YELLOW}Press Ctrl+C to stop{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")


async def binance_ws_stream(symbols: list, duration: float, out_file: str, dashboard: bool = True):
    """Connect to Binance WebSocket and stream prices."""
    
    # Prepare streams - using ticker streams for price updates
    streams = [f"{s.lower()}@ticker" for s in symbols]
    ws_url = f"{BINANCE_WS_URL}/{'/'.join(streams)}" if len(streams) == 1 else f"{BINANCE_WS_URL}"
    
    # For multiple symbols, use combined stream
    if len(symbols) > 1:
        stream_names = "/".join(streams)
        ws_url = f"wss://stream.binance.com:9443/stream?streams={stream_names}"
    
    trader = PaperTrader()
    dash = LiveDashboard() if dashboard else None
    reset_history()
    
    # CSV setup
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    csv_file = open(out_file, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'symbol', 'price', 'decision', 'asset_a', 'asset_b', 'cash', 'pnl'])
    
    start_time = time.time()
    prices: Dict[str, float] = {}
    
    print(f"🔗 Connecting to Binance WebSocket...")
    print(f"📡 Symbols: {', '.join(symbols)}")
    print(f"⏱️  Duration: {duration}s (0 = infinite)")
    print()
    
    try:
        async with websockets.connect(ws_url) as ws:
            print(f"✅ Connected!")
            
            while True:
                # Check duration
                if duration > 0 and (time.time() - start_time) >= duration:
                    print(f"\n⏰ Duration reached ({duration}s). Stopping...")
                    break
                
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    data = json.loads(msg)
                    
                    # Handle combined stream format
                    if 'stream' in data:
                        data = data['data']
                    
                    if 's' in data:  # Symbol present
                        symbol = data['s']
                        price = float(data['c'])  # Current price
                        prices[symbol] = price
                        
                        # Call bot decision
                        try:
                            if len(symbols) >= 2 and all(s in prices for s in symbols[:2]):
                                dec = make_decision(
                                    int(time.time()),
                                    prices[symbols[0]],
                                    prices[symbols[1]]
                                )
                            else:
                                dec = make_decision(int(time.time()), price)
                        except Exception as e:
                            print(f"⚠️ Bot error: {e}")
                            dec = None
                        
                        # Update dashboard
                        if dash and dec:
                            dash.update(symbol, price, dec, trader)
                        
                        # Log to CSV
                        if dec:
                            csv_writer.writerow([
                                int(time.time()),
                                symbol,
                                price,
                                json.dumps(dec),
                                dec.get('Asset A', ''),
                                dec.get('Asset B', ''),
                                dec.get('Cash', ''),
                                trader.get_pnl(prices)
                            ])
                            csv_file.flush()
                
                except asyncio.TimeoutError:
                    print("⚠️ Timeout - reconnecting...")
                    break
                    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 If you're in the EU, Binance might be blocked. Try Kraken instead.")
    finally:
        csv_file.close()
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"📊 SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration: {time.time() - start_time:.1f}s")
        print(f"Final PnL: ${trader.get_pnl(prices):+,.2f}")
        print(f"Log saved to: {out_file}")


async def kraken_ws_stream(pairs: list, duration: float, out_file: str):
    """
    Alternative: Kraken WebSocket for EU users.
    Kraken is fully available in France/EU.
    
    Pairs format: ["XBT/USD", "ETH/USD"]
    """
    trader = PaperTrader()
    reset_history()
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    csv_file = open(out_file, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'pair', 'price', 'decision', 'asset_a', 'asset_b', 'cash'])
    
    start_time = time.time()
    prices: Dict[str, float] = {}
    
    print(f"🔗 Connecting to Kraken WebSocket (EU-friendly)...")
    print(f"📡 Pairs: {', '.join(pairs)}")
    
    try:
        async with websockets.connect(KRAKEN_WS_URL) as ws:
            # Subscribe to ticker
            subscribe_msg = {
                "event": "subscribe",
                "pair": pairs,
                "subscription": {"name": "ticker"}
            }
            await ws.send(json.dumps(subscribe_msg))
            print("✅ Connected to Kraken!")
            
            while True:
                if duration > 0 and (time.time() - start_time) >= duration:
                    break
                
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    data = json.loads(msg)
                    
                    # Kraken ticker format: [channelID, data, "ticker", "XBT/USD"]
                    if isinstance(data, list) and len(data) >= 4:
                        ticker_data = data[1]
                        pair = data[3]
                        
                        if isinstance(ticker_data, dict) and 'c' in ticker_data:
                            price = float(ticker_data['c'][0])  # Current price
                            prices[pair] = price
                            
                            # Call bot
                            try:
                                dec = make_decision(int(time.time()), price)
                            except Exception as e:
                                dec = None
                            
                            if dec:
                                csv_writer.writerow([
                                    int(time.time()), pair, price,
                                    json.dumps(dec),
                                    dec.get('Asset A', ''),
                                    dec.get('Asset B', ''),
                                    dec.get('Cash', '')
                                ])
                                csv_file.flush()
                                
                            print(f"  {pair}: ${price:,.2f}", end='\r')
                            
                except asyncio.TimeoutError:
                    continue
                    
    except Exception as e:
        print(f"❌ Kraken error: {e}")
    finally:
        csv_file.close()
        print(f"\n📁 Log saved to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Live WebSocket connector for trading bot")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                        help='Symbols to monitor (Binance format: BTCUSDT)')
    parser.add_argument('--duration', type=float, default=300,
                        help='Duration in seconds (0 for infinite)')
    parser.add_argument('--out', default='experiments/live/ws_live_log.csv',
                        help='Output CSV file')
    parser.add_argument('--exchange', choices=['binance', 'kraken'], default='binance',
                        help='Exchange to use (kraken recommended for EU)')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable terminal dashboard')
    args = parser.parse_args()
    
    if args.exchange == 'kraken':
        # Convert symbols to Kraken format
        kraken_pairs = []
        for s in args.symbols:
            if 'BTC' in s:
                kraken_pairs.append('XBT/USD')
            elif 'ETH' in s:
                kraken_pairs.append('ETH/USD')
            else:
                kraken_pairs.append(s.replace('USDT', '/USD'))
        asyncio.run(kraken_ws_stream(kraken_pairs, args.duration, args.out))
    else:
        asyncio.run(binance_ws_stream(
            args.symbols,
            args.duration,
            args.out,
            dashboard=not args.no_dashboard
        ))


if __name__ == '__main__':
    main()
