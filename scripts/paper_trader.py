#!/usr/bin/env python3
"""
📊 Paper Trading Simulator with Real-Time Dashboard
=====================================================
Simulates paper trading with live data and tracks performance.

Features:
- Real-time price fetching from Binance
- Paper portfolio management
- Performance metrics (PnL, Sharpe, etc.)
- Terminal dashboard with colors
- CSV logging for later analysis

Usage:
  source venv/bin/activate
  python scripts/paper_trader.py --symbols BTCUSDT ETHUSDT --initial-cash 10000 --duration 300
"""
import argparse
import time
import os
import json
import csv
import requests
from datetime import datetime
from typing import Dict, List, Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bot_trade import make_decision, reset_history


# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Portfolio:
    """Simulates a paper trading portfolio."""
    
    def __init__(self, initial_cash: float = 10000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.entry_prices: Dict[str, float] = {}
        self.trades: List[dict] = []
        self.history: List[dict] = []
        
    def get_value(self, prices: Dict[str, float]) -> float:
        """Total portfolio value."""
        value = self.cash
        for symbol, qty in self.positions.items():
            if symbol in prices:
                value += qty * prices[symbol]
        return value
    
    def get_pnl(self, prices: Dict[str, float]) -> float:
        return self.get_value(prices) - self.initial_cash
    
    def get_pnl_pct(self, prices: Dict[str, float]) -> float:
        return (self.get_pnl(prices) / self.initial_cash) * 100
    
    def rebalance(self, target_weights: Dict[str, float], prices: Dict[str, float], 
                  fee_rate: float = 0.001):
        """
        Rebalance portfolio to match target weights.
        target_weights: {'BTCUSDT': 0.3, 'ETHUSDT': 0.3, 'Cash': 0.4}
        """
        total_value = self.get_value(prices)
        
        # Calculate target values
        target_values = {}
        for symbol, weight in target_weights.items():
            if symbol != 'Cash':
                target_values[symbol] = total_value * weight
        
        # Execute trades
        for symbol, target_value in target_values.items():
            if symbol not in prices:
                continue
                
            current_qty = self.positions.get(symbol, 0)
            current_value = current_qty * prices[symbol]
            
            diff_value = target_value - current_value
            
            if abs(diff_value) > 10:  # Only trade if > $10 diff
                qty_change = diff_value / prices[symbol]
                fee = abs(diff_value) * fee_rate
                
                if diff_value > 0:  # Buy
                    cost = diff_value + fee
                    if cost <= self.cash:
                        self.positions[symbol] = current_qty + qty_change
                        self.cash -= cost
                        self.trades.append({
                            'ts': int(time.time()),
                            'symbol': symbol,
                            'side': 'BUY',
                            'qty': qty_change,
                            'price': prices[symbol],
                            'fee': fee
                        })
                else:  # Sell
                    if current_qty >= abs(qty_change):
                        self.positions[symbol] = current_qty + qty_change
                        self.cash += abs(diff_value) - fee
                        self.trades.append({
                            'ts': int(time.time()),
                            'symbol': symbol,
                            'side': 'SELL',
                            'qty': abs(qty_change),
                            'price': prices[symbol],
                            'fee': fee
                        })
        
        # Record history
        self.history.append({
            'ts': int(time.time()),
            'cash': self.cash,
            'positions': dict(self.positions),
            'value': self.get_value(prices),
            'pnl': self.get_pnl(prices)
        })


def get_binance_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices from Binance."""
    prices = {}
    for symbol in symbols:
        try:
            url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            prices[symbol] = float(r.json()['price'])
        except Exception as e:
            print(f"⚠️ Error fetching {symbol}: {e}")
    return prices


def print_dashboard(portfolio: Portfolio, prices: Dict[str, float], 
                    decision: Optional[dict], elapsed: float, duration: float):
    """Print terminal dashboard."""
    
    # Clear screen
    print('\033[2J\033[H', end='')
    
    # Header
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}🤖 PAPER TRADING DASHBOARD{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if duration > 0:
        remaining = max(0, duration - elapsed)
        print(f"⏱️  Elapsed: {elapsed:.0f}s / {duration:.0f}s ({remaining:.0f}s remaining)")
    else:
        print(f"⏱️  Elapsed: {elapsed:.0f}s (infinite)")
    print()
    
    # Prices
    print(f"{Colors.BOLD}📊 LIVE PRICES:{Colors.END}")
    for symbol, price in prices.items():
        print(f"  {symbol}: ${price:,.2f}")
    print()
    
    # Portfolio
    pnl = portfolio.get_pnl(prices)
    pnl_pct = portfolio.get_pnl_pct(prices)
    pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
    
    print(f"{Colors.BOLD}💰 PORTFOLIO:{Colors.END}")
    print(f"  Initial: ${portfolio.initial_cash:,.2f}")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    for symbol, qty in portfolio.positions.items():
        if qty > 0 and symbol in prices:
            val = qty * prices[symbol]
            print(f"  {symbol}: {qty:.6f} (${val:,.2f})")
    print(f"  {Colors.BOLD}Total: ${portfolio.get_value(prices):,.2f}{Colors.END}")
    print(f"  {Colors.BOLD}PnL: {pnl_color}${pnl:+,.2f} ({pnl_pct:+.2f}%){Colors.END}")
    print()
    
    # Last decision
    if decision:
        print(f"{Colors.BOLD}🎯 BOT DECISION:{Colors.END}")
        asset_a = decision.get('Asset A', 0)
        asset_b = decision.get('Asset B', 0)
        cash = decision.get('Cash', 0)
        print(f"  Asset A: {asset_a:.2%}")
        print(f"  Asset B: {asset_b:.2%}")
        print(f"  Cash: {cash:.2%}")
    print()
    
    # Trades
    print(f"{Colors.BOLD}📝 RECENT TRADES:{Colors.END}")
    for trade in portfolio.trades[-5:]:
        side_color = Colors.GREEN if trade['side'] == 'BUY' else Colors.RED
        print(f"  {side_color}{trade['side']}{Colors.END} {trade['symbol']}: "
              f"{trade['qty']:.6f} @ ${trade['price']:,.2f} (fee: ${trade['fee']:.2f})")
    
    if not portfolio.trades:
        print(f"  {Colors.YELLOW}No trades yet{Colors.END}")
    print()
    
    print(f"{Colors.YELLOW}Press Ctrl+C to stop{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")


def main():
    parser = argparse.ArgumentParser(description='Paper trading simulator with live data')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                        help='Symbols to trade')
    parser.add_argument('--initial-cash', type=float, default=10000,
                        help='Initial cash amount')
    parser.add_argument('--interval', type=float, default=5,
                        help='Update interval in seconds')
    parser.add_argument('--duration', type=float, default=0,
                        help='Duration in seconds (0 for infinite)')
    parser.add_argument('--fee-rate', type=float, default=0.001,
                        help='Trading fee rate (0.001 = 0.1%)')
    parser.add_argument('--out', default='experiments/live/paper_trade_log.json',
                        help='Output file for trade log')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable dashboard (log only)')
    args = parser.parse_args()
    
    # Initialize
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    reset_history()
    portfolio = Portfolio(args.initial_cash)
    
    print(f"🚀 Starting paper trading...")
    print(f"📡 Symbols: {', '.join(args.symbols)}")
    print(f"💰 Initial cash: ${args.initial_cash:,.2f}")
    print(f"⏱️  Interval: {args.interval}s")
    print()
    
    start_time = time.time()
    history = []
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            if args.duration > 0 and elapsed >= args.duration:
                print(f"\n⏰ Duration reached. Stopping...")
                break
            
            # Fetch prices
            prices = get_binance_prices(args.symbols)
            
            if not prices:
                print("⚠️ No prices available")
                time.sleep(args.interval)
                continue
            
            # Get bot decision
            try:
                if len(args.symbols) >= 2:
                    pA = prices.get(args.symbols[0], 0)
                    pB = prices.get(args.symbols[1], 0)
                    decision = make_decision(int(time.time()), pA, pB)
                else:
                    price = list(prices.values())[0]
                    decision = make_decision(int(time.time()), price)
            except Exception as e:
                print(f"⚠️ Bot error: {e}")
                decision = None
                time.sleep(args.interval)
                continue
            
            # Map decision to target weights
            if decision:
                target_weights = {}
                if 'Asset A' in decision and len(args.symbols) >= 1:
                    target_weights[args.symbols[0]] = decision['Asset A']
                if 'Asset B' in decision and len(args.symbols) >= 2:
                    target_weights[args.symbols[1]] = decision['Asset B']
                else:
                    # Single asset mode
                    target_weights[args.symbols[0]] = decision.get('Asset B', 0)
                target_weights['Cash'] = decision.get('Cash', 0)
                
                # Rebalance portfolio
                portfolio.rebalance(target_weights, prices, args.fee_rate)
            
            # Record history
            history.append({
                'ts': int(time.time()),
                'elapsed': elapsed,
                'prices': dict(prices),
                'decision': decision,
                'value': portfolio.get_value(prices),
                'pnl': portfolio.get_pnl(prices),
                'pnl_pct': portfolio.get_pnl_pct(prices)
            })
            
            # Display dashboard
            if not args.no_dashboard:
                print_dashboard(portfolio, prices, decision, elapsed, args.duration)
            else:
                pnl = portfolio.get_pnl(prices)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Value: ${portfolio.get_value(prices):,.2f} | "
                      f"PnL: ${pnl:+,.2f}")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print(f"\n\n⛔ Stopped by user")
    
    # Final summary
    final_prices = get_binance_prices(args.symbols)
    if final_prices:
        print(f"\n{'='*60}")
        print(f"📊 FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Duration: {time.time() - start_time:.1f}s")
        print(f"Initial value: ${portfolio.initial_cash:,.2f}")
        print(f"Final value: ${portfolio.get_value(final_prices):,.2f}")
        
        pnl = portfolio.get_pnl(final_prices)
        pnl_pct = portfolio.get_pnl_pct(final_prices)
        pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
        print(f"PnL: {pnl_color}${pnl:+,.2f} ({pnl_pct:+.2f}%){Colors.END}")
        print(f"Total trades: {len(portfolio.trades)}")
        
        # Calculate total fees
        total_fees = sum(t['fee'] for t in portfolio.trades)
        print(f"Total fees paid: ${total_fees:,.2f}")
    
    # Save results
    results = {
        'config': {
            'symbols': args.symbols,
            'initial_cash': args.initial_cash,
            'fee_rate': args.fee_rate,
            'duration': args.duration
        },
        'summary': {
            'start_time': start_time,
            'end_time': time.time(),
            'final_value': portfolio.get_value(final_prices) if final_prices else 0,
            'pnl': portfolio.get_pnl(final_prices) if final_prices else 0,
            'pnl_pct': portfolio.get_pnl_pct(final_prices) if final_prices else 0,
            'total_trades': len(portfolio.trades)
        },
        'trades': portfolio.trades,
        'history': history
    }
    
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {args.out}")


if __name__ == '__main__':
    main()
