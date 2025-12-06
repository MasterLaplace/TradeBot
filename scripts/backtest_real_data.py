#!/usr/bin/env python3
"""
🧪 Backtest sur Données Réelles
================================
Teste la stratégie bot_trade sur des données historiques réelles.

Usage:
  source venv/bin/activate
  python scripts/backtest_real_data.py --data data/crypto_btc_eth_1h_30d.csv
"""
import argparse
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime

# Import bot
import bot_trade


def run_backtest(data_path: str, initial_capital: float = 10000.0,
                 strategy: str = None) -> Dict:
    """
    Run backtest on historical data.
    
    Args:
        data_path: Path to CSV with columns [Asset A, Asset B, Cash]
        initial_capital: Starting capital in USD
        strategy: Strategy name (None = use default)
    
    Returns:
        Dictionary with backtest results
    """
    # Load data
    df = pd.read_csv(data_path, index_col='epoch')
    
    print(f"📊 Loaded {len(df)} epochs from {data_path}")
    print(f"   Price A range: ${df['Asset A'].min():.2f} - ${df['Asset A'].max():.2f}")
    print(f"   Price B range: ${df['Asset B'].min():.2f} - ${df['Asset B'].max():.2f}")
    
    # Reset bot state
    bot_trade.data = []
    bot_trade.history = []
    
    # Strategy to use
    current_strategy = strategy or 'blended_robust_ensemble'
    
    # Track portfolio
    portfolio_values = []
    decisions = []
    
    # Current holdings
    holdings = {
        'Asset A': 0.0,
        'Asset B': 0.0,
        'Cash': initial_capital
    }
    
    print(f"\n🚀 Starting backtest with ${initial_capital:,.2f}...")
    print(f"   Strategy: {current_strategy}")
    
    for epoch in range(len(df)):
        price_a = df.iloc[epoch]['Asset A']
        price_b = df.iloc[epoch]['Asset B']
        
        # Get decision from bot
        decision = bot_trade.make_decision(epoch, price_a, priceB=price_b, strategy=current_strategy)
        decisions.append(decision)
        
        # Calculate current portfolio value
        portfolio_value = (
            holdings['Cash'] +
            holdings['Asset A'] * price_a +
            holdings['Asset B'] * price_b
        )
        
        # Calculate target holdings
        target_cash = portfolio_value * decision['Cash']
        target_a_value = portfolio_value * decision['Asset A']
        target_b_value = portfolio_value * decision['Asset B']
        
        # Rebalance (simplified - no transaction costs yet)
        holdings['Cash'] = target_cash
        holdings['Asset A'] = target_a_value / price_a if price_a > 0 else 0
        holdings['Asset B'] = target_b_value / price_b if price_b > 0 else 0
        
        portfolio_values.append(portfolio_value)
        
        # Progress every 100 epochs
        if epoch % 100 == 0:
            print(f"   Epoch {epoch}: ${portfolio_value:,.2f} "
                  f"(A: {decision['Asset A']:.1%}, B: {decision['Asset B']:.1%}, Cash: {decision['Cash']:.1%})")
    
    # Final portfolio value
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # Calculate buy & hold benchmarks
    initial_a = df.iloc[0]['Asset A']
    final_a = df.iloc[-1]['Asset A']
    bh_return_a = (final_a - initial_a) / initial_a
    
    initial_b = df.iloc[0]['Asset B']
    final_b = df.iloc[-1]['Asset B']
    bh_return_b = (final_b - initial_b) / initial_b
    
    # 50/50 buy & hold
    bh_5050_return = 0.5 * bh_return_a + 0.5 * bh_return_b
    
    # Calculate metrics
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24) if np.std(returns) > 0 else 0  # Hourly to annualized
    
    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    results = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'epochs': len(df),
        'benchmark_a_return': bh_return_a,
        'benchmark_b_return': bh_return_b,
        'benchmark_5050_return': bh_5050_return,
        'portfolio_values': portfolio_values,
        'decisions': decisions,
        'prices_a': df['Asset A'].tolist(),
        'prices_b': df['Asset B'].tolist(),
    }
    
    return results


def print_results(results: Dict):
    """Print formatted backtest results."""
    print("\n" + "="*60)
    print("📈 BACKTEST RESULTS")
    print("="*60)
    
    print(f"\n💰 Performance:")
    print(f"   Initial Capital:  ${results['initial_capital']:,.2f}")
    print(f"   Final Value:      ${results['final_value']:,.2f}")
    print(f"   Total Return:     {results['total_return']:+.2%}")
    print(f"   Sharpe Ratio:     {results['sharpe_ratio']:.2f} (annualized)")
    print(f"   Max Drawdown:     {results['max_drawdown']:.2%}")
    
    print(f"\n📊 Benchmarks (Buy & Hold):")
    print(f"   Asset A (BTC):    {results['benchmark_a_return']:+.2%}")
    print(f"   Asset B (ETH):    {results['benchmark_b_return']:+.2%}")
    print(f"   50/50 Portfolio:  {results['benchmark_5050_return']:+.2%}")
    
    # Compare to benchmarks
    vs_a = results['total_return'] - results['benchmark_a_return']
    vs_b = results['total_return'] - results['benchmark_b_return']
    vs_5050 = results['total_return'] - results['benchmark_5050_return']
    
    print(f"\n🆚 Strategy vs Benchmarks:")
    print(f"   vs Asset A:       {vs_a:+.2%}")
    print(f"   vs Asset B:       {vs_b:+.2%}")
    print(f"   vs 50/50:         {vs_5050:+.2%}")
    
    if results['total_return'] > results['benchmark_5050_return']:
        print("\n✅ Strategy OUTPERFORMED the 50/50 benchmark!")
    else:
        print("\n⚠️ Strategy underperformed the 50/50 benchmark")
    
    print("="*60)


def plot_results(results: Dict, save_path: str = None):
    """Generate performance visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    epochs = range(len(results['portfolio_values']))
    
    # 1. Portfolio value vs benchmarks
    ax1 = axes[0]
    
    # Normalize to initial capital
    initial = results['initial_capital']
    norm_portfolio = [v / initial for v in results['portfolio_values']]
    norm_a = [p / results['prices_a'][0] for p in results['prices_a']]
    norm_b = [p / results['prices_b'][0] for p in results['prices_b']]
    norm_5050 = [0.5 * a + 0.5 * b for a, b in zip(norm_a, norm_b)]
    
    ax1.plot(epochs, norm_portfolio, label='Strategy', linewidth=2, color='green')
    ax1.plot(epochs, norm_a, label='Asset A (BTC)', linewidth=1, alpha=0.7, color='orange')
    ax1.plot(epochs, norm_b, label='Asset B (ETH)', linewidth=1, alpha=0.7, color='blue')
    ax1.plot(epochs, norm_5050, label='50/50 B&H', linewidth=1.5, linestyle='--', color='gray')
    
    ax1.set_title('Portfolio Performance vs Benchmarks (Normalized)', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Growth (1.0 = initial)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    
    # 2. Allocations over time
    ax2 = axes[1]
    
    alloc_a = [d['Asset A'] for d in results['decisions']]
    alloc_b = [d['Asset B'] for d in results['decisions']]
    alloc_cash = [d['Cash'] for d in results['decisions']]
    
    ax2.stackplot(epochs, alloc_a, alloc_b, alloc_cash, 
                  labels=['Asset A', 'Asset B', 'Cash'],
                  colors=['orange', 'blue', 'green'], alpha=0.7)
    
    ax2.set_title('Allocation Over Time', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Allocation %')
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[2]
    
    portfolio_values = np.array(results['portfolio_values'])
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    
    ax3.fill_between(epochs, 0, -drawdown * 100, color='red', alpha=0.5)
    ax3.set_title('Drawdown', fontsize=12)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Drawdown %')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Chart saved to: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Backtest on real data')
    parser.add_argument('--data', default='data/crypto_btc_eth_1h_30d.csv',
                        help='Path to historical data CSV')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital in USD')
    parser.add_argument('--strategy', default=None,
                        help='Strategy name to test')
    parser.add_argument('--plot', default='experiments/backtest_results.png',
                        help='Path to save performance chart')
    args = parser.parse_args()
    
    # Create output directory
    if args.plot:
        os.makedirs(os.path.dirname(args.plot), exist_ok=True)
    
    # Run backtest
    results = run_backtest(args.data, args.capital, args.strategy)
    
    # Print results
    print_results(results)
    
    # Plot results
    if args.plot:
        plot_results(results, args.plot)


if __name__ == '__main__':
    main()
