#!/usr/bin/env python3
"""
📊 Analyse Comparative des Stratégies
======================================
Compare différentes stratégies sur données réelles.

Usage:
  source venv/bin/activate
  python scripts/compare_strategies.py --data data/crypto_btc_eth_4h_90d.csv
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import bot_trade


STRATEGIES = [
    'baseline',
    'sma',
    'composite',
    'blended',
    'blended_robust',
    'blended_robust_ensemble',
    'blended_robust_safe',
]


def run_single_backtest(data: pd.DataFrame, strategy: str, 
                        initial_capital: float = 10000.0) -> Dict:
    """Run backtest for a single strategy."""
    
    # Reset bot state
    bot_trade.data = []
    bot_trade.history = []
    
    portfolio_values = [initial_capital]
    holdings = {'Asset A': 0.0, 'Asset B': 0.0, 'Cash': initial_capital}
    
    for epoch in range(len(data)):
        price_a = data.iloc[epoch]['Asset A']
        price_b = data.iloc[epoch]['Asset B']
        
        # Get decision
        try:
            decision = bot_trade.make_decision(epoch, price_a, priceB=price_b, strategy=strategy)
        except Exception as e:
            # Strategy not supported, use default
            decision = {'Asset A': 0.33, 'Asset B': 0.33, 'Cash': 0.34}
        
        # Calculate portfolio value
        portfolio_value = (
            holdings['Cash'] +
            holdings['Asset A'] * price_a +
            holdings['Asset B'] * price_b
        )
        
        # Rebalance
        holdings['Cash'] = portfolio_value * decision['Cash']
        holdings['Asset A'] = (portfolio_value * decision['Asset A']) / price_a if price_a > 0 else 0
        holdings['Asset B'] = (portfolio_value * decision['Asset B']) / price_b if price_b > 0 else 0
        
        portfolio_values.append(portfolio_value)
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6) if np.std(returns) > 0 else 0
    
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    return {
        'strategy': strategy,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'portfolio_values': portfolio_values[1:],
    }


def compare_strategies(data_path: str, initial_capital: float = 10000.0) -> pd.DataFrame:
    """Compare all strategies."""
    
    # Load data
    data = pd.read_csv(data_path, index_col='epoch')
    
    print(f"📊 Comparing strategies on {len(data)} epochs")
    print(f"   Price A: ${data['Asset A'].iloc[0]:.2f} → ${data['Asset A'].iloc[-1]:.2f}")
    print(f"   Price B: ${data['Asset B'].iloc[0]:.2f} → ${data['Asset B'].iloc[-1]:.2f}")
    
    # Calculate benchmarks
    bh_a_return = (data['Asset A'].iloc[-1] - data['Asset A'].iloc[0]) / data['Asset A'].iloc[0]
    bh_b_return = (data['Asset B'].iloc[-1] - data['Asset B'].iloc[0]) / data['Asset B'].iloc[0]
    bh_5050 = 0.5 * bh_a_return + 0.5 * bh_b_return
    
    results = []
    portfolio_curves = {}
    
    for strategy in STRATEGIES:
        print(f"   Testing {strategy}...", end=' ')
        try:
            result = run_single_backtest(data, strategy, initial_capital)
            results.append(result)
            portfolio_curves[strategy] = result['portfolio_values']
            print(f"✓ Return: {result['total_return']:+.2%}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Add benchmarks
    results.append({
        'strategy': 'Buy & Hold BTC',
        'final_value': initial_capital * (1 + bh_a_return),
        'total_return': bh_a_return,
        'sharpe': 0,
        'max_drawdown': 0,
    })
    results.append({
        'strategy': 'Buy & Hold ETH',
        'final_value': initial_capital * (1 + bh_b_return),
        'total_return': bh_b_return,
        'sharpe': 0,
        'max_drawdown': 0,
    })
    results.append({
        'strategy': '50/50 B&H',
        'final_value': initial_capital * (1 + bh_5050),
        'total_return': bh_5050,
        'sharpe': 0,
        'max_drawdown': 0,
    })
    
    df = pd.DataFrame(results)
    df = df.sort_values('total_return', ascending=False)
    
    return df, portfolio_curves, data


def print_comparison(df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("📊 STRATEGY COMPARISON")
    print("="*80)
    print(f"\n{'Strategy':<25} {'Return':>12} {'Final Value':>15} {'Max DD':>12} {'Sharpe':>10}")
    print("-"*80)
    
    for _, row in df.iterrows():
        return_color = "+" if row['total_return'] >= 0 else ""
        print(f"{row['strategy']:<25} {return_color}{row['total_return']:>11.2%} ${row['final_value']:>13,.2f} {row['max_drawdown']:>11.2%} {row['sharpe']:>10.2f}")
    
    print("="*80)
    
    # Best strategy
    best = df.iloc[0]
    print(f"\n🏆 Best Strategy: {best['strategy']} with {best['total_return']:+.2%} return")


def plot_comparison(portfolio_curves: Dict, data: pd.DataFrame, save_path: str):
    """Plot comparison chart."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    epochs = range(len(data))
    
    # 1. Portfolio values normalized
    ax1 = axes[0]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(portfolio_curves)))
    
    for (name, values), color in zip(portfolio_curves.items(), colors):
        if len(values) == len(data):
            normalized = [v / values[0] for v in values]
            ax1.plot(epochs, normalized, label=name, linewidth=2, color=color)
    
    # Add benchmarks
    norm_a = data['Asset A'] / data['Asset A'].iloc[0]
    norm_b = data['Asset B'] / data['Asset B'].iloc[0]
    norm_5050 = 0.5 * norm_a + 0.5 * norm_b
    
    ax1.plot(epochs, norm_a.values, label='BTC B&H', linewidth=1, linestyle='--', alpha=0.7, color='orange')
    ax1.plot(epochs, norm_b.values, label='ETH B&H', linewidth=1, linestyle='--', alpha=0.7, color='blue')
    ax1.plot(epochs, norm_5050.values, label='50/50 B&H', linewidth=2, linestyle='--', color='gray')
    
    ax1.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('Strategy Performance Comparison (Normalized)', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Growth (1.0 = initial)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar chart of returns
    ax2 = axes[1]
    
    strategies = list(portfolio_curves.keys()) + ['BTC B&H', 'ETH B&H', '50/50 B&H']
    returns = []
    
    for name, values in portfolio_curves.items():
        ret = (values[-1] - values[0]) / values[0]
        returns.append(ret)
    
    returns.extend([
        (data['Asset A'].iloc[-1] / data['Asset A'].iloc[0]) - 1,
        (data['Asset B'].iloc[-1] / data['Asset B'].iloc[0]) - 1,
        0.5 * ((data['Asset A'].iloc[-1] / data['Asset A'].iloc[0]) - 1) + 
        0.5 * ((data['Asset B'].iloc[-1] / data['Asset B'].iloc[0]) - 1),
    ])
    
    colors_bar = ['green' if r >= 0 else 'red' for r in returns]
    
    bars = ax2.barh(strategies, [r * 100 for r in returns], color=colors_bar, alpha=0.7)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('Return (%)')
    ax2.set_title('Total Return by Strategy', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add labels
    for bar, ret in zip(bars, returns):
        width = bar.get_width()
        ax2.annotate(f'{ret:+.1%}',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5 if width >= 0 else -5, 0),
                    textcoords='offset points',
                    ha='left' if width >= 0 else 'right',
                    va='center',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison chart saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare trading strategies')
    parser.add_argument('--data', default='data/crypto_btc_eth_4h_90d.csv',
                        help='Path to historical data CSV')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital in USD')
    parser.add_argument('--plot', default='experiments/strategy_comparison.png',
                        help='Path to save comparison chart')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.plot) if os.path.dirname(args.plot) else '.', exist_ok=True)
    
    df, portfolio_curves, data = compare_strategies(args.data, args.capital)
    print_comparison(df)
    
    if args.plot:
        plot_comparison(portfolio_curves, data, args.plot)


if __name__ == '__main__':
    main()
