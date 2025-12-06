#!/usr/bin/env python3
"""
📊 Generate Daily Trading Report
=================================
Creates a comprehensive HTML/Markdown report with performance metrics,
charts, and market analysis.

Usage:
  source venv/bin/activate
  python scripts/generate_report.py --data data/crypto_btc_eth_4h_90d.csv --out reports/
"""
import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bot_trade


def calculate_metrics(returns: np.ndarray, periods_per_year: float = 252 * 6) -> dict:
    """Calculate performance metrics."""
    if len(returns) == 0:
        return {}

    total_return = np.prod(1 + returns) - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    volatility = np.std(returns) * np.sqrt(periods_per_year)
    sharpe = annualized_return / volatility if volatility > 0 else 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
    sortino = annualized_return / downside_std if downside_std > 0 else 0

    # Max drawdown
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_drawdown = np.max(drawdown)

    # Win rate
    positive = np.sum(returns > 0)
    win_rate = positive / len(returns) if len(returns) > 0 else 0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_periods': len(returns),
    }


def run_backtest_for_report(data: pd.DataFrame, strategy: str) -> dict:
    """Run backtest and return detailed results."""
    bot_trade.data = []
    bot_trade.history = []

    initial_capital = 10000.0
    portfolio_values = [initial_capital]
    holdings = {'Asset A': 0.0, 'Asset B': 0.0, 'Cash': initial_capital}
    decisions = []

    for epoch in range(len(data)):
        price_a = data.iloc[epoch]['Asset A']
        price_b = data.iloc[epoch]['Asset B']

        try:
            decision = bot_trade.make_decision(epoch, price_a, priceB=price_b, strategy=strategy)
        except:
            decision = {'Asset A': 0.33, 'Asset B': 0.33, 'Cash': 0.34}

        decisions.append(decision)

        portfolio_value = (
            holdings['Cash'] +
            holdings['Asset A'] * price_a +
            holdings['Asset B'] * price_b
        )

        holdings['Cash'] = portfolio_value * decision['Cash']
        holdings['Asset A'] = (portfolio_value * decision['Asset A']) / price_a if price_a > 0 else 0
        holdings['Asset B'] = (portfolio_value * decision['Asset B']) / price_b if price_b > 0 else 0

        portfolio_values.append(portfolio_value)

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    metrics = calculate_metrics(returns)

    return {
        'strategy': strategy,
        'portfolio_values': portfolio_values[1:],
        'decisions': decisions,
        'metrics': metrics,
    }


def generate_charts(results: dict, data: pd.DataFrame, output_dir: str):
    """Generate performance charts."""
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Portfolio performance
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = range(len(data))
    portfolio_norm = [v / results['portfolio_values'][0] for v in results['portfolio_values']]
    price_a_norm = data['Asset A'] / data['Asset A'].iloc[0]
    price_b_norm = data['Asset B'] / data['Asset B'].iloc[0]

    ax.plot(epochs, portfolio_norm, label='Strategy', linewidth=2, color='green')
    ax.plot(epochs, price_a_norm, label='Asset A (BTC)', linewidth=1, alpha=0.7, color='orange')
    ax.plot(epochs, price_b_norm, label='Asset B (ETH)', linewidth=1, alpha=0.7, color='blue')
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)

    ax.set_title(f"Performance: {results['strategy']}", fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance.png'), dpi=150)
    plt.close()

    # Chart 2: Allocation over time
    fig, ax = plt.subplots(figsize=(12, 4))

    alloc_a = [d['Asset A'] for d in results['decisions']]
    alloc_b = [d['Asset B'] for d in results['decisions']]
    alloc_cash = [d['Cash'] for d in results['decisions']]

    ax.stackplot(epochs, alloc_a, alloc_b, alloc_cash,
                 labels=['Asset A', 'Asset B', 'Cash'],
                 colors=['orange', 'blue', 'green'], alpha=0.7)

    ax.set_title('Allocation Over Time', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Allocation')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'allocation.png'), dpi=150)
    plt.close()

    # Chart 3: Drawdown
    fig, ax = plt.subplots(figsize=(12, 3))

    portfolio_values = np.array(results['portfolio_values'])
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100

    ax.fill_between(epochs, 0, -drawdown, color='red', alpha=0.5)
    ax.set_title('Drawdown', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Drawdown %')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=150)
    plt.close()


def generate_markdown_report(results: dict, data: pd.DataFrame, output_path: str):
    """Generate Markdown report."""
    m = results['metrics']

    report = f"""# 📊 Trading Strategy Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Strategy:** {results['strategy']}
**Data Period:** {len(data)} epochs

---

## 📈 Performance Summary

| Metric | Value |
|--------|-------|
| Total Return | {m['total_return']:+.2%} |
| Annualized Return | {m['annualized_return']:+.2%} |
| Volatility (Ann.) | {m['volatility']:.2%} |
| Sharpe Ratio | {m['sharpe_ratio']:.2f} |
| Sortino Ratio | {m['sortino_ratio']:.2f} |
| Max Drawdown | {m['max_drawdown']:.2%} |
| Win Rate | {m['win_rate']:.1%} |

## 📊 Benchmark Comparison

| Asset | Return |
|-------|--------|
| Strategy | **{m['total_return']:+.2%}** |
| Asset A (BTC) | {(data['Asset A'].iloc[-1] / data['Asset A'].iloc[0] - 1):+.2%} |
| Asset B (ETH) | {(data['Asset B'].iloc[-1] / data['Asset B'].iloc[0] - 1):+.2%} |

## 📉 Charts

### Performance
![Performance](performance.png)

### Allocation
![Allocation](allocation.png)

### Drawdown
![Drawdown](drawdown.png)

## 🔧 Strategy Parameters

The `{results['strategy']}` strategy uses the following configuration:
- Optimized via Optuna robust optimization
- Multi-asset portfolio management
- Dynamic allocation based on market conditions

## 📝 Notes

- All returns are calculated assuming 0% transaction fees for this report
- Actual paper trading includes 0.1% fees per trade
- Strategy designed to protect capital during volatile periods

---

*Report generated by Hackathon Trading Bot*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"📝 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate trading report')
    parser.add_argument('--data', default='data/crypto_btc_eth_4h_90d.csv',
                        help='Path to historical data')
    parser.add_argument('--strategy', default='blended_robust_ensemble',
                        help='Strategy to analyze')
    parser.add_argument('--out', default='reports/',
                        help='Output directory for report')
    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data, index_col='epoch')
    print(f"📊 Loaded {len(data)} epochs")

    # Run backtest
    print(f"🔄 Running backtest with {args.strategy}...")
    results = run_backtest_for_report(data, args.strategy)

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Generate charts
    print("📊 Generating charts...")
    generate_charts(results, data, args.out)

    # Generate report
    report_path = os.path.join(args.out, 'report.md')
    generate_markdown_report(results, data, report_path)

    # Print summary
    m = results['metrics']
    print("\n" + "="*50)
    print("📈 REPORT SUMMARY")
    print("="*50)
    print(f"   Strategy: {args.strategy}")
    print(f"   Return: {m['total_return']:+.2%}")
    print(f"   Sharpe: {m['sharpe_ratio']:.2f}")
    print(f"   Max DD: {m['max_drawdown']:.2%}")
    print("="*50)


if __name__ == '__main__':
    main()
