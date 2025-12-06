"""
CLI Command Handlers.

Each handler implements logic for one CLI subcommand.
Handlers are thin wrappers that delegate to business logic modules.
"""

from argparse import Namespace
from pathlib import Path

from ..strategies import StrategyFactory
from ..data import BinanceRESTSource, DataSourceFactory
from ..engine import (
    BacktestConfig,
    BacktestEngine,
    StrategyComparator,
    PaperTradingEngine,
    DashboardDisplay,
)
from ..reporting import ReportGenerator, ChartGenerator


# =============================================================================
# BACKTEST COMMAND
# =============================================================================

def handle_backtest(args: Namespace) -> int:
    """Handle 'backtest' command."""
    print("📊 Running backtest...")
    print(f"   Data: {args.data}")
    print(f"   Strategy: {args.strategy}")
    print(f"   Capital: ${args.capital:,.2f}")
    print(f"   Fee rate: {args.fee_rate:.4f}")
    print()

    # Load data
    data_source = DataSourceFactory.from_csv(args.data)
    prices = data_source.fetch_prices()
    print(f"   Loaded {len(prices)} epochs")

    # Create strategy
    strategy = StrategyFactory.create(args.strategy)

    # Run backtest
    config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee_rate,
    )
    engine = BacktestEngine(config)
    result = engine.run(strategy, data_source)

    # Calculate benchmark
    comparator = StrategyComparator(config)
    benchmark = comparator.calculate_benchmark(data_source)

    # Print results
    print("=" * 60)
    print("📈 BACKTEST RESULTS")
    print("=" * 60)
    print(f"   Strategy: {result.strategy_name}")
    print(f"   Total Return: {result.total_return:+.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.2%}")
    print(f"   Win Rate: {result.win_rate:.1%}")
    print(f"   Trades: {result.num_trades}")
    print(f"   Final Value: ${result.final_value:,.2f}")
    print()
    print("📊 Benchmarks:")
    print(f"   Asset A (BTC): {benchmark['asset_a']:+.2%}")
    print(f"   Asset B (ETH): {benchmark['asset_b']:+.2%}")
    print(f"   50/50 B&H: {benchmark['50_50']:+.2%}")
    print()
    alpha = result.total_return - benchmark['50_50']
    print(f"🎯 Alpha vs 50/50: {alpha:+.2%}")
    print("=" * 60)

    # Generate reports if output specified
    if args.output and not args.no_plot:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate markdown report
        reporter = ReportGenerator()
        reporter.save(result, output_dir / "report.md", benchmark)
        print(f"\n📝 Report saved to: {output_dir / 'report.md'}")

        # Generate charts
        try:
            charts = ChartGenerator()
            prices_a = [p.asset_a for p in prices]
            prices_b = [p.asset_b for p in prices]

            charts.plot_performance(
                result,
                prices_a,
                prices_b,
                output_dir / "performance.png",
            )
            charts.plot_allocations(result, output_dir / "allocations.png")
            charts.plot_drawdown(result, output_dir / "drawdown.png")
            print(f"📊 Charts saved to: {output_dir}")
        except ImportError:
            print("⚠️  matplotlib not installed, skipping charts")

    return 0


# =============================================================================
# PAPER TRADING COMMAND
# =============================================================================

def handle_paper(args: Namespace) -> int:
    """Handle 'paper' command."""
    print("🚀 Starting paper trading...")
    print(f"   Strategy: {args.strategy}")
    print(f"   Capital: ${args.capital:,.2f}")
    print(f"   Symbols: {args.symbols[0]}, {args.symbols[1]}")
    print(f"   Duration: {args.duration}s")
    print(f"   Interval: {args.interval}s")
    print()

    # Create strategy
    strategy = StrategyFactory.create(args.strategy)

    # Create paper trading engine
    engine = PaperTradingEngine(
        strategy=strategy,
        initial_capital=args.capital,
        fee_rate=args.fee_rate,
        symbol_a=args.symbols[0],
        symbol_b=args.symbols[1],
    )

    # Dashboard
    dashboard = DashboardDisplay(args.symbols[0], args.symbols[1])

    def on_tick(result):
        if not args.no_dashboard:
            dashboard.display(result)

    # Run paper trading
    try:
        state = engine.run(
            duration_seconds=args.duration,
            interval_seconds=args.interval,
            on_tick=on_tick,
        )

        # Final summary
        print("\n" + "=" * 60)
        print("📊 PAPER TRADING COMPLETE")
        print("=" * 60)
        print(f"   Duration: {args.duration}s")
        print(f"   Total Trades: {len(state.trades)}")
        print(f"   Final PnL: ${state.pnl:,.2f} ({state.pnl_percent:+.2%})")
        print("=" * 60)

        # Save trade log if requested
        if args.output:
            import csv
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    "timestamp",
                    "symbol",
                    "side",
                    "quantity",
                    "price",
                    "fee",
                ]
                writer.writerow(header)
                for trade in state.trades:
                    writer.writerow([
                        trade.timestamp.isoformat(),
                        trade.symbol,
                        trade.side,
                        trade.quantity,
                        trade.price,
                        trade.fee,
                    ])
            print(f"\n📝 Trade log saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n⛔ Stopped by user")

    return 0


# =============================================================================
# FETCH DATA COMMAND
# =============================================================================

def handle_fetch(args: Namespace) -> int:
    """Handle 'fetch' command."""
    print("📥 Fetching data from Binance...")
    print(f"   Symbols: {args.symbols[0]}, {args.symbols[1]}")
    print(f"   Interval: {args.interval}")
    print(f"   Days: {args.days}")
    print()

    # Create data source and fetch
    source = BinanceRESTSource(
        symbol_a=args.symbols[0],
        symbol_b=args.symbols[1],
        interval=args.interval,
        days=args.days,
    )

    prices = source.fetch_prices()
    print(f"   ✅ Fetched {len(prices)} candles")

    if prices:
        print(f"   Date range: {prices[0].timestamp} to {prices[-1].timestamp}")
        print(f"   Price A: ${prices[0].asset_a:,.2f} → ${prices[-1].asset_a:,.2f}")
        print(f"   Price B: ${prices[0].asset_b:,.2f} → ${prices[-1].asset_b:,.2f}")

    # Save to CSV
    source.save_to_csv(args.output)
    print(f"\n📁 Saved to: {args.output}")

    return 0


# =============================================================================
# COMPARE COMMAND
# =============================================================================

def handle_compare(args: Namespace) -> int:
    """Handle 'compare' command."""
    print("📊 Comparing strategies...")
    print(f"   Data: {args.data}")
    print()

    # Load data
    data_source = DataSourceFactory.from_csv(args.data)
    prices = data_source.fetch_prices()
    print(f"   Loaded {len(prices)} epochs")

    # Get strategies to compare
    strategies = args.strategies or StrategyFactory.available()
    print(f"   Strategies: {', '.join(strategies)}")
    print()

    # Run comparison
    config = BacktestConfig(initial_capital=args.capital)
    comparator = StrategyComparator(config)

    results = comparator.compare(strategies, data_source)
    benchmark = comparator.calculate_benchmark(data_source)

    # Print results table
    print("=" * 80)
    print("📈 STRATEGY COMPARISON")
    print("=" * 80)
    header = "{:<25} {:>12} {:>10} {:>10} {:>12}".format(
        "Strategy", "Return", "Sharpe", "Max DD", "Alpha"
    )
    print(header)
    print("-" * 80)

    ranked = comparator.rank_by_return(results)
    for name, result in ranked:
        alpha = result.total_return - benchmark['50_50']
        print(
            "{:<25} {:>+11.2%} {:>10.2f} {:>10.2%} {:>+11.2%}".format(
                name,
                result.total_return,
                result.sharpe_ratio,
                result.max_drawdown,
                alpha,
            )
        )

    print("-" * 80)
    print(
        "{:<25} {:>+11.2%} {:>10} {:>10} {:>12}".format(
            "50/50 Buy&Hold",
            benchmark["50_50"],
            "N/A",
            "N/A",
            "baseline",
        )
    )
    print("{:<25} {:>+11.2%}".format("Asset A (BTC)", benchmark["asset_a"]))
    print("{:<25} {:>+11.2%}".format("Asset B (ETH)", benchmark["asset_b"]))
    print("=" * 80)

    # Winner
    if ranked:
        winner = ranked[0]
        winner_str = f"{winner[1].total_return:+.2%}"
        print("\n🏆 Best Strategy: {} with {} return".format(winner[0], winner_str))

    # Generate chart if output specified
    if args.output:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))

            for name, result in ranked[:5]:  # Top 5
                norm = [v / result.initial_capital for v in result.portfolio_values]
                ax.plot(norm, label=f'{name} ({result.total_return:+.1%})', linewidth=2)

            # Add benchmarks
            norm_a = [p.asset_a / prices[0].asset_a for p in prices]
            norm_b = [p.asset_b / prices[0].asset_b for p in prices]
            ax.plot(norm_a, label='BTC B&H', linestyle='--', alpha=0.7)
            ax.plot(norm_b, label='ETH B&H', linestyle='--', alpha=0.7)

            ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            ax.set_title('Strategy Comparison', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Growth')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150)
            plt.close()

            print(f"\n📊 Chart saved to: {args.output}")
        except ImportError:
            print("⚠️  matplotlib not installed, skipping chart")

    return 0


# =============================================================================
# REPORT COMMAND
# =============================================================================

def handle_report(args: Namespace) -> int:
    """Handle 'report' command."""
    print("📝 Generating report...")
    print(f"   Data: {args.data}")
    print(f"   Strategy: {args.strategy}")
    print(f"   Output: {args.output}")
    print()

    # Load data and run backtest
    data_source = DataSourceFactory.from_csv(args.data)
    prices = data_source.fetch_prices()

    strategy = StrategyFactory.create(args.strategy)
    config = BacktestConfig(initial_capital=args.capital)
    engine = BacktestEngine(config)
    result = engine.run(strategy, data_source)

    comparator = StrategyComparator(config)
    benchmark = comparator.calculate_benchmark(data_source)

    # Generate outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    reporter = ReportGenerator()
    reporter.save(result, output_dir / "report.md", benchmark)
    print("   ✅ report.md")

    # Charts
    try:
        charts = ChartGenerator()
        prices_a = [p.asset_a for p in prices]
        prices_b = [p.asset_b for p in prices]

        charts.plot_performance(
            result, prices_a, prices_b, output_dir / "performance.png"
        )
        print("   ✅ performance.png")

        charts.plot_allocations(result, output_dir / "allocations.png")
        print("   ✅ allocations.png")

        charts.plot_drawdown(result, output_dir / "drawdown.png")
        print("   ✅ drawdown.png")
    except ImportError:
        print("   ⚠️  matplotlib not installed, skipping charts")

    print(f"\n📁 Reports saved to: {output_dir}")

    # Print summary
    print()
    print("=" * 50)
    print("📈 SUMMARY")
    print("=" * 50)
    print(f"   Return: {result.total_return:+.2%}")
    print(f"   Sharpe: {result.sharpe_ratio:.2f}")
    print(f"   Max DD: {result.max_drawdown:.2%}")
    print(f"   Alpha: {result.total_return - benchmark['50_50']:+.2%}")
    print("=" * 50)

    return 0


# =============================================================================
# LIST COMMAND
# =============================================================================

def handle_list(args: Namespace) -> int:
    """Handle 'list' command."""
    print("📋 Available Strategies")
    print("=" * 80)

    strategies = {
        'safe_profit': 'Conservative strategy combining multiple signals. Best alpha.',
        'adaptive_trend': 'Trend-following with trailing stop and vol filter.',
        'baseline': 'Simple momentum-based strategy.',
        'sma': 'Simple Moving Average crossover strategy.',
        'composite': 'Multi-indicator: SMA + stoploss + volatility scaling.',
        'stoploss': 'Drawdown protection - exits when peak drawdown exceeded.',
        'volscale': 'Volatility scaling - reduces exposure in volatile markets.',
        'blended': 'Optuna-optimized blend of momentum + composite indicators.',
        'blended_tuned': 'Blended with walk-forward tuned parameters.',
        'blended_mo_tuned': 'Blended with multi-objective tuned parameters.',
        'blended_robust': 'Blended with robust parameters (Optuna trial 113).',
        'blended_robust_safe': 'Blended robust with max exposure cap.',
        'blended_robust_ensemble': 'Ensemble of top 5 blended configs (alias).',
        'sma_stoploss': 'SMA with drawdown-based stoploss.',
        'sma_volfilter': 'SMA with volatility filter.',
        'sma_smooth_stop': 'SMA with smoothing, partial stop, and vol scaling.',
        'adaptive_baseline': 'Baseline with volatility and stoploss gates.',
        'ensemble': 'Ensemble of top 5 blended configs for robust predictions.',
    }

    for name in StrategyFactory.available():
        desc = strategies.get(name, 'No description available.')
        print(f"\n  {name}")
        print(f"    {desc}")

    print()
    print("=" * 80)
    print("Use: tradebot backtest --strategy <name> --data <file>")

    return 0

    return 0


# =============================================================================
# TEST COMMAND
# =============================================================================

def handle_test(args: Namespace) -> int:
    """Handle 'test' command."""
    print("🔌 Testing Binance API connection...")
    print()

    try:
        source = BinanceRESTSource(
            symbol_a=args.symbols[0],
            symbol_b=args.symbols[1],
        )

        price = source.get_current_price()

        print("✅ Connection successful!")
        print()
        print(f"   {args.symbols[0]}: ${price.asset_a:,.2f}")
        print(f"   {args.symbols[1]}: ${price.asset_b:,.2f}")
        print(f"   Timestamp: {price.timestamp}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    return 0
