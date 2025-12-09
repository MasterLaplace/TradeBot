"""
CLI Command Handlers.

Each handler implements logic for one CLI subcommand.
Handlers are thin wrappers that delegate to business logic modules.
"""

from argparse import Namespace
from pathlib import Path
import csv
import os

from ..strategies import StrategyFactory
from .command_base import CLICommand
from ..engine import (
    BacktestConfig,
    StrategyComparator,
    PaperTradingEngine,
    DashboardDisplay,
)
from ..tools.parallel_backtest import (
    run_parallel_backtests,
    run_parallel_paper,
    write_summary_csv,
    write_summary_csv_live,
)
from ..reporting import ReportGenerator
import matplotlib.pyplot as plt
import glob


COMMAND_REGISTRY = {}

def register_command(name: str, cls):
    """Register a command class in the registry."""
    COMMAND_REGISTRY[name] = cls


def get_command(name: str):
    """Return an instance of the command registered under *name*.

    If the command is unknown returns None.
    """
    cls = COMMAND_REGISTRY.get(name)
    return cls() if cls else None


# =============================================================================
# BACKTEST COMMAND
# =============================================================================

class BacktestCommand(CLICommand):
    """Command: run a backtest on historical data."""

    def execute(self, args: Namespace) -> int:
        """Execute a backtest for the requested strategy and dataset.

        The method is intentionally small; it delegates loading, engine
        execution and report generation to the base class helpers.
        """
        print("📊 Running backtest...")
        print(f"   Data: {args.data}")
        print(f"   Strategy: {args.strategy}")
        print(f"   Capital: ${args.capital:,.2f}")
        print(f"   Fee rate: {args.fee_rate:.4f}")
        print()

        data_source, prices = self.load_prices_from_csv(args.data)
        print(f"   Loaded {len(prices)} epochs")

        result = self.run_backtest(args.strategy, data_source, args.capital, args.fee_rate)
        benchmark = self.build_benchmark(data_source, args.capital)

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

        if args.output and not getattr(args, 'no_plot', False):
            try:
                self.save_report_and_charts(result, prices, data_source, args.output)
                print(f"\n📝 Report and charts saved to: {args.output}")
            except ImportError:
                self.ensure_output_dir(args.output)
                reporter = ReportGenerator()
                reporter.save(result, Path(args.output) / "report.md", benchmark)
                print("⚠️  matplotlib not installed, saved markdown report only")

        return 0


def handle_backtest(args: Namespace) -> int:
    """Backwards-compatible wrapper for functional API."""
    return BacktestCommand().execute(args)
register_command('backtest', BacktestCommand)


# =============================================================================
# PAPER TRADING COMMAND
# =============================================================================

class PaperCommand(CLICommand):
    """Command: run a paper trading session with live prices."""

    def execute(self, args: Namespace) -> int:
        """Run a paper trading session using live price data.

        This method builds the strategy and engine then runs the simulation
        for the requested duration and optionally saves trade logs.
        """
        print("🚀 Starting paper trading...")
        print(f"   Strategy: {args.strategy}")
        print(f"   Capital: ${args.capital:,.2f}")
        print(f"   Symbols: {args.symbols[0]}, {args.symbols[1]}")
        print(f"   Duration: {args.duration}s")
        print(f"   Interval: {args.interval}s")
        print()

        strategy = StrategyFactory.create(args.strategy)
        engine = PaperTradingEngine(
            strategy=strategy,
            initial_capital=args.capital,
            fee_rate=args.fee_rate,
            symbol_a=args.symbols[0],
            symbol_b=args.symbols[1],
        )
        dashboard = DashboardDisplay(args.symbols[0], args.symbols[1])

        def on_tick(result):
            if not getattr(args, 'no_dashboard', False):
                dashboard.display(result)

        try:
            state = engine.run(
                duration_seconds=args.duration,
                interval_seconds=args.interval,
                on_tick=on_tick,
            )

            print("\n" + "=" * 60)
            print("📊 PAPER TRADING COMPLETE")
            print("=" * 60)
            print(f"   Duration: {args.duration}s")
            print(f"   Total Trades: {len(state.trades)}")
            print(f"   Final PnL: ${state.pnl:,.2f} ({state.pnl_percent:+.2%})")
            print("=" * 60)

            if args.output:
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


def handle_paper(args: Namespace) -> int:
    return PaperCommand().execute(args)
register_command('paper', PaperCommand)


# =============================================================================
# FETCH DATA COMMAND
# =============================================================================

class FetchCommand(CLICommand):
    """Command: fetch historical data from Binance and save to CSV."""

    def execute(self, args: Namespace) -> int:
        """Fetch historical data from Binance REST API and save to CSV."""
        print("📥 Fetching data from Binance...")
        print(f"   Symbols: {args.symbols[0]}, {args.symbols[1]}")
        print(f"   Interval: {args.interval}")
        print(f"   Days: {args.days}")
        print()

        # BinanceRESTSource is a top-level import
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

        source.save_to_csv(args.output)
        print(f"\n📁 Saved to: {args.output}")

        return 0


def handle_fetch(args: Namespace) -> int:
    return FetchCommand().execute(args)
register_command('fetch', FetchCommand)


# =============================================================================
# COMPARE COMMAND
# =============================================================================

class CompareCommand(CLICommand):
    """Command: compare multiple strategies on the same dataset."""

    def execute(self, args: Namespace) -> int:
        """Compare multiple strategies on a single dataset and print a ranking."""
        print("📊 Comparing strategies...")
        print(f"   Data: {args.data}")
        print()
        data_source, prices = self.load_prices_from_csv(args.data)
        print(f"   Loaded {len(prices)} epochs")

        strategies = args.strategies or StrategyFactory.available()
        print(f"   Strategies: {', '.join(strategies)}")
        print()

        # Backtest engine and comparator are top-level imports
        # BacktestConfig, StrategyComparator imported at the module level
        config = BacktestConfig(initial_capital=args.capital)
        comparator = StrategyComparator(config)

        results = comparator.compare(strategies, data_source)
        benchmark = comparator.calculate_benchmark(data_source)

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

        if ranked:
            winner = ranked[0]
            winner_str = f"{winner[1].total_return:+.2%}"
            print("\n🏆 Best Strategy: {} with {} return".format(winner[0], winner_str))

        if args.output:
            try:

                fig, ax = plt.subplots(figsize=(12, 6))

                for name, result in ranked[:5]:  # Top 5
                    norm = [v / result.initial_capital for v in result.portfolio_values]
                    ax.plot(norm, label=f'{name} ({result.total_return:+.1%})', linewidth=2)

            
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
            except Exception as e:
                print(f"⚠️  Chart generation failed: {e}")

        return 0


def handle_compare(args: Namespace) -> int:
    return CompareCommand().execute(args)
register_command('compare', CompareCommand)


# =============================================================================
# REPORT COMMAND
# =============================================================================

class ReportCommand(CLICommand):
    """Command: produce a markdown report and charts for a single strategy."""

    def execute(self, args: Namespace) -> int:
        """Generate a report for a single strategy and dataset.

        The function delegates the data loading and backtest run to shared
        helpers and then saves the report and charts.
        """
        print("📝 Generating report...")
        print(f"   Data: {args.data}")
        print(f"   Strategy: {args.strategy}")
        print(f"   Output: {args.output}")
        print()

        data_source, prices = self.load_prices_from_csv(args.data)
        result = self.run_backtest(args.strategy, data_source, args.capital, args.fee_rate)
        benchmark = self.build_benchmark(data_source, args.capital)

        try:
            self.save_report_and_charts(result, prices, data_source, args.output)
            print(f"\n📁 Reports saved to: {args.output}")
        except ImportError:
            self.ensure_output_dir(args.output)
            reporter = ReportGenerator()
            reporter.save(result, Path(args.output) / "report.md", benchmark)
            print("   ⚠️  matplotlib not installed, saved markdown report only")

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


def handle_report(args: Namespace) -> int:
    return ReportCommand().execute(args)
register_command('report', ReportCommand)


# =============================================================================
# LIST COMMAND
# =============================================================================

class ListCommand(CLICommand):
    """Command: list available strategies."""

    def execute(self, args: Namespace) -> int:
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


def handle_list(args: Namespace) -> int:
    return ListCommand().execute(args)
register_command('list', ListCommand)


# =============================================================================
# TEST COMMAND
# =============================================================================

class TestCommand(CLICommand):
    """Command: test Binance REST API connectivity."""

    def execute(self, args: Namespace) -> int:
        """Test Binance REST connectivity and print current prices."""
        print("🔌 Testing Binance API connection...")
        print()

        skip_api_fail = getattr(args, 'skip_api_fail', False) or (
            os.environ.get('SKIP_API_TESTS', '').lower() in ('1', 'true', 'yes')
        )

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
            if skip_api_fail:
                print("⚠️  Ignoring API failure due to `--skip-api-fail` or SKIP_API_TESTS=1")
                return 0
            return 1

        return 0


def handle_test(args: Namespace) -> int:
    return TestCommand().execute(args)
register_command('test', TestCommand)


# =============================================================================
# PARALLEL BACKTEST COMMAND
# =============================================================================


class ParallelCommand(CLICommand):
    """Command: parallel runner for backtest or paper sessions."""

    def execute(self, args: Namespace) -> int:
        """Handle 'parallel' command to run strategies concurrently across datasets.

        Supports dataset globbing, optional per-strategy CSV outputs, and backtest params.
        """
        mode = getattr(args, 'mode', 'backtest')
        datasets = args.datasets or []
        strategies = args.strategies or StrategyFactory.available()
        workers = args.workers
        output = args.output
        initial_capital = getattr(args, 'capital', 10000.0)
        fee_rate = getattr(args, 'fee_rate', 0.001)
        per_strategy_dir = getattr(args, 'per_strategy_dir', None)
        append = getattr(args, 'append', False)
        show_progress = not getattr(args, 'no_progress', False)

        if mode == 'backtest':
            expanded_datasets = []
            for ds in datasets:
                matches = glob.glob(ds)
                expanded_datasets.extend(matches if matches else [ds])

            datasets = sorted(set(expanded_datasets))

            if not datasets:
                print("⚠️  No datasets matched the provided patterns for backtest mode")
                return 1

        bad = [s for s in strategies if s not in StrategyFactory.available()]
        if bad:
            print(f"⚠️  Unknown strategies: {', '.join(bad)}")
            print(f"Available: {', '.join(StrategyFactory.available())}")
            return 1

        print(f"   Strategies: {', '.join(strategies)}")
        if mode == 'backtest':
            print(f"   Datasets: {', '.join(datasets)}")
            print(f"🔁 Running parallel backtests (historical)")
        else:
            symbols = getattr(args, 'symbols', ['BTCUSDT', 'ETHUSDT'])
            print(f"   Symbols: {symbols[0]}, {symbols[1]}")
            print(f"🔁 Running parallel paper sessions (live simulated)")
        print(f"   Workers: {workers or 'auto'}")

        if mode == 'backtest':
            results = run_parallel_backtests(
                strategies,
                datasets,
                workers=workers,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                per_strategy_dir=per_strategy_dir,
                show_progress=show_progress,
            )
            write_summary_csv(results, output, append=append)
            print(f"📊 Parallel backtest summary written to: {output}")
        else:
            symbols = getattr(args, 'symbols', ['BTCUSDT', 'ETHUSDT'])
            duration = getattr(args, 'duration', 120)
            interval = getattr(args, 'interval', 5.0)
            # run_parallel_paper, write_summary_csv_live are top-level imports

            use_ws = getattr(args, 'use_ws', False)
            results = run_parallel_paper(
                strategies,
                symbols,
                workers=workers,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                duration_seconds=duration,
                interval_seconds=interval,
                per_strategy_dir=per_strategy_dir,
                show_progress=show_progress,
                use_ws=use_ws,
            )
            write_summary_csv_live(results, output, append=append)

        return 0
        print(f"📊 Parallel paper summary written to: {output}")
        return 0


def handle_parallel(args: Namespace) -> int:
    return ParallelCommand().execute(args)
register_command('parallel', ParallelCommand)
