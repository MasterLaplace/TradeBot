"""
TradeBot CLI - Unified Command Line Interface.

This is the single entry point for all trading operations.

Usage:
    python -m src.cli --help
    python -m src.cli backtest --data data/crypto.csv --strategy safe_profit
    python -m src.cli paper --duration 3600 --interval 60
    python -m src.cli fetch --days 30 --output data/btc_eth_30d.csv
    python -m src.cli compare --data data/crypto.csv
    python -m src.cli report --data data/crypto.csv --output reports/

Design:
    - Single entry point for all functionality
    - Subcommands for different operations
    - Comprehensive --help at every level
    - Clean separation between CLI and business logic
"""

import argparse
import sys
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with all subcommands.

    Returns a fully configured ArgumentParser.
    """

    # ==========================================================================
    # MAIN PARSER
    # ==========================================================================

    parser = argparse.ArgumentParser(
        prog='tradebot',
        description="""
╔══════════════════════════════════════════════════════════════════╗
║                    🤖 TradeBot v2.0                              ║
║           Unified Trading Bot Command Line Interface             ║
╠══════════════════════════════════════════════════════════════════╣
║  A clean, SOLID architecture trading application providing:      ║
║  • Multiple trading strategies (safe_profit, adaptive, etc.)     ║
║  • Backtesting on historical data                                ║
║  • Paper trading with live Binance prices                        ║
║  • Data fetching from Binance API                                ║
║  • Strategy comparison and reporting                             ║
╚══════════════════════════════════════════════════════════════════╝
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s backtest --data data/crypto.csv --strategy safe_profit
  %(prog)s paper --duration 3600 --interval 60
  %(prog)s fetch --days 30 --output data/btc_eth.csv
  %(prog)s compare --data data/crypto.csv
  %(prog)s report --data data/crypto.csv --output reports/

For more help on a specific command:
  %(prog)s <command> --help
        """
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version='TradeBot v2.0.0'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    # ==========================================================================
    # SUBCOMMANDS
    # ==========================================================================

    subparsers = parser.add_subparsers(
        dest='command',
        title='Commands',
        description='Available commands (use <command> --help for details)',
        metavar='<command>'
    )

    # --------------------------------------------------------------------------
    # BACKTEST COMMAND
    # --------------------------------------------------------------------------

    backtest_parser = subparsers.add_parser(
        'backtest',
        help='Run backtest on historical data',
        description="""
Run a trading strategy backtest on historical price data.

This simulates how the strategy would have performed historically,
calculating metrics like return, Sharpe ratio, and max drawdown.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data data/crypto.csv
  %(prog)s --data data/crypto.csv --strategy adaptive_trend
  %(prog)s --data data/crypto.csv --capital 50000 --fee-rate 0.001
        """
    )

    backtest_parser.add_argument(
        '-d', '--data',
        required=True,
        metavar='FILE',
        help='Path to CSV file with price data'
    )

    backtest_parser.add_argument(
        '-s', '--strategy',
        default='safe_profit',
        metavar='NAME',
        help='Strategy to test (default: safe_profit). Use `tradebot list`.'
    )

    backtest_parser.add_argument(
        '-c', '--capital',
        type=float,
        default=10000.0,
        metavar='USD',
        help='Initial capital in USD (default: 10000)'
    )

    backtest_parser.add_argument(
        '-f', '--fee-rate',
        type=float,
        default=0.001,
        metavar='RATE',
        help='Transaction fee rate (default: 0.001 = 0.1%%)'
    )

    backtest_parser.add_argument(
        '-o', '--output',
        metavar='DIR',
        help='Output directory for reports (optional)'
    )

    backtest_parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable chart generation'
    )

    # --------------------------------------------------------------------------
    # PAPER TRADING COMMAND
    # --------------------------------------------------------------------------

    paper_parser = subparsers.add_parser(
        'paper',
        help='Run paper trading with live prices',
        description="""
Run paper (simulated) trading using real-time prices from Binance.

This tests the strategy with live market data without risking real money.
Displays a real-time dashboard with portfolio value and trades.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --duration 3600 --interval 60
  %(prog)s --symbols BTCUSDT ETHUSDT --capital 10000
  %(prog)s --strategy adaptive_trend --duration 7200
        """
    )

    paper_parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        metavar='SECS',
        help='Duration in seconds (default: 3600 = 1 hour)'
    )

    paper_parser.add_argument(
        '--interval',
        type=float,
        default=60.0,
        metavar='SECS',
        help='Seconds between ticks (default: 60)'
    )

    paper_parser.add_argument(
        '-s', '--strategy',
        default='safe_profit',
        metavar='NAME',
        help='Strategy to use (default: safe_profit)'
    )

    paper_parser.add_argument(
        '-c', '--capital',
        type=float,
        default=10000.0,
        metavar='USD',
        help='Initial capital in USD (default: 10000)'
    )

    paper_parser.add_argument(
        '--symbols',
        nargs=2,
        default=['BTCUSDT', 'ETHUSDT'],
        metavar=('SYM_A', 'SYM_B'),
        help='Trading symbols (default: BTCUSDT ETHUSDT)'
    )

    paper_parser.add_argument(
        '-f', '--fee-rate',
        type=float,
        default=0.001,
        metavar='RATE',
        help='Transaction fee rate (default: 0.001)'
    )

    paper_parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Disable live dashboard display'
    )

    paper_parser.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Output CSV file for trade log'
    )

    # --------------------------------------------------------------------------
    # FETCH DATA COMMAND
    # --------------------------------------------------------------------------

    fetch_parser = subparsers.add_parser(
        'fetch',
        help='Fetch historical data from Binance',
        description="""
Download historical price data from Binance API.

Saves data in CSV format compatible with backtesting.
No API key required (uses public endpoints).
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --days 30 --output data/btc_eth_30d.csv
  %(prog)s --days 90 --interval 4h --symbols BTCUSDT ETHUSDT
  %(prog)s --days 7 --interval 1h --output data/week.csv
        """
    )

    fetch_parser.add_argument(
        '--days',
        type=int,
        default=30,
        metavar='N',
        help='Number of days of history (default: 30)'
    )

    fetch_parser.add_argument(
        '--interval',
        default='4h',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        help='Candle interval (default: 4h)'
    )

    fetch_parser.add_argument(
        '--symbols',
        nargs=2,
        default=['BTCUSDT', 'ETHUSDT'],
        metavar=('SYM_A', 'SYM_B'),
        help='Trading symbols (default: BTCUSDT ETHUSDT)'
    )

    fetch_parser.add_argument(
        '-o', '--output',
        required=True,
        metavar='FILE',
        help='Output CSV file path'
    )

    # --------------------------------------------------------------------------
    # COMPARE COMMAND
    # --------------------------------------------------------------------------

    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare multiple strategies',
        description="""
Compare performance of multiple trading strategies on the same data.

Runs backtests for all specified strategies and displays a ranking.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data data/crypto.csv
  %(prog)s --data data/crypto.csv --strategies safe_profit adaptive_trend baseline
  %(prog)s --data data/crypto.csv --output comparison.png
        """
    )

    compare_parser.add_argument(
        '-d', '--data',
        required=True,
        metavar='FILE',
        help='Path to CSV file with price data'
    )

    compare_parser.add_argument(
        '--strategies',
        nargs='+',
        metavar='NAME',
        help='Strategies to compare (default: all available)'
    )

    compare_parser.add_argument(
        '-c', '--capital',
        type=float,
        default=10000.0,
        metavar='USD',
        help='Initial capital in USD (default: 10000)'
    )

    compare_parser.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Output file for comparison chart'
    )

    # --------------------------------------------------------------------------
    # REPORT COMMAND
    # --------------------------------------------------------------------------

    report_parser = subparsers.add_parser(
        'report',
        help='Generate performance report',
        description="""
Generate a comprehensive performance report with charts.

Creates markdown report and PNG charts for analysis.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data data/crypto.csv --output reports/
  %(prog)s --data data/crypto.csv --strategy safe_profit --output reports/
        """
    )

    report_parser.add_argument(
        '-d', '--data',
        required=True,
        metavar='FILE',
        help='Path to CSV file with price data'
    )

    report_parser.add_argument(
        '-s', '--strategy',
        default='safe_profit',
        metavar='NAME',
        help='Strategy to report on (default: safe_profit)'
    )

    report_parser.add_argument(
        '-o', '--output',
        required=True,
        metavar='DIR',
        help='Output directory for reports'
    )

    report_parser.add_argument(
        '-c', '--capital',
        type=float,
        default=10000.0,
        metavar='USD',
        help='Initial capital in USD (default: 10000)'
    )

    # --------------------------------------------------------------------------
    # LIST STRATEGIES COMMAND
    # --------------------------------------------------------------------------

    subparsers.add_parser(
        'list',
        help='List available strategies',
        description='Display all available trading strategies with descriptions.'
    )

    # --------------------------------------------------------------------------
    # TEST CONNECTION COMMAND
    # --------------------------------------------------------------------------

    test_parser = subparsers.add_parser(
        'test',
        help='Test Binance API connection',
        description='Test connectivity to Binance API and display current prices.'
    )

    test_parser.add_argument(
        '--symbols',
        nargs=2,
        default=['BTCUSDT', 'ETHUSDT'],
        metavar=('SYM_A', 'SYM_B'),
        help='Symbols to test (default: BTCUSDT ETHUSDT)'
    )
    test_parser.add_argument(
        '--skip-api-fail',
        action='store_true',
        default=False,
        help='Treat API connection failures as warnings (exit code 0)'
    )

    # --------------------------------------------------------------------------
    # PARALLEL BACKTEST COMMAND
    # --------------------------------------------------------------------------

    parallel_parser = subparsers.add_parser(
        'parallel',
        help='Run parallel backtests for multiple strategies',
        description='Execute backtests in parallel across strategies and datasets.'
    )

    parallel_parser.add_argument(
        '--datasets', '-d', nargs='+', required=False, help='Dataset CSV files (required for backtest mode). Supports glob patterns.'
    )
    parallel_parser.add_argument(
        '--strategies', '-s', nargs='*', help='Strategy names to run (default: all available)'
    )
    parallel_parser.add_argument(
        '--workers', '-w', type=int, default=None, help='Number of parallel workers (default: cpu_count)'
    )
    parallel_parser.add_argument(
        '--mode', choices=['backtest', 'paper'], default='backtest', help='Mode: backtest (default) or paper (live simulated trading)'
    )
    parallel_parser.add_argument(
        '--symbols', nargs=2, default=['BTCUSDT', 'ETHUSDT'], metavar=('SYM_A', 'SYM_B'), help='Symbols for paper mode (default: BTCUSDT ETHUSDT)'
    )
    parallel_parser.add_argument(
        '--duration', type=int, default=120, help='Duration in seconds for paper mode per run (default: 120)'
    )
    parallel_parser.add_argument(
        '--interval', type=float, default=5.0, help='Seconds between price ticks for paper mode (default: 5.0)'
    )
    parallel_parser.add_argument(
        '--use-ws', action='store_true', default=False, help='Use WebSocket-based centralized broadcaster for paper mode (reduces REST calls)'
    )
    parallel_parser.add_argument(
        '--capital', '-c', type=float, default=10000.0, help='Initial capital for backtests (default: 10000)'
    )
    parallel_parser.add_argument(
        '--fee-rate', type=float, default=0.001, help='Fee rate for backtests (default: 0.001)'
    )
    parallel_parser.add_argument(
        '--per-strategy-dir', '-p', help='Optional directory to write per-strategy CSVs (one file per strategy)'
    )
    parallel_parser.add_argument(
        '--append', action='store_true', default=False, help='Append to existing summary CSV instead of overwriting'
    )
    parallel_parser.add_argument(
        '--no-progress', action='store_true', default=False, help='Disable progress bar during execution'
    )
    parallel_parser.add_argument(
        '--output', '-o', default='outputs/parallel_backtest_summary.csv', help='Output CSV summary path'
    )

    return parser


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        args: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 = success, 1 = error)
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 0

    # Import handlers lazily to speed up --help
    from .commands import (
        handle_backtest,
        handle_paper,
        handle_fetch,
        handle_compare,
        handle_report,
        handle_list,
        handle_test,
        handle_parallel,
        get_command,
    )

    handlers = {
        'backtest': handle_backtest,
        'paper': handle_paper,
        'fetch': handle_fetch,
        'compare': handle_compare,
        'report': handle_report,
        'list': handle_list,
        'test': handle_test,
        'parallel': handle_parallel,
    }

    handler = handlers.get(parsed.command)
    if not handler:
        cmd = get_command(parsed.command)
        if cmd:
            handler = lambda a: cmd.execute(a)
    if handler:
        try:
            return handler(parsed)
        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user")
            return 130
        except Exception as e:
            if parsed.verbose:
                import traceback
                traceback.print_exc()
            else:
                print(f"❌ Error: {e}")
            return 1

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
