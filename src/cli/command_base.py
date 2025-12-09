"""
Command base class for the CLI commands.

Each command implements an `execute(args)` method and can reuse shared
utilities provided by this base class: loading CSVs, running backtests,
building benchmarks and creating output directories.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Optional, List, Tuple

from ..strategies import StrategyFactory
from ..reporting import ReportGenerator, ChartGenerator
from ..core.models import BacktestResult, Price
from ..data import DataSourceFactory
from ..engine import BacktestConfig, BacktestEngine, StrategyComparator


class CLICommand:
    """Base class for CLI commands.

    Subclasses should implement `execute(self, args: Namespace) -> int`.
    """

    def load_prices_from_csv(self, path: str):
        """Load CSV using `DataSourceFactory` and return (data_source, prices).
        """
        ds = DataSourceFactory.from_csv(path)
        prices = ds.fetch_prices()
        return ds, prices

    def run_backtest(self, strategy_name: str, data_source, capital: float, fee_rate: float) -> BacktestResult:
        """Run a backtest for a single strategy on a data_source.
        """
        strategy = StrategyFactory.create(strategy_name)
        cfg = BacktestConfig(initial_capital=capital, fee_rate=fee_rate)
        engine = BacktestEngine(cfg)
        return engine.run(strategy, data_source)

    def build_benchmark(self, data_source, capital: float):
        comparator = StrategyComparator(BacktestConfig(initial_capital=capital))
        return comparator.calculate_benchmark(data_source)

    def save_report_and_charts(self, result: BacktestResult, prices: List[Price], data_source, output_dir: str) -> None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        reporter = ReportGenerator()
        reporter.save(result, p / "report.md", self.build_benchmark(data_source, result.initial_capital))
        charts = ChartGenerator()
        prices_a = [p.asset_a for p in prices]
        prices_b = [p.asset_b for p in prices]
        charts.plot_performance(result, prices_a, prices_b, p / "performance.png")
        charts.plot_allocations(result, p / "allocations.png")
        charts.plot_drawdown(result, p / "drawdown.png")

    def ensure_output_dir(self, path: str) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def execute(self, args: Namespace) -> int:  # pragma: no cover - override
        raise NotImplementedError()
