"""Parallel backtesting utilities.

Provides a function to run strategies concurrently across multiple datasets
and produce a CSV summary. This is integrated with CLI `parallel` command.
"""

from __future__ import annotations
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Iterable
import os
import glob
from multiprocessing import Manager, Process
import requests
import time
from datetime import datetime

# Top-level imports for data and engine modules (pandas is expected to be installed)
from ..strategies import StrategyFactory
from ..data.sources import DataSourceFactory, BinanceRESTSource
from ..core.models import Price
from ..engine.backtest import BacktestConfig, BacktestEngine, StrategyComparator
from ..engine.paper_trading import PaperTradingEngine


def _run_strategy_on_dataset(strategy_name: str, dataset_path: str, initial_capital: float, fee_rate: float):
    """Run backtest engine for a strategy on a CSV dataset and return metrics tuple.

    This small helper keeps the heavy-lifting inside the BacktestEngine while
    providing a simple return format for parallelization.
    """
    cfg = BacktestConfig(initial_capital=initial_capital, fee_rate=fee_rate)
    engine = BacktestEngine(cfg)
    comp = StrategyComparator(cfg)
    strat = StrategyFactory.create(strategy_name)

    ds = DataSourceFactory.from_csv(dataset_path)
    res = engine.run(strat, ds)
    bm = comp.calculate_benchmark(ds)
    base = getattr(res, "total_return", 0.0)
    sharpe = getattr(res, "sharpe_ratio", 0.0)
    max_dd = getattr(res, "max_drawdown", 0.0)
    final_value = getattr(res, "final_value", 0.0)
    num_trades = getattr(res, "num_trades", 0)
    alpha = base - bm.get("50_50", 0.0)
    return (strategy_name, dataset_path, base, final_value, sharpe, max_dd, num_trades, alpha)


def run_parallel_backtests(
    strategies: List[str],
    datasets: List[str],
    workers: Optional[int] = None,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    per_strategy_dir: Optional[str] = None,
    show_progress: bool = True,
) -> List[Tuple[str, str, float, float, float, float, int, float]]:
    """Run backtests in parallel for the given strategies and datasets.

    Returns a flat list of tuples: (strategy, dataset, base_return, sharpe, max_dd, alpha)
    """
    results = []
    workers = workers or min(len(strategies) * len(datasets), os.cpu_count() or 2)

    expanded_datasets: List[str] = []
    for ds in datasets:
        matches = glob.glob(ds)
        if matches:
            expanded_datasets.extend(sorted(matches))
        else:
            expanded_datasets.append(ds)

    if per_strategy_dir:
        Path(per_strategy_dir).mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {}
        for s in strategies:
            for ds in expanded_datasets:
                fut = exe.submit(_run_strategy_on_dataset, s, ds, initial_capital, fee_rate)
                futures[fut] = (s, ds)

        futures_list = list(futures.keys())

        use_tqdm = False
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                use_tqdm = True
            except Exception:
                use_tqdm = False

        iter_futures = as_completed(futures_list)
        if use_tqdm:
            iter_futures = tqdm(iter_futures, total=len(futures_list), desc="Parallel backtests")

        for fut in iter_futures:
            s, ds = futures[fut]
            try:
                r = fut.result()
                results.append(r)

                if per_strategy_dir:
                    per_file = Path(per_strategy_dir) / f"{s}.csv"
                    existed = per_file.exists()
                    with open(per_file, 'a', newline='') as fh:
                        writer = csv.writer(fh)
                        if not existed:
                            writer.writerow(['strategy', 'dataset', 'base_return', 'final_value', 'sharpe', 'max_drawdown', 'num_trades', 'alpha'])
                        writer.writerow(r)

            except Exception as e:
                print(f"⛔ Error for {s} on {ds}: {e}")
                results.append((s, ds, float('nan'), float('nan'), float('nan'), float('nan'), 0, float('nan')))

    return results


def write_summary_csv(results: List[Tuple[str, str, float, float, float, float, int, float]], path: str, append: bool = False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if append and Path(path).exists() else 'w'
    with open(path, mode, newline="") as fh:
        writer = csv.writer(fh)
        if mode == 'w' or (mode == 'a' and fh.tell() == 0):
            writer.writerow(["strategy", "dataset", "base_return", "final_value", "sharpe", "max_drawdown", "num_trades", "alpha"])
        for row in results:
            writer.writerow(row)


def _run_strategy_live(strategy_name: str, symbols: List[str], duration_seconds: int, interval_seconds: float, initial_capital: float, fee_rate: float, shared_prices=None, key=None):
    """Execute a single live paper trading session.

    The function can use a shared price map (populated by a broadcaster)
    to avoid hitting external APIs per worker. If no shared prices are
    available the engine falls back to a small REST request per tick.
    """
    try:
        strat = StrategyFactory.create(strategy_name)
    except Exception as e:
        import traceback
        print(f"⛔ Error instantiating strategy {strategy_name}: {e}")
        traceback.print_exc()
        raise

    price_provider = None
    if shared_prices is not None and key is not None:

        def _provider():
            data = shared_prices.get(key)
            if not data:
                base_url = 'https://api.binance.com/api/v3'
                r_a = requests.get(f"{base_url}/ticker/price", params={'symbol': symbols[0].upper()}, timeout=10)
                r_b = requests.get(f"{base_url}/ticker/price", params={'symbol': symbols[1].upper()}, timeout=10)
                r_a.raise_for_status()
                r_b.raise_for_status()
                price_a = float(r_a.json().get('price', 0))
                price_b = float(r_b.json().get('price', 0))
                return Price(asset_a=price_a, asset_b=price_b, timestamp=datetime.utcnow())
            return Price(asset_a=float(data['asset_a']), asset_b=float(data['asset_b']), timestamp=datetime.fromisoformat(data['timestamp']))

        price_provider = _provider

    try:
        engine = PaperTradingEngine(
        strategy=strat,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        symbol_a=symbols[0],
        symbol_b=symbols[1],
        price_provider=price_provider,
    )
    except Exception as e:
        import traceback
        print(f"⛔ Error creating PaperTradingEngine for {strategy_name}: {e}")
        traceback.print_exc()
        raise

    state = engine.run(duration_seconds=duration_seconds, interval_seconds=interval_seconds, on_tick=None)

    final_value = state.portfolio_history[-1] if state.portfolio_history else state.portfolio.value(engine._price_source.get_current_price())
    pnl = state.pnl
    pnl_percent = state.pnl_percent
    num_trades = len(state.trades)

    return (strategy_name, f"{symbols[0]}_{symbols[1]}", final_value, pnl, pnl_percent, num_trades)


def run_parallel_paper(
    strategies: List[str],
    symbols: List[str],
    workers: Optional[int] = None,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    duration_seconds: int = 120,
    interval_seconds: float = 5.0,
    per_strategy_dir: Optional[str] = None,
    show_progress: bool = True,
    use_ws: bool = False,
) -> List[Tuple[str, str, float, float, float, int]]:
    """Run paper trading sessions in parallel for each strategy.

    Returns tuples: (strategy, symbols, final_value, pnl, pnl_percent, num_trades)
    """
    results = []
    workers = workers or min(len(strategies), os.cpu_count() or 2)

    if per_strategy_dir:
        Path(per_strategy_dir).mkdir(parents=True, exist_ok=True)

    manager = Manager()
    shared_prices = manager.dict()

    def broadcaster_loop(shared_prices, key, symbols, interval_seconds):
        base_url = 'https://api.binance.com/api/v3'
        while True:
            try:
                r_a = requests.get(f"{base_url}/ticker/price", params={'symbol': symbols[0].upper()}, timeout=10)
                r_b = requests.get(f"{base_url}/ticker/price", params={'symbol': symbols[1].upper()}, timeout=10)
                r_a.raise_for_status()
                r_b.raise_for_status()
                price_a = float(r_a.json().get('price', 0))
                price_b = float(r_b.json().get('price', 0))
                shared_prices[key] = {'asset_a': price_a, 'asset_b': price_b, 'timestamp': datetime.utcnow().isoformat()}
            except Exception:
                pass
            time.sleep(max(0.25, interval_seconds))

    key = f"{symbols[0]}_{symbols[1]}"
    broadcaster = None
    if use_ws:
        try:
            from ..data.broadcaster import PriceBroadcaster

            pb = PriceBroadcaster(symbols, interval_seconds, use_ws=True)
            pb.start(shared_prices, key)
            broadcaster = pb
        except Exception:
            broadcaster = Process(target=broadcaster_loop, args=(shared_prices, key, symbols, interval_seconds))
            broadcaster.daemon = True
            broadcaster.start()
    else:
        broadcaster = Process(target=broadcaster_loop, args=(shared_prices, key, symbols, interval_seconds))
        broadcaster.daemon = True
        broadcaster.start()
    print(f"🔁 Started centralized price broadcaster for {symbols[0]}/{symbols[1]}")

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {}
        for s in strategies:
            fut = exe.submit(_run_strategy_live, s, symbols, duration_seconds, interval_seconds, initial_capital, fee_rate, shared_prices, key)
            futures[fut] = (s, f"{symbols[0]}_{symbols[1]}")

        futures_list = list(futures.keys())

        use_tqdm = False
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                use_tqdm = True
            except Exception:
                use_tqdm = False

        iter_futures = as_completed(futures_list)
        if use_tqdm:
            iter_futures = tqdm(iter_futures, total=len(futures_list), desc="Parallel paper")

        for fut in iter_futures:
            s, sym = futures[fut]
            try:
                r = fut.result()
                results.append(r)

                if per_strategy_dir:
                    per_file = Path(per_strategy_dir) / f"{s}.csv"
                    existed = per_file.exists()
                    with open(per_file, 'a', newline='') as fh:
                        writer = csv.writer(fh)
                        if not existed:
                            writer.writerow(['strategy', 'symbols', 'final_value', 'pnl', 'pnl_percent', 'num_trades'])
                        writer.writerow(r)

            except Exception as e:
                import traceback
                print(f"⛔ Error for {s} on live session: {e}")
                traceback.print_exc()
                results.append((s, sym, float('nan'), float('nan'), float('nan'), 0))

    try:
        if isinstance(broadcaster, Process):
            broadcaster.terminate()
        elif broadcaster is not None:
            broadcaster.stop()  # type: ignore
    except Exception:
        pass

    return results


def write_summary_csv_live(results: List[Tuple[str, str, float, float, float, int]], path: str, append: bool = False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if append and Path(path).exists() else 'w'
    with open(path, mode, newline="") as fh:
        writer = csv.writer(fh)
        if mode == 'w' or (mode == 'a' and fh.tell() == 0):
            writer.writerow(["strategy", "symbols", "final_value", "pnl", "pnl_percent", "num_trades"])
        for row in results:
            writer.writerow(row)
