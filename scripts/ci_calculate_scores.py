"""CI helper script to run strategy backtests and compute scores.
This script is intended to be executed from GitHub Actions or locally e.g.

python scripts/ci_calculate_scores.py --data data/ci_data.csv --strategy safe_profit

It will write outputs to `outputs/ci_backtest.csv` and optionally append job outputs to
$GITHUB_OUTPUT if present (GitHub Actions).
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from typing import List

from src.data.sources import DataSourceFactory
from src.strategies import StrategyFactory
from src.engine.backtest import BacktestConfig, BacktestEngine, StrategyComparator


def run_backtests(datasets: List[str], strategy_name: str) -> tuple[list, list]:
    cfg = BacktestConfig(initial_capital=10000)
    engine = BacktestEngine(cfg)
    comp = StrategyComparator(cfg)
    strat = StrategyFactory.create(strategy_name)

    results = []
    alphas = []
    for ds_file in datasets:
        ds = DataSourceFactory.from_csv(ds_file)
        res = engine.run(strat, ds)
        bm = comp.calculate_benchmark(ds)
        base = getattr(res, "total_return", 0.0)
        sharpe = getattr(res, "sharpe_ratio", 0.0)
        pnl = getattr(res, "pnl", 0.0)
        alpha = base - bm.get("50_50", 0.0)
        alphas.append(alpha)
        results.append((ds_file, base, sharpe, res.max_drawdown, pnl, alpha))
    return results, alphas


def write_outputs(results, alphas, out_csv: str = "outputs/ci_backtest.csv"):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w") as csvfile:
        csvfile.write("dataset,base_return,sharpe,max_drawdown,pnl,alpha\n")
        for dsf, base, sharpe, max_dd, pnl, alpha in results:
            csvfile.write(f"{dsf},{base:.6f},{sharpe:.6f},{max_dd:.6f},{pnl:.6f},{alpha:.6f}\n")
        avg_alpha = sum(alphas) / len(alphas) if alphas else 0.0
        csvfile.write(f"avg_alpha,{avg_alpha:.6f}\n")
    return avg_alpha


def set_github_outputs(base, sharpe, pnl):
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as out:
            out.write(f"base_score={base:.4f}\n")
            out.write(f"sharpe_score={sharpe:.4f}\n")
            out.write(f"pnl_score={pnl:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="*",
        default=["data/ci_data.csv"],
        help="Datasets CSV files",
    )
    parser.add_argument("--strategy", "-s", default="safe_profit", help="Strategy name")
    parser.add_argument("--run-extended", action="store_true", default=False)
    parser.add_argument("--output", "-o", default="outputs/ci_backtest.csv")
    args = parser.parse_args()

    datasets = list(args.datasets)
    if args.run_extended:
        extended_files = [
            "data/crypto_btc_eth_4h_90d.csv",
            "data/crypto_btc_eth_1h_60d.csv",
            "data/crypto_btc_eth_1h_30d.csv",
        ]
        datasets = [f for f in extended_files if os.path.exists(f)]
        if not datasets:
            # generate a few variants from ci_data.csv
            import csv

            base = []
            with open("data/ci_data.csv", "r") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    base.append(row)
            for i, factor in enumerate([1.0, 0.98, 1.02], start=1):
                out_path = f"data/ci_data_var{i}.csv"
                with open(out_path, "w") as out:
                    out.write("epoch,Asset A,Asset B,Cash\n")
                    for r in base:
                        epoch = r["epoch"]
                        a = float(r["Asset A"]) * factor
                        b = float(r["Asset B"]) * (1.0 + (factor - 1.0) * 0.5)
                        out.write(f"{epoch},{a:.2f},{b:.2f},{r['Cash']}\n")
                datasets.append(out_path)

    # Run backtests
    results, alphas = run_backtests(datasets, args.strategy)

    # Print & write outputs
    print("=" * 50)
    print("📊 BACKTEST RESULTS")
    print("=" * 50)
    for dsf, base, sharpe, max_dd, pnl, alpha in results:
        print("\nDataset:", dsf)
        print("Return:", base)
        print("Sharpe:", sharpe)
        print("Max DD:", max_dd)
        print("Alpha vs 50/50:", alpha)
        print("PnL:", pnl)

    avg_alpha = write_outputs(results, alphas, out_csv=args.output)

    # Job outputs
    if results:
        last = results[-1]
        set_github_outputs(last[1], last[2], last[4])

    print("\nAGGREGATE: avg_alpha =", avg_alpha)
    fail_on_underperform = (
        os.environ.get("VALIDATION_FAIL_ON_UNDERPERFORM", "false").lower()
        in ("1", "true", "yes")
    )
    min_alpha = float(os.environ.get('MIN_ALPHA_TO_FAIL', '-0.005'))
    if avg_alpha < min_alpha:
        print('❌ Average alpha below threshold')
        if fail_on_underperform:
            print('FAILING because VALIDATION_FAIL_ON_UNDERPERFORM is set')
            sys.exit(1)
        else:
            print(
                'Warning: average alpha below threshold but '
                'VALIDATION_FAIL_ON_UNDERPERFORM is false; continuing'
            )
    else:
        print('✅ Strategy passes validation!')


if __name__ == '__main__':
    main()
