#!/usr/bin/env python3
import os
import sys
import json
import importlib.util
from typing import List
import pandas as pd
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scoring.scoring import get_local_score


def load_bot(path_to_bot: str):
    spec = importlib.util.spec_from_file_location("module.name", path_to_bot)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_on_csv(csv_path: str, bot_module, strategy: str, strategy_params, transaction_fees=0.0001, slippage=0.0, fixed_cost=0.0):
    df = pd.read_csv(csv_path, index_col=0)
    df['Cash'] = 1
    history = []
    positions = []
    # detect asset columns
    is_multi = 'Asset A' in df.columns and 'Asset B' in df.columns
    for epoch, row in df.iterrows():
        if is_multi:
            if strategy and strategy != 'default':
                decision = bot_module.make_decision(int(epoch), float(row['Asset A']), float(row['Asset B']), strategy=strategy, params=strategy_params, history=history)
            else:
                decision = bot_module.make_decision(int(epoch), float(row['Asset A']), float(row['Asset B']), params=strategy_params, history=history)
        else:
            if strategy and strategy != 'default':
                decision = bot_module.make_decision(int(epoch), float(row['Asset B']), strategy=strategy, params=strategy_params, history=history)
            else:
                decision = bot_module.make_decision(int(epoch), float(row['Asset B']), params=strategy_params, history=history)
        decision['epoch'] = int(epoch)
        positions.append(decision)
    positions_df = pd.DataFrame(positions).set_index('epoch')
    res = get_local_score(prices=df, positions=positions_df, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost)
    return res


def discover_csvs(folder: str) -> List[str]:
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.csv')]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets-dir', default='data')
    parser.add_argument('--strategy', default='default')
    parser.add_argument('--strategy-params', default=None)
    parser.add_argument('--transaction-fees', type=float, default=0.0005)
    parser.add_argument('--slippage', type=float, default=0.0001)
    parser.add_argument('--fixed-cost', type=float, default=0.0)
    parser.add_argument('--bot-path', default=None, help='Path to bot_trade.py to use (defaults to root bot_trade.py which is multi-asset capable)')
    args = parser.parse_args()

    csvs = discover_csvs(args.datasets_dir)
    print(f"Found {len(csvs)} CSV files in {args.datasets_dir}")

    # prepare bot module (single implementation at repository root handles both single and multi-asset)
    root_bot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bot_trade.py')
    root_bot = load_bot(root_bot_path)
    print('Loaded root bot (multi-asset capable)')

    strategy_params = None
    if args.strategy_params:
        try:
            strategy_params = json.loads(args.strategy_params)
        except Exception:
            strategy_params = None

    results = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path, index_col=0)
        is_multi = 'Asset A' in df.columns and 'Asset B' in df.columns
        # our root bot supports multi-asset when two prices are provided, so pass root_bot always
        res = run_on_csv(csv_path, root_bot, args.strategy, strategy_params, args.transaction_fees, args.slippage, args.fixed_cost)
        stats = res['stats']
        score = res['scores']['base_score']
        print(f"{os.path.basename(csv_path)} => base={score:.4f}, PnL={stats['cumulative_return']*100:.2f}%, Sharpe={stats['sharpe_ratio']:.3f}, MDD={stats['max_drawdown']:.3f}")
        results.append({'csv': csv_path, 'base_score': score, 'pnl%': stats['cumulative_return']*100, 'sharpe': stats['sharpe_ratio'], 'mdd': stats['max_drawdown']})

    df_out = pd.DataFrame(results)
    os.makedirs('experiments/eval', exist_ok=True)
    df_out.to_csv('experiments/eval/multi_dataset_results.csv', index=False)
    print('Saved experiments/eval/multi_dataset_results.csv')


if __name__ == '__main__':
    main()
