#!/usr/bin/env python3
"""
Simulate a small production run: stream through a dataset epoch-by-epoch, call `make_decision` and
persist actions + metrics. Useful as a quick "mise en prod" smoke test.

Usage examples:
  venv/bin/python scripts/prod_run_simulator.py --dataset data/asset_b_train.csv --out experiments/eval/prod_sim.csv --fees 0.0001 --slippage 0.0 --use-best

This script supports 3 modes: `--use-best` (uses best robust candidate), `--params` (JSON string) or both.
"""
import os
import sys
import json
import argparse
import importlib.util
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scoring.scoring import get_local_score


def load_bot(path_to_bot: str):
    spec = importlib.util.spec_from_file_location("m", path_to_bot)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_best_candidate():
    best_path = os.path.join('experiments', 'optuna', 'robust_blend_best.json')
    fallback_top5 = os.path.join('experiments', 'optuna', 'phase3_blend_top5.json')
    if os.path.exists(best_path):
        with open(best_path, 'r') as f:
            j = json.load(f)
            if 'best_params' in j:
                # optuna best_params is flattened; make it compatible with `strategy_params` used in code
                p = j['best_params']
                params = {'blend': float(p['blend']), 'sub_params': {'short': int(p['short']), 'long': int(p['long']), 'threshold': float(p['threshold']), 'vol_window': int(p['vol_window']), 'vol_multiplier': float(p['vol_multiplier'])}}
                return params
    if os.path.exists(fallback_top5):
        with open(fallback_top5, 'r') as f:
            arr = json.load(f)
            if arr:
                p = arr[0]['params']
                params = {'blend': float(p['blend']), 'sub_params': {'short': int(p['short']), 'long': int(p['long']), 'threshold': float(p['threshold']), 'vol_window': int(p['vol_window']), 'vol_multiplier': float(p['vol_multiplier'])}}
                return params
    return None


def run_stream(dataset_path, bot, params, transaction_fees=0.0001, slippage=0.0, fixed_cost=0.0, out_csv=None, strategy_arg: str | None = 'default'):
    df = pd.read_csv(dataset_path, index_col=0)
    df['Cash'] = 1
    history = []
    decisions = []
    is_multi = 'Asset A' in df.columns and 'Asset B' in df.columns

    for epoch, row in df.iterrows():
        if is_multi:
            if strategy_arg is not None and strategy_arg != 'default':
                decision = bot.make_decision(int(epoch), float(row['Asset A']), float(row['Asset B']), strategy=strategy_arg, params=params, history=history)
            else:
                decision = bot.make_decision(int(epoch), float(row['Asset A']), float(row['Asset B']), params=params, history=history)
        else:
            if strategy_arg is not None and strategy_arg != 'default':
                decision = bot.make_decision(int(epoch), float(row['Asset B']), strategy=strategy_arg, params=params, history=history)
            else:
                decision = bot.make_decision(int(epoch), float(row['Asset B']), params=params, history=history)
        decision['epoch'] = int(epoch)
        decisions.append(decision)

    positions_df = pd.DataFrame(decisions).set_index('epoch')
    res = get_local_score(prices=df, positions=positions_df, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    positions_df.to_csv(out_csv)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/asset_b_train.csv')
    parser.add_argument('--out', default='experiments/eval/prod_sim.csv')
    parser.add_argument('--transaction-fees', type=float, default=0.0001)
    parser.add_argument('--slippage', type=float, default=0.0)
    parser.add_argument('--fixed-cost', type=float, default=0.0)
    parser.add_argument('--use-best', action='store_true', help='Use best candidate saved by optuna')
    parser.add_argument('--params', default=None, help='JSON params string for strategy')
    parser.add_argument('--strategy', default='default', help='Strategy to force: pass "default" to use bot default')
    args = parser.parse_args()

    bot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bot_trade.py')
    bot = load_bot(bot_path)

    params = None
    if args.use_best:
        params = load_best_candidate()
        if params is None:
            print('No best candidate file found; pass --params or run optuna first')
            sys.exit(2)
    if args.params:
        try:
            params = json.loads(args.params)
        except Exception as e:
            print('Failed to load params JSON:', e)
            sys.exit(2)
    # If no params and not using --use-best, we will call make_decision without overriding strategy / params
    # so it will use the bot's default strategy and tuned parameters.

    print('Running prod sim on', args.dataset, 'with params', params, 'strategy', args.strategy)
    res = run_stream(args.dataset, bot, params, transaction_fees=args.transaction_fees, slippage=args.slippage, fixed_cost=args.fixed_cost, out_csv=args.out, strategy_arg=args.strategy)
    print('Sim done: base_score =', res['scores']['base_score'])
    print('Stats:', res['stats'])
    # Save metrics
    os.makedirs('experiments/eval', exist_ok=True)
    with open('experiments/eval/prod_sim_result.json', 'w') as f:
        json.dump({'dataset': args.dataset, 'params': params, 'scores': res['scores'], 'stats': res['stats']}, f, indent=2)


if __name__ == '__main__':
    main()
