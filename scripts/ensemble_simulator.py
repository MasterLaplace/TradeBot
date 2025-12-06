#!/usr/bin/env python3
"""
Create an ensemble of top-N candidates and run a simulated production run by averaging their per-epoch allocations.

Creates positions CSV and computes the score via `get_local_score`.
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


def load_topn(n=5, path='experiments/optuna/phase3_blend_top5.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path,'r') as f:
        arr = json.load(f)
    return [x['params'] for x in arr[:n]]


def ensemble_run(dataset_path, bot, candidates, transaction_fees=0.0001, slippage=0.0, fixed_cost=0.0, out_csv=None):
    df = pd.read_csv(dataset_path, index_col=0)
    df['Cash'] = 1
    history_per_candidate = [[] for _ in candidates]
    decisions = []
    is_multi = 'Asset A' in df.columns and 'Asset B' in df.columns

    for epoch, row in df.iterrows():
        # gather decisions from each candidate
        per_allocs = []
        for i, cand in enumerate(candidates):
            hist = history_per_candidate[i]
            if is_multi:
                d = bot.make_decision(int(epoch), float(row['Asset A']), float(row['Asset B']), strategy='blended', params=cand, history=hist)
            else:
                d = bot.make_decision(int(epoch), float(row['Asset B']), strategy='blended', params=cand, history=hist)
            # append to candidate history
            d['epoch'] = int(epoch)
            per_allocs.append(d)
            if is_multi:
                hist.append({'epoch': int(epoch), 'priceA': float(row['Asset A']), 'priceB': float(row['Asset B'])})
            else:
                hist.append({'epoch': int(epoch), 'price': float(row['Asset B'])})
        # average allocations: for each asset key, average weights
        # read keys from per_allocs[0]
        keys = per_allocs[0].keys()
        avg = {k: sum(d[k] for d in per_allocs) / len(per_allocs) for k in keys}
        avg['epoch'] = int(epoch)
        decisions.append(avg)
    positions_df = pd.DataFrame(decisions).set_index('epoch')
    res = get_local_score(prices=df, positions=positions_df, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    positions_df.to_csv(out_csv)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/asset_b_train.csv')
    parser.add_argument('--out', default='experiments/eval/ensemble_sim.csv')
    parser.add_argument('--topn', default=5, type=int)
    parser.add_argument('--transaction-fees', type=float, default=0.0001)
    parser.add_argument('--slippage', type=float, default=0.0)
    args = parser.parse_args()

    bot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bot_trade.py')
    bot = load_bot(bot_path)
    candidates = load_topn(args.topn)
    print('Using topn candidates:', candidates)
    res = ensemble_run(args.dataset, bot, candidates, transaction_fees=args.transaction_fees, slippage=args.slippage, out_csv=args.out)
    print('Ensemble done: base_score =', res['scores']['base_score'])
    print('Stats:', res['stats'])

if __name__ == '__main__':
    main()
