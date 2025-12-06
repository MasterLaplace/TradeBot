#!/usr/bin/env python3
import os
import sys
import json
import math
from typing import Dict

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from scoring.scoring import get_local_score
from bot_trade import make_decision, reset_history

CSV = 'data/asset_b_train.csv'


def walk_forward_eval(strategy: str = 'baseline', strategy_params: Dict = None, n_splits: int = 3, initial_window: int = None, transaction_fees: float = 0.0001, slippage: float = 0.0, fixed_cost_per_trade: float = 0.0):
    df = pd.read_csv(CSV, index_col=0)
    df['Cash'] = 1
    n = len(df)
    if initial_window is None:
        initial_window = max(10, n // (n_splits + 1))
    step = max(2, (n - initial_window) // n_splits)

    scores = []
    for k in range(n_splits):
        start_train = 0
        end_train = initial_window + k * step
        start_test = end_train
        end_test = end_train + step if k < n_splits - 1 else n
        # if test window too small, skip
        if end_test - start_test < 2:
            scores.append(float('nan'))
            continue

        # Run bot on the whole data but later evaluate on test slice
        reset_history()
        positions = []
        history = []
        for idx, row in df.iterrows():
            decision = make_decision(int(idx), float(row['Asset B']), strategy=strategy, params=strategy_params, history=history)
            positions.append({**decision, 'epoch': int(idx)})
        positions_df = pd.DataFrame(positions).set_index('epoch')

        prices_sub = df.iloc[start_test:end_test].copy()
        positions_sub = positions_df.iloc[start_test:end_test].copy()
        # Reindex to 0..n-1 for the local backtest expectations
        positions_sub.index = range(len(positions_sub))
        prices_sub.index = range(len(prices_sub))
        try:
            res = get_local_score(prices=prices_sub, positions=positions_sub, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost_per_trade)
            scores.append(res['scores']['base_score'])
        except Exception as e:
            print('Exception evaluating fold', k, 'start_test', start_test, 'end_test', end_test, 'exc:', e)
            scores.append(float('nan'))

    mean_score = float(pd.Series(scores).dropna().mean()) if len([s for s in scores if not math.isnan(s)]) > 0 else float('nan')
    return {'fold_scores': scores, 'mean_score': mean_score}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='baseline')
    parser.add_argument('--strategy-params', default=None)
    parser.add_argument('--n-splits', type=int, default=3)
    parser.add_argument('--initial-window', type=int, default=None)
    parser.add_argument('--transaction-fees', type=float, default=0.0001)
    parser.add_argument('--slippage', type=float, default=0.0)
    parser.add_argument('--fixed-cost-per-trade', type=float, default=0.0)
    args = parser.parse_args()

    params = json.loads(args.strategy_params) if args.strategy_params else {}
    out = walk_forward_eval(
        strategy=args.strategy,
        strategy_params=params,
        n_splits=args.n_splits,
        initial_window=args.initial_window,
        transaction_fees=args.transaction_fees,
        slippage=args.slippage,
        fixed_cost_per_trade=args.fixed_cost_per_trade,
    )
    print(out)