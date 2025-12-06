#!/usr/bin/env python3
import optuna
import pandas as pd
import numpy as np
import json
import os
import sys
# add workspace root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scoring.scoring import get_local_score
from bot_trade import make_decision

CSV = 'data/asset_b_train.csv'

# Load data
prices_df = pd.read_csv(CSV, index_col=0)
prices_df.columns = [c for c in prices_df.columns]  # simple
prices_df['Cash'] = 1

# We'll do walk-forward: split into n_splits folds
n_splits = 3
n = len(prices_df)
fold_size = n // n_splits

# Helper to run backtest with SMA short/long

def run_backtest_sma(short, long):
    positions = []
    history = []
    for epoch, row in prices_df.iterrows():
        price = float(row['Asset B'])
        decision = make_decision(int(epoch), price, strategy='sma', params={'short': short, 'long': long}, history=history)
        decision['epoch'] = int(epoch)
        positions.append(decision)
    positions_df = pd.DataFrame(positions).set_index('epoch')
    results = get_local_score(prices=prices_df, positions=positions_df)
    return results['scores']['base_score']


def crossval_score_sma(short, long, transaction_fees=0.0001, slippage=0.0, fixed_cost_per_trade=0.0):
    # for simplicity we'll run the strategy on the entire series and evaluate on each fold's test set
    # For true walk-forward we'd retrain model or re-run strategy only up to that fold, but our SMA is simple and does not train
    # So we will compute the score across each fold using the entire backtest and then isolate test indices
    # Simple: compute positions based on full history and compute results once, then compute sub-scores per fold
    positions = []
    history = []
    for epoch, row in prices_df.iterrows():
        price = float(row['Asset B'])
        decision = make_decision(int(epoch), price, strategy='sma', params={'short': short, 'long': long}, history=history)
        decision['epoch'] = int(epoch)
        positions.append(decision)
    positions_df = pd.DataFrame(positions).set_index('epoch')

    # now compute per fold score using subsets
    scores = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i+1) * fold_size if i < n_splits-1 else n
        # select subset
        prices_sub = prices_df.iloc[start:end]
        positions_sub = positions_df.iloc[start:end]
        try:
            res = get_local_score(prices=prices_sub, positions=positions_sub, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost_per_trade)
            scores.append(res['scores']['base_score'])
        except Exception as e:
            # if sub-sample too small or invalid, skip
            scores.append(0)
    return float(np.nanmean(scores))


def objective(trial: optuna.Trial, transaction_fees=0.0001, slippage=0.0, fixed_cost_per_trade=0.0):
    short = trial.suggest_int('short', 2, 20)
    long = trial.suggest_int('long', 21, 60)
    if long <= short:
        return 0.0
    score = crossval_score_sma(short, long, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost_per_trade)
    return score


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=60)
    parser.add_argument('--transaction-fees', type=float, default=0.0005)
    parser.add_argument('--slippage', type=float, default=0.0)
    parser.add_argument('--fixed-cost-per-trade', type=float, default=0.0)
    args = parser.parse_args()
    def wrapped_objective(trial: optuna.Trial):
        return objective(trial, transaction_fees=args.transaction_fees, slippage=args.slippage, fixed_cost_per_trade=args.fixed_cost_per_trade)
    study.optimize(wrapped_objective, n_trials=args.n_trials)
    print('Best params:', study.best_params)
    print('Best value:', study.best_value)
    # Save results
    os.makedirs('experiments/optuna', exist_ok=True)
    with open('experiments/optuna/sma_optuna_best.json', 'w') as f:
        json.dump({'best_params': study.best_params, 'best_value': study.best_value}, f)
    print('Saved best to experiments/optuna/sma_optuna_best.json')
