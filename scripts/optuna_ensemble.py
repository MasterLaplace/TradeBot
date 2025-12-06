#!/usr/bin/env python3
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import optuna
import pandas as pd
import math
from bot_trade import make_decision, reset_history
from scoring.scoring import get_local_score

CSV = 'data/asset_b_train.csv'


def walk_forward_mean_ensemble(params, n_splits=5, transaction_fees=0.0001, slippage=0.0, fixed_cost_per_trade=0.0):
    df = pd.read_csv(CSV, index_col=0)
    df['Cash'] = 1
    n = len(df)
    initial_window = max(10, n // (n_splits + 1))
    step = max(2, (n - initial_window) // n_splits)
    scores = []
    for k in range(n_splits):
        start_test = initial_window + k * step
        end_test = start_test + step if k < n_splits - 1 else n
        reset_history()
        positions = []
        history = []
        for idx, row in df.iterrows():
            d = make_decision(int(idx), float(row['Asset B']), strategy='ensemble', params=params, history=history)
            d['epoch'] = int(idx)
            positions.append(d)
        positions_df = pd.DataFrame(positions).set_index('epoch')
        prices_sub = df.iloc[start_test:end_test].copy()
        positions_sub = positions_df.iloc[start_test:end_test].copy()
        prices_sub.index = range(len(prices_sub))
        positions_sub.index = range(len(positions_sub))
        try:
            res = get_local_score(prices=prices_sub, positions=positions_sub, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost_per_trade)
            scores.append(res['scores']['base_score'])
        except Exception:
            scores.append(float('nan'))
    return float(pd.Series(scores).dropna().mean()) if len([s for s in scores if not math.isnan(s)]) > 0 else float('nan')


def objective(trial: optuna.Trial, transaction_fees=0.0001, slippage=0.0, fixed_cost_per_trade=0.0):
    switch_vol_window = trial.suggest_int('switch_vol_window', 5, 20)
    switch_vol_threshold = trial.suggest_float('switch_vol_threshold', 0.01, 0.06)
    # preferred strategy: 0 = blended, 1 = composite
    preferred = trial.suggest_categorical('preferred', [0, 1])
    # params for blended sub-strategy
    short = trial.suggest_int('short', 2, 10)
    long = trial.suggest_int('long', 15, 60)
    if long <= short:
        return float('nan')
    threshold = trial.suggest_float('threshold', 0.005, 0.05)
    vol_window = trial.suggest_int('vol_window', 5, 20)
    vol_multiplier = trial.suggest_float('vol_multiplier', 0.0, 2.0)
    # compose params
    preferred_params = {'short': short, 'long': long, 'threshold': threshold, 'vol_window': vol_window, 'vol_multiplier': vol_multiplier}
    params = {
        'switch_vol_window': switch_vol_window,
        'switch_vol_threshold': switch_vol_threshold,
        'fallback': 'baseline',
        'preferred': 'blended' if preferred == 0 else 'composite',
        'preferred_params': preferred_params,
    }
    score = walk_forward_mean_ensemble(params, n_splits=5, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost_per_trade)
    return score


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--transaction-fees', type=float, default=0.0005)
    parser.add_argument('--slippage', type=float, default=0.0)
    parser.add_argument('--fixed-cost-per-trade', type=float, default=0.0)
    args = parser.parse_args()
    def wrapped_objective(trial: optuna.Trial):
        return objective(trial, transaction_fees=args.transaction_fees, slippage=args.slippage, fixed_cost_per_trade=args.fixed_cost_per_trade)
    study.optimize(wrapped_objective, n_trials=args.n_trials)
    print('Best params:', study.best_params)
    print('Best value:', study.best_value)
    os.makedirs('experiments/optuna', exist_ok=True)
    with open('experiments/optuna/ensemble_optuna_best.json', 'w') as f:
        json.dump({'best_params': study.best_params, 'best_value': study.best_value}, f)
