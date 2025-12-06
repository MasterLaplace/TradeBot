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


def walk_forward_mean_blend_mo(blend, sub_params, n_splits=3, transaction_fees=0.0001, slippage=0.0, fixed_cost_per_trade=0.0):
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
            d = make_decision(int(idx), float(row['Asset B']), strategy='blended', params={'blend': blend, 'sub_params': sub_params}, history=history)
            d['epoch'] = int(idx)
            positions.append(d)
        positions_df = pd.DataFrame(positions).set_index('epoch')
        prices_sub = df.iloc[start_test:end_test].copy()
        positions_sub = positions_df.iloc[start_test:end_test].copy()
        prices_sub.index = range(len(prices_sub))
        positions_sub.index = range(len(positions_sub))
        try:
            res = get_local_score(prices=prices_sub, positions=positions_sub, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost_per_trade)
            stats = res['stats']
            scores.append({'base': res['scores']['base_score'], 'sharpe': stats['sharpe_ratio'], 'mdd': stats['max_drawdown']})
        except Exception:
            scores.append({'base': float('nan'), 'sharpe': float('nan'), 'mdd': float('nan')})
    df_scores = pd.DataFrame(scores).dropna()
    if df_scores.empty:
        return float('nan')
    # Combine metrics: base_score (weighted), sharpe (scaled), drawdown penalty
    # Normalization heuristics: sharpe normally 0-3; mdd 0-1
    base_mean = df_scores['base'].mean()
    sharpe_mean = df_scores['sharpe'].mean()
    mdd_mean = df_scores['mdd'].mean()
    # Score composition
    mo_score = base_mean + 0.15 * max(sharpe_mean, 0) - 0.25 * abs(mdd_mean)
    return float(mo_score)


def objective(trial: optuna.Trial, transaction_fees: float = 0.0001, slippage: float = 0.0, fixed_cost_per_trade: float = 0.0):
    blend = trial.suggest_float('blend', 0.5, 1.0)
    short = trial.suggest_int('short', 2, 10)
    long = trial.suggest_int('long', 15, 60)
    if long <= short:
        return float('nan')
    threshold = trial.suggest_float('threshold', 0.01, 0.1)
    vol_window = trial.suggest_int('vol_window', 5, 20)
    vol_multiplier = trial.suggest_float('vol_multiplier', 0.0, 2.0)
    sub_params = {'short': short, 'long': long, 'threshold': threshold, 'vol_window': vol_window, 'vol_multiplier': vol_multiplier}
    score = walk_forward_mean_blend_mo(blend, sub_params, n_splits=5, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost_per_trade)
    return score


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=200)
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
    with open('experiments/optuna/blended_mo_optuna_best.json', 'w') as f:
        json.dump({'best_params': study.best_params, 'best_value': study.best_value}, f)
