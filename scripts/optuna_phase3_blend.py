#!/usr/bin/env python3
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import optuna
import pandas as pd
import math
import importlib.util
from scoring.scoring import get_local_score


def load_bot(path_to_bot: str):
    spec = importlib.util.spec_from_file_location('m', path_to_bot)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evaluate_params_on_datasets(datasets, params, transaction_fees=0.0005, slippage=0.0001, fixed_cost=0.0):
    # Load root bot implementation which is multi-asset capable
    bot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bot_trade.py')
    bot = load_bot(bot_path)
    scores = []
    for csv in datasets:
        df = pd.read_csv(csv, index_col=0)
        df['Cash'] = 1
        history = []
        positions = []
        # Support both multi-asset (Asset A + Asset B) and single-asset (Asset B)
        if 'Asset A' in df.columns and 'Asset B' in df.columns:
            for epoch, row in df.iterrows():
                decision = bot.make_decision(int(epoch), float(row['Asset A']), float(row['Asset B']), strategy='blended', params=params, history=history)
                decision['epoch'] = int(epoch)
                positions.append(decision)
        else:
            for epoch, row in df.iterrows():
                decision = bot.make_decision(int(epoch), float(row['Asset B']), strategy='blended', params=params, history=history)
            decision['epoch'] = int(epoch)
            positions.append(decision)
        positions_df = pd.DataFrame(positions).set_index('epoch')
        try:
            res = get_local_score(prices=df, positions=positions_df, transaction_fees=transaction_fees, slippage=slippage, fixed_cost_per_trade=fixed_cost)
            scores.append(res['scores']['base_score'])
        except Exception:
            scores.append(float('nan'))
    return float(pd.Series(scores).dropna().mean()) if len([s for s in scores if not math.isnan(s)]) > 0 else float('nan')


def objective(trial: optuna.Trial, datasets, transaction_fees=0.0005, slippage=0.0001, fixed_cost=0.0):
    blend = trial.suggest_float('blend', 0.3, 1.0)
    short = trial.suggest_int('short', 2, 10)
    long = trial.suggest_int('long', 15, 60)
    if long <= short:
        return float('nan')
    threshold = trial.suggest_float('threshold', 0.005, 0.05)
    vol_window = trial.suggest_int('vol_window', 5, 20)
    vol_multiplier = trial.suggest_float('vol_multiplier', 0.0, 2.0)
    params = {'blend': blend, 'sub_params': {'short': short, 'long': long, 'threshold': threshold, 'vol_window': vol_window, 'vol_multiplier': vol_multiplier}}
    score = evaluate_params_on_datasets(datasets, params, transaction_fees, slippage, fixed_cost)
    return score


def main():
    parser = argparse = __import__('argparse').ArgumentParser()
    parser.add_argument('--datasets-dir', default='data')
    parser.add_argument('--n-trials', type=int, default=60)
    parser.add_argument('--transaction-fees', type=float, default=0.0005)
    parser.add_argument('--slippage', type=float, default=0.0001)
    parser.add_argument('--fixed-cost', type=float, default=0.0)
    args = parser.parse_args()

    datasets = [os.path.join(args.datasets_dir, f) for f in os.listdir(args.datasets_dir) if f.endswith('.csv')]
    datasets = sorted(datasets)
    print('Using datasets:', datasets)

    storage_path = 'sqlite:///experiments/optuna/optuna_phase3_blend.db'
    study = optuna.create_study(direction='maximize', storage=storage_path, study_name='phase3_blend', load_if_exists=True)
    def wrapped(trial):
        return objective(trial, datasets, transaction_fees=args.transaction_fees, slippage=args.slippage, fixed_cost=args.fixed_cost)
    study.optimize(wrapped, n_trials=args.n_trials)
    print('Best params:', study.best_params)
    print('Best value:', study.best_value)
    # Save trials table
    df = study.trials_dataframe()
    os.makedirs('experiments/optuna', exist_ok=True)
    df.to_csv('experiments/optuna/phase3_blend_trials.csv', index=False)
    os.makedirs('experiments/optuna', exist_ok=True)
    with open('experiments/optuna/phase3_blended_best.json', 'w') as f:
        json.dump({'best_params': study.best_params, 'best_value': study.best_value}, f)


if __name__ == '__main__':
    main()
