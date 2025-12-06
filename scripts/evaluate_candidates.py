#!/usr/bin/env python3
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from scoring.scoring import get_local_score
from bot_trade import reset_history, make_decision

CSV = 'data/asset_b_train.csv'

def run_full(strategy, params=None, fees=0.0005, slippage=0.0, fixed_cost=0.0):
    df = pd.read_csv(CSV, index_col=0)
    df['Cash'] = 1
    reset_history()
    positions = []
    history = []
    for idx, row in df.iterrows():
        d = make_decision(int(idx), float(row['Asset B']), strategy=strategy, params=params or {}, history=history)
        d['epoch'] = int(idx)
        positions.append(d)
    positions_df = pd.DataFrame(positions).set_index('epoch')
    res = get_local_score(prices=df, positions=positions_df, transaction_fees=fees, slippage=slippage, fixed_cost_per_trade=fixed_cost)
    return res

def main():
    # load optuna best results
    candidates = []
    # baseline
    candidates.append({'name': 'baseline', 'strategy': 'baseline', 'params': None})
    # composite best
    try:
        with open('experiments/optuna/composite_optuna_best.json') as f:
            c = json.load(f)
            candidates.append({'name': 'composite_opt', 'strategy': 'composite', 'params': c['best_params']})
    except Exception:
        pass
    # blended tuned and mo
    try:
        with open('experiments/optuna/blended_optuna_best.json') as f:
            b = json.load(f)
            candidates.append({'name': 'blended_opt', 'strategy': 'blended', 'params': {'blend': b['best_params'].get('blend'), 'sub_params': {k: b['best_params'][k] for k in ['short','long','threshold','vol_window','vol_multiplier']}}})
    except Exception:
        pass
    try:
        with open('experiments/optuna/blended_mo_optuna_best.json') as f:
            bmo = json.load(f)
            p = bmo['best_params']
            candidates.append({'name': 'blended_mo_opt', 'strategy': 'blended', 'params': {'blend': p.get('blend'), 'sub_params': {k: p[k] for k in ['short','long','threshold','vol_window','vol_multiplier']}}})
    except Exception:
        pass
    try:
        with open('experiments/optuna/ensemble_optuna_best.json') as f:
            e = json.load(f)
            candidates.append({'name': 'ensemble_opt', 'strategy': 'ensemble', 'params': e['best_params']})
    except Exception:
        pass

    # Evaluate each candidate for default fee and slippage
    results = []
    for cand in candidates:
        res = run_full(cand['strategy'], cand['params'], fees=0.0005, slippage=0.0001, fixed_cost=0.0)
        stats = res['stats']
        sc = res['scores']
        results.append({'name': cand['name'], 'strategy': cand['strategy'], 'base_score': sc['base_score'], 'pnl%': stats['cumulative_return']*100, 'sharpe': stats['sharpe_ratio'], 'mdd': stats['max_drawdown']})
        print(f"Candidate {cand['name']}: base={sc['base_score']:.4f}, pnl={stats['cumulative_return']*100:.2f}%, sh={stats['sharpe_ratio']:.3f}, mdd={stats['max_drawdown']:.3f}")

    # Save to CSV
    df_out = pd.DataFrame(results)
    os.makedirs('experiments/eval', exist_ok=True)
    df_out.to_csv('experiments/eval/candidates_summary.csv', index=False)
    print('Saved results to experiments/eval/candidates_summary.csv')

if __name__ == '__main__':
    main()
