#!/usr/bin/env python3
"""
Quick grid search across blends and total exposure caps to find a practical tradeoff between base_score and drawdown.
It runs the runner per dataset and aggregates per-dataset stats.
"""
import os
import sys
import json
import argparse
import subprocess
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def run_combo(blend, max_total_exposure, datasets_dir='data'):
    datasets = [os.path.join(datasets_dir, f) for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    results = []
    for d in datasets:
        cmd = ["venv/bin/python", "scripts/multi_dataset_runner.py", "--datasets-dir", os.path.dirname(d), "--strategy", "blended", "--strategy-params", str({"blend": blend, "sub_params": {"short":10,"long":27,"threshold":0.0452921667498627,"vol_window":13,"vol_multiplier":0.7322341609310995}}), "--transaction-fees", "0.0001", "--slippage", "0.0"]
        # Use subprocess and capture stdout
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout
        # The runner writes a CSV results file; we can read experiments/eval/multi_dataset_results.csv
        try:
            df = pd.read_csv('experiments/eval/multi_dataset_results.csv')
            # Filter for our dataset
            row = df[df['csv'].str.endswith(os.path.basename(d))].iloc[0]
            results.append({'csv': d, 'base_score': float(row['base_score']), 'mdd': float(row['mdd'])})
        except Exception as e:
            results.append({'csv': d, 'base_score': None, 'mdd': None, 'error': str(e)})
    return results


def main():
    blends = [0.99, 0.95, 0.9, 0.85]
    caps = [1.0, 0.9, 0.8, 0.7]
    all_results = []
    for b in blends:
        for c in caps:
            res = run_combo(b, c)
            # compute metrics: min base_score and avg mdd
            base_scores = [r['base_score'] for r in res if r['base_score'] is not None]
            mdds = [r['mdd'] for r in res if r['mdd'] is not None]
            min_base = min(base_scores) if base_scores else None
            mean_base = sum(base_scores)/len(base_scores) if base_scores else None
            mean_mdd = sum(mdds)/len(mdds) if mdds else None
            all_results.append({'blend': b, 'max_total_exposure': c, 'min_base': min_base, 'mean_base': mean_base, 'mean_mdd': mean_mdd, 'per_dataset': res})
            print('Blend', b, 'cap', c, 'min_base', min_base, 'mean_base', mean_base, 'mean_mdd', mean_mdd)
    with open('experiments/eval/grid_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == '__main__':
    main()
