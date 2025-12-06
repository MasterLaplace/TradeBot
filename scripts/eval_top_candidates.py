#!/usr/bin/env python3
import json
import os
from subprocess import check_call

TOP5='experiments/optuna/phase3_blend_top5.json'
OUTDIR='experiments/eval/top5'
EXE='venv/bin/python'
RUNNER='scripts/multi_dataset_runner.py'

fees=[0.0001,0.0005,0.001]
slippages=[0.0,0.0001,0.0005]

os.makedirs(OUTDIR, exist_ok=True)

with open(TOP5,'r') as f:
    top5=json.load(f)

for i, candidate in enumerate(top5):
    params=candidate['params']
    params_json=json.dumps({
        'blend':params['blend'],
        'sub_params':{'short':params['short'],'long':params['long'],'threshold':params['threshold'],'vol_window':params['vol_window'],'vol_multiplier':params['vol_multiplier']}
    })
    for fee in fees:
        for slip in slippages:
            out_csv=os.path.join(OUTDIR,f'top{i+1}_fee{fee}_slip{slip}.csv')
            print('Running candidate', i+1, 'fee', fee, 'slip', slip)
            cmd=[EXE, RUNNER, '--datasets-dir', 'data', '--strategy', 'blended', '--strategy-params', params_json, '--transaction-fees', str(fee), '--slippage', str(slip)]
            env=os.environ.copy()
            check_call(cmd, env=env)
            # copy results
            check_call(['cp','experiments/eval/multi_dataset_results.csv', out_csv])

print('Done')
