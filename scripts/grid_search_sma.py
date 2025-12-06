#!/usr/bin/env python3
import subprocess
import json
import os
import itertools

# Simple grid search for SMA parameters
SHORTS = [3,5,10]
LONGS = [20,30,50]
CSV = 'data/asset_b_train.csv'

BEST = (None, -1.0)
for short, long in itertools.product(SHORTS, LONGS):
    if long <= short:
        continue
    params = json.dumps({'short': short, 'long': long})
    out = subprocess.run(['./run_experiment.sh', CSV, '--save-graph', f'sma-{short}-{long}.png', '--strategy', 'sma', '--strategy-params', params], capture_output=True, text=True)
    # find experiments folder created by run_experiment.sh (last created)
    # try to capture the experiment dir from output
    exp_dir = None
    for line in out.stdout.splitlines():
        if 'Experiment saved in:' in line:
            exp_dir = line.strip().split(':', 1)[1].strip()
            break
    if not exp_dir:
        experiments = sorted([d for d in os.listdir('experiments') if d.startswith('exp-')])
        if not experiments:
            print('No experiments found')
            continue
        exp_dir = os.path.join('experiments', experiments[-1])
    # parse the results.md (update from console.log if needed)
    import subprocess
    subprocess.run(['python3', 'scripts/parse_results.py', exp_dir], check=True)
    results_md = os.path.join(exp_dir, 'results.md')
    base_score = None
    with open(results_md, 'r') as f:
        content = f.read()
        import re
        m = re.search(r'- Base Score:\s*([0-9.\-]+)', content)
        if m:
            base_score = float(m.group(1))
    print(f'SMA {short}/{long} -> base_score={base_score}')
    if base_score is not None and base_score > BEST[1]:
        BEST = ((short, long, base_score, exp_dir), base_score)

print('Best SMA:', BEST)
