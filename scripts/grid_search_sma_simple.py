#!/usr/bin/env python3
import subprocess
import json
import os
import itertools
import datetime
import re

SHORTS = [3,5,10]
LONGS = [20,30,50]
CSV = 'data/asset_b_train.csv'

BEST = (None, -1.0)
for short, long in itertools.product(SHORTS, LONGS):
    if long <= short:
        continue
    params = json.dumps({'short': short, 'long': long})
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    out_dir = f'experiments/exp-{timestamp}-sma-{short}-{long}'
    os.makedirs(out_dir, exist_ok=True)
    save_path = f'{out_dir}/sma-{short}-{long}.png'
    cmd = ['venv/bin/python', 'main.py', CSV, '--strategy', 'sma', '--strategy-params', params, '--save-graph', save_path]
    print('Running:', ' '.join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(f'{out_dir}/console.log', 'w') as f:
        f.write(p.stdout)
        f.write('\n')
        f.write(p.stderr)
    # parse base score
    m = re.search(r'Base Score:\s*([0-9.\-]+)', p.stdout)
    base_score = float(m.group(1)) if m else None
    print(f'SMA {short}/{long} -> base_score={base_score}')
    # record in results.md
    with open(f'{out_dir}/results.md', 'w') as f:
        f.write(f"# {out_dir}\n\nBase Score: {base_score}\n")
    if base_score is not None and base_score > BEST[1]:
        BEST = ((short, long, base_score, out_dir), base_score)

print('Best SMA found:', BEST)
