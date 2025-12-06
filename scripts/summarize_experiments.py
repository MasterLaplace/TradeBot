#!/usr/bin/env python3
import os
import re
import glob
import math

rows = []
for d in sorted(glob.glob('experiments/exp-*')):
    results_md = os.path.join(d, 'results.md')
    if not os.path.exists(results_md):
        continue
    text = open(results_md).read()
    # Try to extract base score
    m = re.search(r'Base Score:\s*([0-9.\-]+)', text)
    base_score = float(m.group(1)) if m else None
    # Extract scenario name from folder
    name = os.path.basename(d)
    rows.append({'experiment': name, 'base_score': base_score, 'path': d})

if not rows:
    print('No experiments found')
    exit(0)

rows_sorted = sorted(rows, key=lambda r: (r['base_score'] is None, -(r['base_score'] or 0)))
lines = ["| experiment | base_score | path |", "|---|---:|---|"]
for r in rows_sorted:
    lines.append(f"| {r['experiment']} | {'' if r['base_score'] is None else r['base_score']} | {r['path']} |")
out = '\n'.join(lines)
print(out)
os.makedirs('experiments/summary', exist_ok=True)
with open('experiments/summary/top_experiments.md', 'w') as f:
    f.write(out)
print('Saved: experiments/summary/top_experiments.md')
