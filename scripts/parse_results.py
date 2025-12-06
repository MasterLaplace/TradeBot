#!/usr/bin/env python3
import re
import sys
import os

# Parse the console.log to extract Sharpe Score, PnL (Brut PnL) and Base Score
PATTERNS = {
    'sharpe': re.compile(r"Sharpe Score:\s*([0-9.-]+)"),
    'pnl': re.compile(r"Brut PnL:\s*([0-9.-]+)%"),
    'base': re.compile(r"⭐ Base Score:\s*([0-9.-]+)"),
    'mdd': re.compile(r"Max Drawdown Score:\s*([0-9.-]+)"),
}


def parse_console_log(path):
    data = {'sharpe': None, 'pnl': None, 'base': None, 'mdd': None}
    with open(path, 'r') as f:
        content = f.read()
    for key, pat in PATTERNS.items():
        m = pat.search(content)
        if m:
            try:
                data[key] = float(m.group(1))
            except Exception:
                data[key] = None
    return data


def update_results_md(exp_dir):
    console_log = os.path.join(exp_dir, 'console.log')
    results_md = os.path.join(exp_dir, 'results.md')
    if not os.path.exists(console_log) or not os.path.exists(results_md):
        print('Missing files', console_log, results_md)
        return
    data = parse_console_log(console_log)
    # read results.md
    with open(results_md, 'r') as f:
        content = f.read()
    # fill placeholders
    content = content.replace('- Sharpe: ', f"- Sharpe: {data['sharpe'] if data['sharpe'] is not None else ''}")
    content = content.replace('- PnL: ', f"- PnL: {data['pnl'] if data['pnl'] is not None else ''}%")
    content = content.replace('- Max Drawdown: ', f"- Max Drawdown: {data['mdd'] if data['mdd'] is not None else ''}")
    with open(results_md, 'w') as f:
        f.write(content)
    print(f'Updated {results_md} with metrics: {data}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: parse_results.py <experiment_dir>')
        sys.exit(1)
    update_results_md(sys.argv[1])
