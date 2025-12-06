#!/usr/bin/env python3
import pandas as pd
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bot_trade import make_decision, reset_history
from scoring.scoring import get_local_score

df = pd.read_csv('data/asset_b_train.csv', index_col=0)
df['Cash'] = 1

# run blended tuned alias
reset_history()
positions_blended_tuned = []
for idx, row in df.iterrows():
    d = make_decision(int(idx), float(row['Asset B']), strategy='blended_tuned')
    d['epoch'] = int(idx)
    positions_blended_tuned.append(d)

pos_bt = pd.DataFrame(positions_blended_tuned).set_index('epoch')
res_bt = get_local_score(prices=df, positions=pos_bt, transaction_fees=0.0005)

# run blended with explicit params
reset_history()
positions_blended = []
params = {"blend":0.5004034618154071, "sub_params": {"short":6, "long":39, "threshold":0.010008502229651432, "vol_window":9, "vol_multiplier":0.6940552234824028}}
for idx, row in df.iterrows():
    d = make_decision(int(idx), float(row['Asset B']), strategy='blended', params=params)
    d['epoch'] = int(idx)
    positions_blended.append(d)
pos_b = pd.DataFrame(positions_blended).set_index('epoch')
res_b = get_local_score(prices=df, positions=pos_b, transaction_fees=0.0005)

print('blended tuned base score', res_bt['scores']['base_score'], 'pnl', res_bt['stats']['cumulative_return'])
print('blended custom base score', res_b['scores']['base_score'], 'pnl', res_b['stats']['cumulative_return'])

# check the first 10 positions for differences
print('\nDifferences (first 10):')
for i in range(10):
    if abs(pos_bt.iloc[i]['Asset B'] - pos_b.iloc[i]['Asset B']) > 1e-6:
        print(i, pos_bt.iloc[i]['Asset B'], pos_b.iloc[i]['Asset B'])
