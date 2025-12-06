#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bot_trade import make_decision, reset_history, MODULE_HISTORY

# Reset module history
reset_history()

# Simulate judge sequential calls
prices = [100, 105, 110, 108, 112, 115]
print('Simulating judge:')
for i, p in enumerate(prices):
    print(i, p, make_decision(i, p))

print('\nReset local: now use explicit history param')
reset_history()
local_history = []
for i, p in enumerate(prices):
    decision = make_decision(i, p, history=local_history)
    print(i, p, decision)

print('\nModule history length:', len(MODULE_HISTORY))
