#!/usr/bin/env bash
set -e

# Run baseline
./run_experiment.sh data/asset_b_train.csv --save-graph baseline.png --strategy baseline

# Run basic variants
./run_experiment.sh data/asset_b_train.csv --save-graph sma.png --strategy sma --strategy-params '{"short":5,"long":20}'
./run_experiment.sh data/asset_b_train.csv --save-graph volscale.png --strategy volscale --strategy-params '{"base":0.6,"vol_window":10}'
./run_experiment.sh data/asset_b_train.csv --save-graph stoploss.png --strategy stoploss --strategy-params '{"base":0.7,"threshold":0.05}'

# Run SMA grid
./scripts/grid_search_sma_simple.py
# Run volscale grid
./scripts/grid_search_volscale.py
# Run stoploss grid
./scripts/grid_search_stoploss.py

# Summarize
python3 scripts/summarize_experiments.py

echo "Full suite completed. Consult experiments/summary/top_experiments.md"
