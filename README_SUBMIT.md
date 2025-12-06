Final Bot Submission - Ok_v3
================================

Files included:
- bot_trade.py (final strategy: blended_robust_ensemble as default)
- main.py (runner for CSVs)
- scoring/ (backtest & score tools)
- scripts/ (evaluation, optuna, and simulator scripts)
- requirement.txt (python dependencies)
- setup_env.sh (setup environment)

How to run (quick example):
1) Create venv & install:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirement.txt
2) Run main (single csv example):
    venv/bin/python main.py data/asset_b_train.csv --strategy default --transaction-fees 0.0001 --slippage 0.0
3) Run the prod simulator:
    venv/bin/python scripts/prod_run_simulator.py --dataset data/asset_b_train.csv --out experiments/eval/final_prod_b_ensemble.csv --transaction-fees 0.0001 --slippage 0.0 --strategy default

Notes:
- Default strategy in `bot_trade.py` is `blended_robust_ensemble` (ensembled top5 of robust candidates).
- A safer strategy variant is `blended_robust_safe` (cap exposure) if you need lower drawdowns.
- The `scripts/` folder includes useful runners & scripts for evaluation, ensemble simulation and optimization.

Contact:
- If you want further changes (dockerize, CI, or more tuning), ask and I’ll implement quickly.
