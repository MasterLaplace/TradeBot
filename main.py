#! /usr/bin/env python3

import csv
import json
import os
import sys
import argparse

# Empêcher la création de __pycache__
sys.dont_write_bytecode = True


from scoring.scoring import get_local_score, show_result
import pandas as pd
from bot_trade import make_decision as decision_generator
import matplotlib.pyplot as plt


def find_csv_file(path_csv: str) -> pd.DataFrame:
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Le fichier CSV {path_csv} n'existe pas")
    prices_list = [
        pd.read_csv(path_csv, index_col=0)
    ]
    prices = pd.concat(prices_list, axis=1)
    prices["Cash"] = 1
    return prices

def validate_decision(decision: dict, expected_keys=None) -> bool:
    expected_keys = expected_keys or {'Asset B', 'Cash'}
    if set(decision.keys()) != expected_keys:
        print(f"ERREUR: Les clés attendues sont {expected_keys}, mais reçu {set(decision.keys())}")
        return False

    for key, value in decision.items():
        if not isinstance(value, (int, float)):
            print(f"ERREUR: La valeur pour '{key}' n'est pas numérique: {value}")
            return False
        if value < 0 or value > 1:
            print(f"ERREUR: La valeur pour '{key}' doit être entre 0 et 1, reçu: {value}")
            return False

    total = sum(decision.values())
    if abs(total - 1.0) > 0.00001:
        print(f"ERREUR: La somme des allocations doit être égale à 1, mais vaut {total}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Run the bot and backtest it on a CSV file")
    parser.add_argument("path_csv", help="Path to CSV file with prices")
    parser.add_argument("--show-graph", action="store_true", help="Open an interactive window to show the PnL graph")
    parser.add_argument("--save-graph", nargs="?", const="pnl.png", help="Save the PnL graph to a file. Optionally pass filename (default: pnl.png)")
    parser.add_argument("--strategy", choices=["default","baseline","sma","volscale","stoploss","sma_stoploss","sma_volfilter","sma_smooth_stop","adaptive_baseline","blended","blended_tuned","blended_robust","blended_robust_ensemble","blended_mo_tuned","composite","ensemble"], default="blended_robust_ensemble", help="Which strategy to use for the bot (use 'default' to use the bot's internal default strategy)")
    parser.add_argument("--strategy-params", nargs='?', help="JSON string of strategy parameters, e.g., '{\"short\":10,\"long\":30}'")
    parser.add_argument("--transaction-fees", type=float, default=0.0001, help="Proportional transaction fees to apply in backtest (e.g. 0.0005 for 0.05%)")
    parser.add_argument("--slippage", type=float, default=0.0, help="Proportional slippage modeled as percent of traded notional, e.g. 0.001 for 0.1%")
    parser.add_argument("--fixed-cost-per-trade", type=float, default=0.0, help="Fixed cost applied to each executed trade (absolute) e.g. 0.02 per trade)")
    args = parser.parse_args()

    output = []
    path_csv = args.path_csv
    prices = find_csv_file(path_csv=path_csv)

    # history passed to bot so strategies can compute moving averages etc.
    history = []
    strategy_params = None
    if args.strategy_params:
        try:
            strategy_params = json.loads(args.strategy_params)
        except Exception:
            strategy_params = None

    is_multi = 'Asset A' in prices.columns and 'Asset B' in prices.columns
    for index, row in prices.iterrows():
        if is_multi:
            decision = decision_generator(int(index), float(row['Asset A']), float(row['Asset B']), strategy=args.strategy, params=strategy_params, history=history)
        else:
            decision = decision_generator(int(index), float(row['Asset B']), strategy=args.strategy, params=strategy_params, history=history)
        expected = {'Asset A', 'Asset B', 'Cash'} if is_multi else {'Asset B', 'Cash'}
        if not validate_decision(decision, expected):
            raise ValueError(f"Décision invalide: {decision}")
        decision['epoch'] = int(index)
        output.append(decision)
    positions = pd.DataFrame(output).set_index("epoch")
    local_score = get_local_score(
        prices=prices,
        positions=positions,
        transaction_fees=args.transaction_fees,
        slippage=args.slippage,
        fixed_cost_per_trade=args.fixed_cost_per_trade,
    )
    show_result(local_score, is_show_graph=args.show_graph, save_path=args.save_graph)
    if not args.show_graph and not args.save_graph:
        print("\033[91mpour afficher le graphique, utilisez la commande --show-graph: ./main.py <path_to_csv> --show-graph\033[0m")

if __name__ == "__main__":
    main()
