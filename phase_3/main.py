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

def validate_decision(decision: dict) -> bool:
    expected_keys = {'Asset A', 'Asset B', 'Cash'}
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
    output = []

    parser = argparse.ArgumentParser(description="Run bot on a csv with Asset A and Asset B")
    parser.add_argument('path_csv')
    parser.add_argument('--strategy', default='blended')
    parser.add_argument('--strategy-params', default=None)
    parser.add_argument('--show-graph', action='store_true')
    args = parser.parse_args()
    path_csv = args.path_csv
    prices = find_csv_file(path_csv=path_csv)

    # parse params
    strategy_params = None
    if args.strategy_params:
        try:
            strategy_params = json.loads(args.strategy_params)
        except Exception:
            strategy_params = None
    for index, row in prices.iterrows():
        decision = decision_generator(int(index), float(row['Asset A']), float(row['Asset B']), strategy=args.strategy, params=strategy_params)
        if not validate_decision(decision):
            raise ValueError(f"Décision invalide: {decision}")
        decision['epoch'] = int(index)
        output.append(decision)
    positions = pd.DataFrame(output).set_index("epoch")
    local_score = get_local_score(prices=prices, positions=positions)
    show_result(local_score, is_show_graph=args.show_graph)

if __name__ == "__main__":
    main()