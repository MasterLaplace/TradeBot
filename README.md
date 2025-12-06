# Bot Trade Project

Ce dépôt contient un bot de trading simple (`bot_trade.py`) avec un backtester et un script principal (`main.py`) permettant d'exécuter le bot sur un CSV de prix local (ex : `data/asset_b_train.csv`).

## Objectif

- Fournir un ensemble minimal pour développer, tester et améliorer des stratégies de trading et mesurer leurs performances avec des métriques financières (Sharpe, PnL, Max Drawdown...).
- Maintenir un journal de recherche évolutif ("Research Log") pour consigner hypothèses, tests, résultats et améliorations.

## Installation & Setup

Utilisez le script `setup_env.sh` qui crée un environnement virtuel, met à jour `pip` et installe les dépendances listées dans `requirement.txt`.

```bash
./setup_env.sh
```

Si vous préférez faire les opérations manuellement:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
```

## Usage

- Executer le bot sur un CSV :

```bash
venv/bin/python main.py data/asset_b_train.csv
```

- Afficher la figure (interface graphique requise - X11 / Wayland) :

```bash
venv/bin/python main.py data/asset_b_train.csv --show-graph
```

- Sauvegarder la figure dans un fichier (utile en mode headless / serveur) :

```bash
venv/bin/python main.py data/asset_b_train.csv --save-graph result.png

## Packaging for online judge / production

If you want to submit your bot to an online judge that simply zips your `bot_trade.py` file and calls `make_decision(epoch, price)` sequentially, follow these constraints:

- Keep `make_decision(epoch, price)` as the main entry point; the platform will only call it with these two arguments.
- We provide an internal `MODULE_HISTORY` (in `bot_trade.py`) that stores calls across the single process run. It ensures your bot retains state across calls (e.g., previous prices) without relying on external files or variables.
- If you need to reset history mid-run for local tests, use `reset_history()` (available in `bot_trade.py`). The platform won't call it.
- The `bot_trade.py` file should be self-contained: no external configuration files required. The platform will not have `main.py` or any scripts.

To create the ZIP for submission (example):

```bash
zip submission.zip bot_trade.py
```

Tip: If you want to test the same zipped file locally, you can simulate the judge by writing a small script that imports the function and calls it sequentially, or just run `venv/bin/python main.py data/asset_b_train.csv` using the same file.
```

## Développement

- Points d'entrée :
  - `main.py` : orchestration (lecture CSV -> appel du bot -> validation -> backtest -> affichage)
  - `bot_trade.py` : votre bot à développer/évouler
  - `scoring/scoring.py` : backtester et calculs de métriques

- Exemple rapide pour tester la logique du bot :

```bash
venv/bin/python - <<'PY'
from bot_trade import make_decision
print('Sample decisions:')
for i,p in enumerate([100,105,110,108,112]):
    print(i, p, make_decision(i,p))
PY
```

## Research Log & Experimental Workflow (section évolutive)

> Cette section est le coeur du README. Elle centralisera les hypothèses, tests, résultats, analyses et les solutions retenues.

Règles simples pour mener une expérience structurée (méthode scientifique appliquée au trading) :

1. Hypothèse : décrire en une phrase claire ce que vous voulez tester ("Ajouter un stop-loss devrait augmenter Sharpe sans trop réduire PnL").
2. Setup : préciser les changements code, les paramètres testés, jeu de données et les commandes utilisées.
3. Métriques : lister ce que vous mesurerez (Sharpe, cumulative_return, max_drawdown, time_in_market, etc.).
4. Résultat : enregistrer les résultats (table + figures) et comparer à la baseline.
5. Conclusion & Next steps : si l’amélioration est significative, garder la modification, sinon proposer une nouvelle hypothèse.

### Exemple d'entrée d'expérience

- ID : EXP-001
- Date : 2025-12-05
- Hypothèse : "Augmenter l'allocation à Asset B après une hausse de prix (change : baseline 0.5 -> 0.7) améliore le Sharpe sur ce dataset"
- Changes : `bot_trade.py` (ajout d'une règle simple d'allocation selon delta)
- Commande :

```bash
venv/bin/python main.py data/asset_b_train.csv --save-graph exp-001.png
```

- Résultat : Sharpe = 1.269, PnL = 202%, Max Drawdown = -x% (voir `exp-001.md`)
- Conclusion : traitement empirique, Score de base amélioré mais nécessite un test sur d'autres jeux de données et avec coûts de transaction supérieurs pour robustesse.

#### Journalisation automatisée

Vous trouverez des scripts `run_experiment.sh` et un dossier `experiments/` pour standardiser la création d’un nouvel essai (génération automatique d’un dossier timestamp et d’un fichier `results.md`).

---

## Bonnes pratiques et idées d'amélioration (non exhaustif)

- Normalisation / Feature engineering : ajoutez moyennes mobiles (SMA/EMA), signaux de momentum, volabilité (ATR), volume features, etc.
- Position sizing : appliquez des règles de taille proportionnelles à la volatilité (volatility scaling), Kelly criterion pour sizing, ou gestion basée sur la VaR/CVaR.
- Risk management : implémenter stop-loss / take-profit, cap sur exposition, time-in-market controls.
- Robustesse des backtests : utilises walk-forward validation, transaction costs, slippage, latences, plus de datasets (assets) pour réduire sur-optimisation.
- Hyperparameter tuning : optuna / scikit-optimize pour calibrer paramètres et valider via crossvalidation/walk-forward.
- Logs et traçabilité : git branches pour chaque expérience, fichier `experiments/EXP-XXX.md` pour reproduire chaque essai.

## Suggestions d'expériences initiales (à tester)

- Ajouter un égaliseur simple (ex: moyenne mobile 10/30) comme signal d'entrée et comparer avec la baseline.
- Tester la taille de position proportionnelle à l'inverse de la volatilité (vol scaling).
- Implémenter stop-loss et take-profit et un coût de transaction plus élevé (pour simuler slippage) afin de mesurer la robustesse.
- Faire de la validation walk-forward (training on a rolling window) pour vérifier la sur-optimisation.

## Ressources utiles (référence / lecture)

- Backtesting best practices (intro) — QuantStart: https://www.quantstart.com/articles/Backtesting-Best-Practices/
- Walk-forward analysis — Investopedia (voir notion d'analyse ex post / look-forward)
- Sharpe ratio, Sortino, et métriques de risk-adjusted returns — Investopedia: https://www.investopedia.com/terms/s/sharperatio.asp
- Hyperparameter tuning: Optuna — https://optuna.org/

## Automatisation et prochaines étapes

- Pour conserver une bonne traçabilité, ouvrez une branche git par expérience (`git checkout -b exp/EXP-xxx`) et y appliquez vos changements.
- Utilisez `./run_experiment.sh` pour automatiser la sauvegarde des artefacts (image, logs, result template).
- Je peux vous aider à ajouter une intégration CI (GitHub Actions / GitLab CI) pour exécuter automatiquement toutes les expériences de référence et publier un rapport centralisé.

---

Merci de m'avoir confié le soin d'aider ce projet — je me chargerai d'ajouter des entrées dans la section Research Log après chaque expérience validée (manuelle ou automatisée). Dites-moi quelle première expérience vous voulez que j'ajoute et j'exécuterai la boucle complète : hypothèse -> implémentation -> test -> résultat -> conclusion.

## Aide / Support

Contactez-moi pour automatiser des recherches (features), écrire des tests unitaires, ou intégrer CI pour l’exécution automatisée des expériences.

---

Ce README contient une section "Research Log" propre à votre workflow. Je peux exécuter et documenter automatiquement de nouvelles expériences, et je mettrai à jour cette section de façon incrémentale à chaque expérience. N'hésitez pas à demander si vous préférez un autre format (JSON, YAML, DB). 
