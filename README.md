# 🤖 TradeBot v2.0

Bot de trading algorithmique avec architecture SOLID et interface unifiée.

## 🎯 Résultats

Sur 90 jours de données crypto réelles (BTC/ETH), nos stratégies surperforment significativement :

| Stratégie | Return | Alpha vs 50/50 |
|-----------|--------|----------------|
| **adaptive_trend** | -12.14% | **+12.30%** |
| **safe_profit** | -12.60% | **+11.84%** |
| **composite** | -14.75% | **+9.68%** |
| 50/50 Buy & Hold | -24.44% | baseline |
| Buy & Hold BTC | -19.53% | - |
| Buy & Hold ETH | -29.34% | - |

> La stratégie `safe_profit` combine plusieurs indicateurs pour une performance robuste avec un drawdown limité.

## 🏗️ Architecture

```
tradebot.py              # 🎯 Point d'entrée unique
src/
├── __init__.py
├── cli/                 # Interface ligne de commande
│   ├── main.py          # Parser argparse (exposes `tradebot` CLI)
│   └── commands.py      # Handlers des commandes
├── core/                # Modèles de domaine
│   └── models.py        # Price, Allocation, Portfolio
├── data/                # Sources de données
│   └── sources.py       # CSV, Binance REST
├── engine/              # Moteurs de trading
│   ├── backtest.py      # Backtesting historique
│   └── paper_trading.py # Paper trading temps réel
├── reporting/           # Génération de rapports
│   └── reports.py       # Markdown, PNG charts
└── strategies/          # Stratégies de trading
    └── base.py          # safe_profit, adaptive_trend...
```

**Principes SOLID appliqués :**
- **S**ingle Responsibility : chaque module a un rôle unique
- **O**pen/Closed : stratégies extensibles sans modification
- **L**iskov Substitution : interfaces Protocol cohérentes
- **I**nterface Segregation : interfaces spécialisées
- **D**ependency Inversion : dépendances vers abstractions

## 🚀 Installation

```bash
# Cloner et créer l'environnement
git clone <repo-url>
cd Hackaton
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirement.txt
```

## 📈 Utilisation

### Afficher l'aide complète

```bash
python tradebot.py --help
python tradebot.py <commande> --help
```

### Commandes disponibles

| Commande | Description |
|----------|-------------|
| `backtest` | Backtester une stratégie sur données historiques |
| `compare` | Comparer plusieurs stratégies |
| `paper` | Paper trading avec prix Binance temps réel |
| `fetch` | Télécharger données historiques de Binance |
| `report` | Générer rapport complet avec graphiques |
| `list` | Lister les stratégies disponibles |
| `test` | Tester la connexion Binance |

### Exemples

```bash
# Tester la connexion API
python tradebot.py test

# Lister les stratégies
python tradebot.py list

# Télécharger 30 jours de données BTC/ETH
python tradebot.py fetch --days 30 --output data/crypto_30d.csv

# Backtester la stratégie safe_profit
python tradebot.py backtest --data data/crypto_30d.csv --strategy safe_profit

# Comparer toutes les stratégies
python tradebot.py compare --data data/crypto_30d.csv

# Générer un rapport complet
python tradebot.py report --data data/crypto_30d.csv --output reports/

# Paper trading temps réel (1 heure)
python tradebot.py paper --duration 3600 --strategy safe_profit
```

## 📊 Stratégies

| Nom | Description | Caractéristiques |
|-----|-------------|------------------|
| `safe_profit` | Combinaison conservative | Meilleur alpha cross-validé, faible drawdown |
| `adaptive_trend` | Suivi de tendance adaptatif | Trailing stop, filtre volatilité |
| `composite` | Multi-indicateurs | SMA + stoploss + scaling volatilité |
| `sma` | Moving Average Crossover | Simple mais efficace |
| `baseline` | Momentum basique | Référence de comparaison |

## 🐳 Docker

```bash
  # Build image
  docker build -t trading-bot:latest .

  # Run a backtest inside container
  docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs trading-bot:latest backtest --data data/crypto_btc_eth_4h_90d.csv --strategy safe_profit --output /app/outputs/docker_backtest

  # Run paper trading (1 hour)
  docker run --rm -v $(pwd)/experiments:/app/experiments trading-bot:latest paper --duration 3600 --strategy safe_profit --symbols BTCUSDT ETHUSDT
```

## 📦 Publish

Push a semantic tag to trigger automatic image publishing to GHCR (GitHub Container Registry):

```bash
# Tag and push
git tag v1.0.0
git push origin v1.0.0
```

The CI will build and push the image to `ghcr.io/<owner>/<repo>` if the workflow detects a tag push.

## 📁 Structure des données

Format CSV attendu :
```csv
epoch,Asset A,Asset B,Cash
0,100000.0,3500.0,1.0
1,100500.0,3520.0,1.0
...
```

## 🧪 Tests

```bash
# Test rapide de connexion
python tradebot.py test

# Backtest avec données de test
python tradebot.py backtest --data data/crypto_btc_eth_4h_90d.csv
```

## 📄 License

MIT License

---

> **Note:** Ce bot est destiné à l'éducation et au paper trading. Utilisez-le à vos propres risques pour du trading réel.
