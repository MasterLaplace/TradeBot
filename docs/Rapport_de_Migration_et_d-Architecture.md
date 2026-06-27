# **Étude d'Opportunité et Plan de Migration : Évolution de TradeBot vers une Suite d'Aide à la Décision Modernisée**

Ce document évalue la viabilité technique du dépôt existant TradeBot (développé il y a 7 mois) au regard des objectifs d'une suite logicielle moderne (analyse d'images YOLOv8, IA locale Ollama, scraping d'actualités et notifications asynchrones). Il propose une recommandation claire et un plan de migration par étapes.

## **1. Audit Technique du Dépôt Actuel (TradeBot)**

La structure de ton dépôt montre un projet d'une qualité technique bien supérieure à la moyenne des projets de hackathons académiques.

### **Les Points Forts (À Conserver)**

* **Architecture Modulaire Propre (src/)** : La séparation claire entre core/ (modèles), data/ (sources de données), engine/ (backtest et paper trading) et strategies/ (logique de trading) est excellente. Elle respecte les principes SOLID.  
* **Infrastructure & DevOps** : La présence de Dockerfile, docker-compose.yml, Makefile, et d'un pipeline de CI/CD GitHub Workflows (ci.yml, release.yml) fournit une base de déploiement robuste qui prendrait des heures à réécrire.  
* **Moteur de Backtest et de Simulation** : Les modules src/engine/backtest.py et src/engine/paper_trading.py offrent déjà un cycle de vie pour tester et simuler des stratégies. C'est un atout majeur pour tester tes idées à 50 € sans risque.  
* **Gestion de Configuration** : L'utilisation de pyproject.toml (probablement couplé à Poetry ou Ruff) et de setup_env.sh montre un environnement de développement sain et standardisé.

### **Les Faiblesses vis-à-vis des Nouveaux Objectifs**

* **Paradigme Synchrone** : Le framework actuel semble principalement synchrone. L'introduction de flux de données en temps réel (WebSockets) et d'un système de notification de masse (Telegram) nécessite un moteur asynchrone basé sur asyncio et aiohttp.  
* **Briques Expérimentales Obsolètes** : Le dossier experimental/ contient des scripts de scraping (scrape_thread.py, cookies.json) qui sont par nature fragiles, difficiles à maintenir et potentiellement bloqués par les politiques anti-robots des sites d'actualités.  
* **Absence de Support pour l'IA et la Vision** : Rien n'est prévu pour héberger des modèles de Deep Learning (comme YOLOv8) ou communiquer avec un serveur d'inférence local (Ollama).

## **2. Le Verdict : Conserver, Refactoriser ou Réécrire ?**

### **Recommandation : Refactorisation Partielle (Hybride)**

**Pourquoi ne pas archiver ?**

Réécrire tout le squelette (CLI, parser de configuration, abstractions de stratégies, moteur de backtest, setup Docker, CI/CD et suite de tests) représente un travail fastidieux et sans grande valeur ajoutée. Ton dépôt actuel fournit une **excellente plomberie technique**.

**Pourquoi ne pas le garder tel quel ?**

La couche d'ingestion de données et la boucle principale d'exécution doivent être adaptées pour supporter l'asynchronisme et les nouvelles sources de données (APIs officielles, IA locale).

La meilleure approche consiste à **conserver le dépôt existant comme socle**, à nettoyer les modules obsolètes (comme les scripts de scraping du dossier experimental/) et à ajouter les nouvelles fonctionnalités sous forme de modules ou de services complémentaires dans src/.

## **3. Plan de Migration Étape par Étape**

Pour faire évoluer ton dépôt sans casser l'existant, voici la feuille de route recommandée :

                  ┌──────────────────────────────┐  
                  │      Dépôt TRADEBOT (Socle)  │  
                  └──────────────┬───────────────┘  
                                 │  
         ┌───────────────────────┴───────────────────────┐  
         ▼                                               ▼  
[1. Nettoyage & Setup]                         [2. Modernisation Data]  
 ├── Supprimer experimental/                    ├── Intégrer aiohttp / asyncio  
 └── Migrer vers 'uv' (pyproject.toml)          └── Connecter Websockets Finnhub  
         │                                               │  
         └───────────────────────┬───────────────────────┘  
                                 ▼  
                       [3. Nouvelles Briques]  
                        ├── YOLOv8 (Vision) dans src/strategies/  
                        ├── Ollama (NLP) dans src/core/nlp.py  
                        └── Telegram (Notif) dans src/reporting/

### **Étape 1 : Nettoyage et Modernisation de l'Environnement**

1. **Archiver l'ancien scraping** : Supprime le dossier experimental/ qui repose sur du scraping de thread synchrone instable.  
2. **Adopter uv** : Remplace l'installation classique par le gestionnaire de paquets ultra-rapide uv. Mets à jour ton pyproject.toml pour y ajouter les nouvelles dépendances requises :  
   [tool.poetry.dependencies] # ou format standard PEP 621  
   ultralytics = "^8.1.0"      # Pour YOLOv8  
   ollama = "^0.1.0"           # Pour l'IA locale  
   aiohttp = "^3.9.0"          # Pour les requêtes asynchrones  
   pydantic = "^2.6.0"         # Pour la validation de schémas JSON  
   pandas-ta = "^0.3.14b"      # Pour les indicateurs techniques (EMA)

### **Étape 2 : Moderniser la Couche d'Ingestion (src/data/)**

Ton fichier src/data/sources.py doit être modifié ou étendu pour supporter les flux asynchrones :

* Conserve les sources CSV pour le backtesting historique (asset_b_train.csv).  
* Ajoute une classe AsyncWebsocketSource dans src/data/broadcaster.py pour écouter les flux en direct (par exemple via l'API gratuite de Finnhub ou d'Alpaca).

### **Étape 3 : Intégrer la Détection de Formes Chartistes (src/strategies/)**

Crée une nouvelle stratégie dans src/strategies/chart_pattern_strategy.py qui hérite de ta classe de base src/strategies/base.py :

* Elle récupère les données récentes de ton broadcaster.py.  
* Elle utilise pandas-ta pour lisser la courbe.  
* Elle utilise soit l'extraction de points pivots (PIPs) pour la détection mathématique, soit elle génère une image temporaire avec mplfinance pour l'envoyer au modèle YOLOv8 local.

### **Étape 4 : Ajouter le Cerveau Sémantique (Analyse des News)**

Crée un nouveau module src/tools/news_analyzer.py ou src/core/nlp.py :

* Il interroge l'API d'actualités gratuite de Finnhub (REST).  
* Il appelle l'API locale d'Ollama (modèle qwen3 ou gemma4) pour obtenir une analyse de sentiment structurée en JSON via Pydantic (comme décrit dans l'architecture de référence).

### **Étape 5 : Déployer le Pipeline de Notification Asynchrone**

Modifie ton module src/reporting/reports.py :

* Remplace l'ancienne logique de rapport par un bot Telegram asynchrone.  
* Utilise aiohttp pour envoyer les alertes rédigées par Ollama et les graphiques de signaux annotés par YOLOv8 directement sur ton téléphone.

## **4. Exemple d'Architecture Cible du Code Refactorisé**

Voici comment ton arborescence va accueillir ces nouvelles technologies tout en valorisant ton travail passé :

TradeBot/  
├── .github/workflows/          # CONSERVÉ : Automatisation de tes tests (CI/CD)  
├── data/                       # CONSERVÉ : Tes datasets de test historique  
├── src/  
│   ├── cli/                    # CONSERVÉ & ADAPTÉ : Tes commandes pour lancer le bot  
│   ├── core/  
│   │   ├── models.py           # CONSERVÉ : Tes objets métiers (Asset, Order, Position)  
│   │   └── nlp.py              # AJOUTÉ : Analyseur Ollama local (Pydantic / JSON)  
│   ├── data/  
│   │   ├── sources.py          # ADAPTÉ : Ajout du client WebSocket asynchrone  
│   │   └── broadcaster.py      # ADAPTÉ : Dispatcher de prix temps réel  
│   ├── engine/  
│   │   ├── backtest.py         # CONSERVÉ : Pour valider tes patterns sur l'historique !  
│   │   └── paper_trading.py    # CONSERVÉ : Pour simuler à blanc avec tes 50 € virtuels  
│   ├── reporting/  
│   │   └── telegram_bot.py     # AJOUTÉ : Moteur d'alertes asynchrone Telegram (30 req/s max)  
│   └── strategies/  
│       ├── base.py             # CONSERVÉ : Classe abstraite de tes stratégies  
│       ├── moving_average.py   # CONSERVÉ : Ton ancienne stratégie du hackathon  
│       └── geometric_yolo.py   # AJOUTÉ : Ta nouvelle stratégie combinant YOLOv8 & PIPs  
├── tradebot.py                 # CONSERVÉ & ADAPTÉ : Point d'entrée principal  
├── pyproject.toml              # MIS À JOUR : Dépendances modernes (uv, ultralytics, ollama)  
└── docker-compose.yml          # MIS À JOUR : Ajout du service Ollama GPU/CPU local

## **5. Pourquoi la conservation du moteur de Backtest est cruciale pour toi**

Puisque tu investis de petites sommes (50 €), tu ne peux pas te permettre de perdre bêtement de l'argent sur des faux signaux techniques.

Le fait de conserver ton dépôt actuel te permet de **garder ton moteur de backtest**. Avant d'envoyer ta stratégie YOLOv8 ou tes signaux sur le marché réel, tu pourras la faire tourner sur tes fichiers CSV historiques (data/asset_b_train.csv) pour vérifier si tes intuitions géométriques sont réellement rentables sur les 12 derniers mois.