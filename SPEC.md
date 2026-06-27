# TradeBot — Spécification

> Document de référence du projet. Il fige **ce que le bot doit être**, pour
> que le développement puisse avancer en autonomie sans dériver.
> Statut : **brouillon à valider** — n'hésite pas à annoter / corriger / ajouter.

---

## 1. La vision en une phrase

Un **assistant personnel d'analyse boursière** qui tourne en fond pendant des
mois, surveille un ensemble d'actions, repère des opportunités, et me donne des
**conseils clairs (acheter / vendre / conserver)** que je suis libre d'exécuter
**à la main** sur Trade Republic.

## 2. Le contexte (qui je suis, comment j'investis)

- J'investis de **petites sommes** (~50 € par mouvement), à la main, sur **Trade
  Republic**.
- Je ne suis **pas** trader pro : je veux un copilote qui analyse pour moi et
  m'explique pourquoi, pas une boîte noire.
- Le bot doit pouvoir tourner **en continu, des semaines/mois**, sans surveillance,
  et me laisser un historique consultable quand je reviens.

## 3. Ce que le bot doit faire (fonctionnel)

1. **Interface console interactive (REPL).** Je lance simplement `tradebot` et
   j'obtiens une **console qui tourne en continu** où je tape des commandes
   (`add AAPL`, `analyze NVDA`, `portfolio`, `report`…). **Pas** de CLI à
   arguments passés au lancement. La console est le poste de pilotage 24/7.
2. **Gérer ma base d'actions à la main** depuis la console : ajouter / retirer une
   action de la watchlist, saisir mes positions réelles (ce que j'ai acheté sur
   Trade Republic). Stockage **local**.
3. **Analyser une action à la demande** (`analyze AAPL`) : indicateurs techniques,
   figures chartistes, sentiment des news → un **signal consolidé** (BUY / SELL /
   HOLD) avec score, confiance, et **explication**.
4. **Surveiller la watchlist en fond** : analyse périodique **pendant les heures
   de marché**, alertes sur les signaux actionnables (console + Discord).
5. **Scanner le marché** pour **découvrir** des opportunités hors watchlist
   (anomalies de volatilité, croissance régulière).
6. **Suivre DEUX portefeuilles** (séparés) :
   - **Réel** : mes vraies positions, saisies à la main quand j'achète sur Trade
     Republic. Mark-to-market + P&L → savoir où j'en suis pour de vrai.
   - **Simulé** : portefeuille fictif que le bot gère tout seul (ouvre/ferme des
     positions papier sur signaux forts). Sert à **juger si le bot est bon avant
     de lui faire confiance** ; quand il est fiable, je reproduis ses bons coups
     en vrai sur TR et je les enregistre dans le portefeuille réel.
7. **Journaliser tout** dans `data/journal.jsonl` (signaux, trades, équité).
8. **Notifier via Discord** (serveur perso/familial dédié) : alertes en temps réel.
9. **Newsletter hebdomadaire** : résumé de fin de semaine (P&L, trades, meilleurs/
   pires signaux, opportunités repérées) envoyé par **Discord et/ou e-mail**.

## 4. Comment les conseils sont produits (logique de décision)

Trois sources de signal, combinées par pondération :

| Source | Poids | Outil |
|---|---|---|
| Technique (RSI, MACD, MA, momentum…) | 0.40 | `analysis/technical.py` |
| Figures chartistes (math/PIPs) | 0.35 | `analysis/pattern_detector.py` |
| Sentiment des news | 0.25 | `analysis/sentiment.py` |

- Score composite > **+0.20** → BUY ; < **−0.20** → SELL ; sinon HOLD.
- Le sentiment utilise **Ollama** en local s'il est joignable, sinon **fallback
  par règles** (mots-clés) — jamais de blocage si Ollama est down.

## 5. Contraintes dures (non négociables)

- **Gratuit.** Aucune dépense, aucun abonnement payant.
- **Vie privée.** Aucun service tiers exigeant ma pièce d'identité. **Pas de
  connexion à mon vrai compte Trade Republic** (risque de fermeture + ID). Les
  données de marché viennent de **yfinance** (aucun compte, aucun ID requis).
- **Pas de trading réel automatisé.** Le bot conseille et simule ; j'exécute
  moi-même mes vrais achats. Je saisis mes positions réelles à la main.
- **Console, pas CLI à arguments.** Le lancement de `tradebot` ouvre une console
  interactive ; on ne passe pas de sous-commandes en paramètre au démarrage.
- **Dégradation gracieuse.** Si une brique tombe (Ollama, news, réseau), le bot
  continue avec ce qu'il a au lieu de planter.
- **Local-first.** Tourne sur ma machine (WSL), peut joindre l'Ollama de Windows.

## 5b. Sobriété énergétique (le bot tourne 24/7)

Comme la console reste ouverte en permanence, il ne faut **rien consommer pour
rien** :

- **Une seule "fenêtre active" configurable** (au lieu de modéliser tous les
  calendriers de bourse mondiaux). Elle correspond aux heures où je peux
  réellement agir sur **Trade Republic** : en gros **en semaine ~8h → 23h
  (Europe/Paris), fermé le week-end**. Hors de cette fenêtre, agir est impossible
  (TR passe par un teneur de marché, pas la bourse en direct), donc analyser ne
  sert à rien.
  - Rationale : même si une action US/asiatique bouge la nuit, je ne peux pas
    l'acheter avant la réouverture de TR ; inutile de réveiller le bot pour ça.
  - Le calendrier précis par place boursière (NYSE, Euronext, Tokyo…) reste une
    **idée future**, pas le MVP.
- **Veille hors fenêtre.** Le bot calcule le prochain créneau actif et **sleep
  jusque-là** (réveil automatique). Jours fériés gérables en option plus tard.
- **Cadence adaptative.** Analyse périodique dans la fenêtre active, rien en
  dehors. Le scan et la **newsletter hebdo** tournent 1× (ex. dimanche soir).
- **Push plutôt que poll** quand c'est possible : s'abonner aux news plutôt que
  d'interroger en boucle ; mutualiser les appels yfinance (batch + cache).
- **Option cron / relance auto.** Possibilité de piloter via cron système (ex.
  réveiller l'app au début de la fenêtre) plutôt que de la laisser tourner à
  vide ; le programme doit pouvoir s'arrêter proprement et se relancer.
- L'interface console, elle, reste réactive (elle ne consomme rien tant que je
  ne tape rien) ; ce sont les **tâches de fond** qui se mettent en veille.

## 6. Hors périmètre (ce que le bot N'EST PAS)

- ❌ Pas de **crypto**, pas de **Binance**, pas de pairs-trading BTC/ETH
  (= tout l'ancien moteur, à supprimer).
- ❌ Pas de **CLI à arguments** au lancement (on remplace par la console REPL).
- ❌ Pas de **détection par vision YOLOv8** pour l'instant (lourd, modèles à gérer)
  — on reste sur la détection mathématique des figures.
- ❌ Pas d'exécution d'ordres réels, pas de connexion broker.
- ❌ Pas de gestion multi-utilisateurs / compte / web app. C'est **mon** outil.

## 7. Sources de données & intégrations

| Donnée | Source | Note |
|---|---|---|
| Prix historiques & cours | **yfinance** | gratuit, sans compte. Source par défaut. |
| News | yfinance (fallback) / Finnhub si clé | Finnhub free ne sert plus l'historique. |
| Sentiment | **Ollama** local / **règles** en fallback | jamais bloquant. |
| Notifications | **Discord webhook** | serveur perso/familial dédié, URL dans `.env`. Simple, sans token. |
| Newsletter | **Discord** (webhook) | résumé hebdomadaire posté dans le salon. |
| Fenêtre active | horaires configurables (défaut : semaine 8h–23h Europe/Paris) | pour la veille hors fenêtre. |

> Telegram : un `telegram_bot.py` existe déjà ; on le garde optionnel mais la
> cible principale des notifs devient **Discord (webhook)**.
> E-mail/SMTP pour la newsletter : **idée future**, pas le MVP (Discord suffit).
> Bot Discord interactif (recevoir des commandes depuis Discord) : **futur** ;
> pour l'instant Discord = réception de notifs uniquement, le pilotage se fait
> dans la console.

## 8. Architecture cible (après nettoyage)

```
src/
  config.py            # config centralisée (pydantic-settings, .env)
  core/models.py       # modèles métier (Position, Portfolio, TradingSignal, …)
  data/
    quotes.py          # cours temps réel (yfinance, cache)
    finnhub_source.py  # prix historiques (Finnhub → fallback yfinance)
    news_fetcher.py    # news (Finnhub → fallback yfinance)
  analysis/
    technical.py       # indicateurs techniques
    pattern_detector.py# figures chartistes (math)
    sentiment.py       # sentiment Ollama + fallback règles
    signal_aggregator.py # combine les 3 scores → signal
    scanner.py         # scan du marché (opportunités)
  portfolio/
    manager.py         # portefeuille (cash, positions, mark-to-market) — réutilisé
                       #   pour 2 instances : réel (data/portfolio_real.json) et
                       #   simulé (data/portfolio_sim.json)
    journal.py         # log append-only JSONL
    watchlist.py       # (NOUVEAU) liste d'actions suivies, persistée localement
  market/
    schedule.py        # (NOUVEAU) fenêtre active configurable, prochain créneau, sleep
  scheduler.py         # (NOUVEAU) tâches de fond (analyse en fenêtre active, scan,
                       #           newsletter hebdo) avec mise en veille hors fenêtre
  reporting/
    discord.py         # (NOUVEAU) notifications via webhook Discord (cible principale)
    telegram_bot.py    # notifications Telegram (optionnel, conservé)
    report.py          # (NOUVEAU) résumé du journal + newsletter hebdo
  engine.py            # cœur applicatif (analyze/scan/simulate) appelé par la console
  console.py           # (NOUVEAU) REPL interactif — point d'entrée principal
  runner.py            # orchestrateur async des tâches de fond
__main__ / tradebot    # lance la console (aucun argument requis)
```

## 9. Interface : la console interactive (REPL)

`tradebot` (sans argument) ouvre une console qui reste ouverte 24/7. Les tâches
de fond (surveillance, scan, newsletter) tournent en parallèle et se mettent en
veille hors séance ; la console reste réactive.

Commandes envisagées (à affiner pendant le dev) :

```
help                         # liste des commandes
add AAPL                     # ajoute une action à la watchlist
remove AAPL                  # retire une action de la watchlist
list                         # affiche la watchlist
analyze NVDA [days]          # analyse one-shot, affiche le signal + explication
scan                         # scan d'opportunités maintenant
buy AAPL 0.5 152.30          # saisir une position RÉELLE (qty, prix d'achat)
sell AAPL 0.5 161.00         # saisir une vente RÉELLE
portfolio                    # état du portefeuille réel + P&L live
sim                          # état du portefeuille simulé (géré par le bot) + P&L
report [n]                   # résumé du journal (P&L réel & simulé, trades, top signaux)
status                       # marché ouvert/fermé, prochaine ouverture, tâches de fond
start / stop                 # démarrer/arrêter la surveillance de fond
quit                         # quitter proprement
```

## 10. Critères de « terminé »

- [x] Le repo ne contient **plus** de code crypto/Binance (~3,9k lignes utiles).
- [x] `tradebot` lance une **console interactive** (sans argument) qui tourne 24/7.
- [x] Depuis la console : ajouter/retirer des actions, saisir des positions, lancer
      analyses/scans, voir le portefeuille et un rapport.
- [x] Tâches de fond : surveillance en séance, **veille hors séance** (sleep jusqu'à
      la prochaine ouverture), scan + newsletter hebdo.
- [x] **Notifications Discord** opérationnelles (webhook, serveur perso).
- [x] **Newsletter hebdomadaire** (Discord) avec résumé.
- [x] Le scanner alimente la surveillance/simulation (découverte d'opportunités).
- [x] `report` résume le journal (P&L réel & simulé, trades, meilleurs/pires signaux).
- [x] Tests sur : agrégation des signaux, P&L des portefeuilles, fallback sentiment,
      logique heures de marché, watchlist (23 tests).
- [x] README à jour, cohérent avec cette spec.

## 11. Idées futures (pas maintenant — à discuter)

- Détection de figures par vision (YOLOv8).
- Bot Discord/Telegram **interactif** (piloter le portefeuille depuis le téléphone,
  pas seulement recevoir des notifs).
- Backtest des stratégies **actions** sur historique (réécriture propre).
- Plusieurs profils de risque / tailles de position dynamiques.
- Pilotage via **cron système** pour réveiller l'app à l'ouverture du marché.

### Idées IA avancées (très long terme, une fois le reste fini)

- **Auto-apprentissage sur les performances simulées.** Le bot analyse ses
  propres résultats de simulation (signaux gagnants vs perdants) et **progresse**
  tout seul : ajuste les poids/seuils, voire entraîne un modèle d'IA spécialisé
  sur l'historique de ses décisions. But : qu'il « comprenne » ce qui marche.
- **Veille d'actualité automatisée à grande échelle.** Un crawler / des
  recherches Google News (ou autre) utilisant comme mots-clés **tous les noms
  des entreprises** de la liste boursière, pour que l'IA détermine à partir des
  articles si une entreprise « s'en sort bien » — sentiment macro par société,
  au-delà des news fournies par yfinance/Finnhub.
- Plus généralement : **modèles d'IA optimisés** dédiés à ces tâches (scoring
  d'opportunités, résumé de news, détection d'événements marquants).

> Ces pistes sont volontairement hors MVP : on construit d'abord un bot fiable,
> puis on lui ajoute de l'intelligence apprenante par-dessus.

---

*Quand cette spec te convient, je passe à la Phase 1 (nettoyage).*
