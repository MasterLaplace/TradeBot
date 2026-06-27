# **Architecture de Référence d'une Suite Logicielle de Trading Quantitatif : Évaluation des API de Courtage, Détection de Formations Chartistes et Intelligence Artificielle en Local**

L'évolution des technologies financières permet désormais aux développeurs indépendants de transformer des prototypes académiques simples, souvent conçus lors de hackathons (comme les architectures modulaires classiques de bots de trading à l'instar de TradeBot.git), en suites logicielles quantitatives hautement sophistiquées1. La transition d'un script d'automatisation basique vers une infrastructure de trading systématique en temps réel exige une refonte architecturale structurée2. Ce document présente l'état de l'art pour concevoir une suite logicielle intégrée, combinant l'accès programmatique aux courtiers, l'analyse technique géométrique et sémantique avancée, l'inférence d'intelligence artificielle locale, et le respect rigoureux du cadre fiscal français.

## **1. Interfaces d'Accès aux Services de Courtage : Trade Republic et Alternatives**

L'accès automatisé aux données de marché et l'exécution d'ordres nécessitent une évaluation rigoureuse des interfaces de programmation d'application (API) disponibles2. Le choix de l'infrastructure de courtage dépend de la gratuité, de la stabilité officielle de l'API et de la complexité opérationnelle d'intégration4.

### **L'Écosystème Trade Republic et les Limites de la Rétro-Ingénierie**

Trade Republic ne propose pas d'API publique officielle, documentée et libre d'accès pour les clients particuliers1. L'accès par programmation repose sur des projets communautaires de rétro-ingénierie, principalement les bibliothèques Python pytr et py_tr2. Ces outils interceptent et répliquent les protocoles de communication WebSocket asynchrones utilisés par l'application mobile et l'interface de l'application web (app.traderepublic.com)2.  
La bibliothèque pytr s'installe de manière moderne via le gestionnaire de paquets uv7 :

Bash  
uvx pytr@latest

Les commandes CLI disponibles permettent d'exécuter des actions de diagnostic, d'extraire l'état du portefeuille, de suivre les plans d'épargne actifs ou de télécharger l'historique des transactions au format CSV7 :

Bash  
pytr portfolio  
pytr dl_docs

Pour une intégration applicative au sein d'un bot de trading existant, la bibliothèque asynchrone py_tr permet d'établir une connexion WebSocket persistante afin de souscrire à des flux de données en temps réel2. L'API asynchrone de Trade Republic fonctionne entièrement par abonnements à des « sujets » (*topics*) qui retournent un identifiant unique d'abonnement (subscription_id)2. Le flux renvoie une réponse initiale, puis des mises à jour en temps réel dès qu'une modification survient sur le marché ou le compte2.  
Les appels d'API pris en charge par cette méthode de rétro-ingénierie couvrent un large spectre d'actions2 :

* **Abonnements de Portefeuille** : Récupération des liquidités (tr.cash()), de l'évaluation globale (tr.portfolio()) et de l'historique de performance selon différentes granularités temporelles2.  
* **Flux de Marché** : Abonnement aux carnets d'ordres simplifiés et aux cours d'exécution via la place boursière LSX (tr.ticker(isin, exchange="LSX")) avec une résolution minimale de mise à jour fixée à 60 000 millisecondes (1 minute) pour les séries temporelles historiques2.  
* **Actualités** : Flux d'actualités associés à un instrument financier spécifique (tr.news(isin))2.  
* **Transactions REST** : Les ordres de virement sortant (tr.payout(amount)) s'exécutent de manière synchrone via des requêtes HTTP REST standards et requièrent une confirmation par double facteur (2FA) via la validation d'un code SMS reçu sur le terminal mobile lié (tr.confirm_payout(process_id, code))2.

Malgré sa richesse fonctionnelle, cette architecture non officielle présente des risques opérationnels et de sécurité majeurs1. L'authentification impose des contraintes lourdes7. La méthode par connexion web envoie un code à quatre chiffres par SMS à chaque tentative, ce qui empêche une automatisation autonome continue7.  
La méthode par clé applicative exige une procédure de réinitialisation de l'appareil (*device reset*) générant une clé privée locale au format PEM (keyfile.pem) qui valide le script comme un terminal de confiance, mais cette procédure déconnecte de manière irréversible l'utilisateur de son application mobile officielle2. En outre, toute mise à jour de la structure réseau par Trade Republic peut rendre l'infrastructure logicielle du bot instantanément obsolète1 et l'utilisation d'une API non officielle expose à une fermeture unilatérale du compte pour violation des conditions générales d'utilisation6.

### **Alternatives avec API Officielles : Scalable Capital et Trading 212**

Pour concevoir une architecture logicielle robuste et pérenne, l'analyse s'oriente vers des courtiers proposant des accès applicatifs officiellement pris en charge en Europe3.

#### **Scalable Capital**

Le courtier allemand Scalable Capital met à disposition une interface en ligne de commande officielle et robuste, appelée scalable-cli (sc), développée en Rust8. Ce client natif est conçu spécifiquement pour les développeurs, l'automatisation locale et l'intégration directe avec des agents d'intelligence artificielle8.  
Pour utiliser l'outil, le compte de l'utilisateur doit être explicitement inscrit sur une liste d'autorisation (*allowlist*)8. Le développeur doit générer un code d'installation via la commande sc installation-code8, puis soumettre une demande par courrier électronique à l'adresse cli.beta@scalable.capital avec pour objet *Scalable CLI Allowlisting*8.  
Une fois l'autorisation accordée, le client permet d'obtenir des sorties structurées au format JSON (sc capabilities --json) et de passer des ordres d'achat ou de vente de manière déterministe avec un processus de validation en deux étapes pour sécuriser les transactions8.

#### **Trading 212**

Trading 212 propose une API publique officielle en version bêta, accessible gratuitement pour les comptes de démonstration (*Demo*) et de production (*Live*)3. L'authentification utilise un protocole d'authentification basique HTTP standard via l'encodage en Base64 d'une paire de clés d'accès générée dans l'application mobile sous l'onglet *Settings → API*3.  
L'écosystème de développement Python propose des wrappers matures pour interagir avec cette API, notamment trading212-connector ou t212-api3. L'API permet d'accéder à l'état des liquidités de manière instantanée, de suivre les positions ouvertes et d'envoyer des requêtes d'achat ou de vente d'actions physiques ou fractionnées3. Les ventes s'exécutent en soumettant une quantité négative dans le paramètre de transaction11.  
Cependant, le retour d'expérience de la communauté de développeurs met en évidence plusieurs limitations techniques et anomalies logicielles13 :

* **Anomalie de Pagination** : L'extraction de l'historique complet des transactions via le paramètre de pagination par curseur présente des instabilités14. La solution technique consiste à remplacer dynamiquement la valeur du curseur par l'horodatage exact (*timestamp*) de la dernière transaction renvoyée14.  
* **Incohérence des Identifiants (Tickers)** : Le point de terminaison du portefeuille (/api/v0/equity/portfolio) peut renvoyer de manière erronée des symboles boursiers incorrects (par exemple, afficher le ticker "BVS" au lieu de "VTY", ou "PMO" au lieu de "HBR")14.  
* **Limitation d'Exécution Horaires** : Tout ordre à cours limité soumis en dehors des heures de négociation officielles du marché ne subit aucune pré-exécution ou file d'attente valide, restant bloqué jusqu'à l'ouverture réelle du carnet d'ordres13.  
* **Validation GTC** : Les ordres soumis avec une validité de type *Good-'til-Canceled* (GTC) échouent fréquemment à l'enregistrement système, imposant l'utilisation exclusive de la validité de type *DAY*13.

### **Synthèse Comparative des Solutions de Courtage**

Le tableau ci-dessous regroupe les caractéristiques des principales infrastructures d'exécution utilisables depuis la France :

| Courtier | Interface d'Accès | Statut de l'API | Mode de Transmission | Gestion 2FA / Enregistrement | Types d'Ordre Supportés |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Trade Republic** [cite: 1, 2, 7] | Projet communautaire pytr / py_tr | Non officiel | Asynchrone (WebSockets) et REST | Complexe (Clé PEM avec déconnexion mobile obligatoire) | Marché, Limite, Plans d'épargne programmés |
| **Scalable Capital** [cite: 8] | CLI Officiel Rust (sc) | Officiel (Bêta) | REST local (Sorties JSON) | Standard (Enregistrement par e-mail à cli.beta@scalable.capital) | Marché, Limite, Stop |
| **Trading 212** [cite: 3, 10, 11] | Requêtes REST / SDK Python t212 | Officiel (Bêta) | REST Synchrone | Simple (Clé API générée via l'interface client) | Marché, Limite, Stop, Stop-Limite |
| **Interactive Brokers** [cite: 4, 15, 16] | API Native / TWS / ib_insync | Officiel (Production) | TCP Local via passerelle logicielle | Authentification sécurisée locale par TWS ou IB Gateway | Plus de 100 algorithmes complexes (Bracket, OCO, Pegged) |

## **2. Modélisation Algorithmique de la Détection de Formations Chartistes**

L'automatisation de la détection de motifs géométriques sur les courbes de prix s'appuie sur deux méthodologies distinctes : l'analyse mathématique par extraction de points d'inflexion locaux et la classification d'images par vision par ordinateur17.

### **Analyse Mathématique par Points Pivots Extrêmes (PIPs)**

Pour formaliser de manière algorithmique des figures telles que les Épaules-Tête-Épaule (IHS) ou les Doubles Bas (W_Bottom), le système doit réduire le bruit structurel de la série temporelle17.

#### **Étape 1 : Lissage et Filtrage Temporel**

La série brute des prix de clôture subit un lissage initial à l'aide d'une Moyenne Mobile Exponentielle (![][image1]) afin de filtrer les micro-fluctuations sans induire de déphasage excessif17. La formule récursive de l'indicateur s'exprime ainsi :  
![][image2]  
Où ![][image3] est le prix de clôture actuel et le coefficient de lissage ![][image4] est défini par :  
![][image5]  
Avec ![][image6] représentant la taille de la fenêtre de lissage17.

#### **Étape 2 : Extraction des Extrema Locaux via SciPy**

L'identification des sommets (*maxima*) et des creux (*minima*) s'effectue à l'aide de l'algorithme des extrema locaux de la bibliothèque SciPy17. La fonction examine les prix lissés au sein d'une fenêtre glissante d'amplitude ![][image7]17 :  
![][image8]  
![][image9]

#### **Étape 3 : Algorithme de Changement Directionnel (Zigzag) et PIPs**

Pour affiner l'extraction des points d'inflexion majeurs, la suite logicielle peut mettre en œuvre l'algorithme des Points Perceptuellement Importants (PIPs) ou la méthode du Changement Directionnel (Zigzag)19. Le Zigzag filtre les mouvements insignifiants en appliquant un seuil de retracement minimum défini par un pourcentage ![][image10]19. Un nouvel extrême local n'est confirmé que si le prix diverge du précédent extrême confirmé d'une proportion supérieure ou égale à ![][image10]19 :  
![][image11]  
L'approche PIPs, quant à elle, fonctionne de manière récursive en calculant la distance orthogonale ou euclidienne de chaque point intermédiaire par rapport à une droite reliant deux extrêmes confirmés, sélectionnant à chaque itération le point de distance maximale comme nouveau pivot structurel19.

#### **Étape 4 : Validation Géométrique des Figures**

Une fois les points pivots successifs extraits sous la forme d'une séquence alternée de creux et de sommets ![][image12], des filtres mathématiques stricts valident la présence d'une figure spécifique17. Pour une figure de Double Bottom (W)17 :

* ![][image13] doivent être des minima locaux (les creux)17.  
* ![][image14] doivent être des maxima locaux (la ligne de cou ou sommet intermédiaire)17.  
* La différence de prix entre les deux creux principaux doit être inférieure à un seuil de tolérance géométrique de ![][image15]17 :

![][image16]  
Ces opérations analytiques de filtrage peuvent être grandement simplifiées par l'utilisation de bibliothèques comme pandas-ta-classic ou chart_patterns, qui implémentent nativement la détection des structures géométriques de chandeliers (tels que l'Engulfing ou le Hammer)21.

### **Analyse par Vision Numérique Globale : YOLOv8**

L'alternative moderne consiste à traiter la recherche de structures chartistes comme un problème de détection d'objets bidimensionnels18. Ce paradigme s'affranchit des équations rigides de proportions en apprenant de manière globale la topographie d'un graphique financier18.  
Le modèle de réseau de neurones convolutifs foduucom/stockmarket-pattern-detection-yolov8 est entraîné à repérer et délimiter les structures géométriques sur des représentations en chandeliers japonais18.  
Le tableau ci-dessous dresse l'inventaire des classes détectables par l'architecture YOLOv8 :

| ID de Classe | Identifiant Technique | Description du Motif Chartiste | Signification Stratégique |
| :---- | :---- | :---- | :---- |
| **0** | Head and shoulders bottom | Épaule-Tête-Épaule Inversée | Signal de retournement haussier majeur26 |
| **1** | Head and shoulders top | Épaule-Tête-Épaule Classique | Signal de retournement baissier majeur26 |
| **2** | M_Head | Double Sommet (Double Top) | Rejet de résistance, retournement baissier26 |
| **3** | StockLine | Ligne de tendance technique | Tracé automatique des supports et résistances26 |
| **4** | Triangle | Figure de compression (Triangle) | Phase de consolidation précédant une cassure26 |
| **5** | W_Bottom | Double Creux (Double Bottom) | Phase d'accumulation, retournement haussier26 |

Pour l'implémentation opérationnelle, le pipeline d'inférence exécute les opérations suivantes :

1. **Génération d'images** : Conversion des données historiques OHLC en images graphiques au format PNG ou JPEG via la bibliothèque mplfinance ou opencv-python18.  
2. **Filtrage des discontinuités** : Une fonction de contrôle analyse l'index temporel et élimine le graphique si un écart de temps supérieur à 10 minutes (*time gap*) est détecté dans le flux de données, prévenant ainsi les fausses détections liées aux trous de cotation25.  
3. **Inférence** : Passage de l'image générée au modèle YOLOv818 :

Python  
from ultralytics import YOLO

modele = YOLO("foduucom/stockmarket-pattern-detection-yolov8")  
predictions = modele.predict(source="graphique_ohlc.png", conf=0.30, save=True)

4. **Parsing des Résultats** : Le modèle renvoie une liste contenant pour chaque détection l'identifiant de la classe, le score de confiance et les coordonnées spatiales normalisées de la boîte de délimitation [x1, y1, x2, y2]26.

## **3. Traitement du Flux d'Actualités Financières (News Parsing) et Sentiment**

L'analyse technique pure doit être corrélée à une analyse fondamentale instantanée pour atténuer les faux signaux géométriques lors de publications macroéconomiques ou d'annonces de résultats d'entreprises28.

### **Évaluation des API de Récupération d'Actualités**

Le marché propose plusieurs interfaces d'extraction d'actualités financières dotées d'options gratuites adaptées à un usage individuel29.

* **Finnhub API** : Fournit un flux d'actualités générales de marché /news et d'actualités d'entreprises ciblées /company-news à l'aide d'une authentification par clé d'API standard31. Le plan gratuit autorise l'accès aux flux avec une limite d'appel robuste de 30 requêtes par seconde31.  
* **Alpha Vantage** : Ce service intègre un algorithme d'analyse sémantique automatique du sentiment au niveau de son point de terminaison NEWS_SENTIMENT33. L'API calcule de manière native le score de sentiment pour chaque article (allant de -1.0 pour un ton extrêmement baissier à +1.0 pour un ton haussier) ainsi que des scores de pertinence pour chaque entité d'actif détectée par reconnaissance d'entités nommées (NER)28.  
* **Webz.io News API Lite** : Orienté vers la recherche, ce fournisseur offre un plan d'évaluation gratuit de 1 000 requêtes mensuelles permettant de faire des requêtes booléennes complexes et d'extraire des entités textuelles spécifiques30.

Le tableau ci-dessous compare les performances et fonctionnalités de ces différentes solutions de collecte :

| Fournisseur | Point de Terminaison Clé | Latence Estimée | Analyse du Sentiment Intégrée | Limite du Plan Gratuit | Type de Données Retournées |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Finnhub** [cite: 31, 32] | /api/v1/company-news | Temps réel | Non | 30 requêtes / seconde | Titre, résumé, source, URL, horodatage UNIX31 |
| **Alpha Vantage** [cite: 28, 34] | /query?function=NEWS_SENTIMENT | Secondes (Polling) | Oui (Direction et Magnitude) | 25 requêtes / jour | Titre, URL, score de sentiment global et par ticker28 |
| **Webz.io** [cite: 30] | /newsApiLite | Près de temps réel | Oui | 1 000 requêtes / mois | Article complet enrichi par indexation IPTC et entités30 |
| **NewsAPI.org** [cite: 30] | /v2/everything | Quotidien (Historique) | Non | 100 requêtes / jour | Agrégation d'articles de presse de 50 000 sources mondiales30 |

## **4. Traitement Sémantique Local via Ollama et Pipeline de Notification Asynchrone**

Pour traiter et analyser les données collectées sans dépendre de services cloud payants, la suite logicielle s'appuie sur l'orchestration d'un moteur d'intelligence artificielle hébergé localement35.

### **Configuration d'Ollama et Modèles Quantifiés**

Ollama permet d'exécuter localement des modèles de langage de grande taille adaptés aux contraintes matérielles d'un serveur personnel35. Pour l'exercice de l'analyse financière et de l'extraction d'entités structurées, la suite logicielle recommande les modèles validés suivants36 :

* qwen3 : Modèle hautement stable, optimisé pour les tâches de logique mathématique et la génération de schémas JSON structurés (taille de 5,2 Go)36. Note: Ce modèle correspond au référentiel technique stable identifié localement36.  
* gemma4 : Modèle polyvalent prenant en charge le traitement multimodal (analyse conjointe d'images de graphiques financiers et de texte) ainsi que l'appel natif d'outils (taille de 9,6 Go)36.

### **Extraction Structurée de Données Financières**

Pour garantir que les analyses produites par le modèle local puissent alimenter de façon déterministe un outil d'aide à la décision ou un automate d'exécution, la réponse du modèle doit être contrainte par un schéma de données rigoureux38. Ollama prend en charge la définition de structures JSON obligatoires en transmettant un schéma JSON brut à son paramètre format lors de l'appel38.  
Voici l'implémentation d'un module d'évaluation sémantique utilisant la bibliothèque Pydantic pour valider et contraindre la sortie du modèle36 :

Python  
from pydantic import BaseModel, Field  
from typing import List  
import ollama

# Définition de la structure de données cible pour l'évaluation de l'actualité  
class RapportSentimentEntreprise(BaseModel):  
    nom_entreprise: str = Field(description="Nom exact de l'entreprise identifiée")  
    ticker: str = Field(description="Symbole boursier ou ISIN associé à l'instrument")  
    indicateurs_clefs: List[str] = Field(description="Indicateurs mentionnés : chiffre d'affaires, marge, endettement")  
    risques_identifies: List[str] = Field(description="Risques de marché, opérationnels ou réglementaires relevés")  
    sentiment_polarite: float = Field(description="Polarité sémantique de l'article : -1.0 (baissier) à +1.0 (haussier)")  
    confiance_analyse: float = Field(description="Indice de confiance de l'évaluation locale entre 0.0 et 1.0")

def executer_analyse_semantique(texte_actualite: str) -> RapportSentimentEntreprise:  
    prompt_instruction = (  
        "Vous agissez en qualité d'analyste financier quantitatif pour un fonds d'investissement. "  
        "Examinez l'article fourni pour extraire les mesures fondamentales, les risques opérationnels et évaluer la polarité sémantique globale. "  
        "Vous devez exclusivement répondre sous la forme d'un objet JSON valide conforme au schéma imposé."  
    )  
      
    reponse = ollama.chat(  
        model='qwen3', # Utilisation du modèle quantifié de référence pour les structures JSON  
        messages=[  
            {'role': 'system', 'content': prompt_instruction},  
            {'role': 'user', 'content': texte_actualite}  
        ],  
        format=RapportSentimentEntreprise.model_json_schema(), # Contrainte stricte de format  
        options={'temperature': 0.0} # Neutralisation de l'aléa linguistique [cite: 37, 38]  
    )  
      
    # Validation stricte de la structure de sortie par Pydantic  
    return RapportSentimentEntreprise.model_validate_json(reponse['message']['content'])

### **Pipeline Asynchrone de Notification de Masse via Telegram**

L'envoi des alertes de trading s'effectue vers le terminal de l'utilisateur à l'aide d'un canal ou d'une discussion privée gérée par un bot Telegram40.  
Pour s'assurer que l'infrastructure logicielle reste réactive lors de phases de volatilité extrême (où le bot doit émettre des dizaines d'alertes simultanées), l'utilisation d'un pipeline d'envoi synchrone basé sur des boucles séquentielles doit être rejetée car elle bloque l'exécution globale42. Le système met en œuvre une architecture entièrement asynchrone s'appuyant sur la bibliothèque aiohttp et la commande de regroupement de coroutines asyncio.gather42.  
Le module d'envoi doit intégrer les contraintes réseau de l'API de Telegram41 :

1. **Limitation de Débit (Rate Limiting)** : L'API de Telegram limite l'envoi de messages à un maximum de ![][image17] messages par seconde pour les utilisateurs de l'API de bot42. Pour respecter cette contrainte matérielle, la session d'appel asynchrone configure un connecteur réseau limité à 30 connexions simultanées maximum via la classe aiohttp.TCPConnector(limit=30)42.  
2. **Limite de Taille de Message** : La longueur maximale d'un unique message transmis via Telegram est fixée à ![][image18] caractères41. Au-delà, l'API renvoie une erreur de type BadRequest41. Le module scinde donc automatiquement tout texte trop long en fragments de taille conforme avant transmission41.

Voici le code du moteur de notification asynchrone à haute performance :

Python  
import asyncio  
import aiohttp

TELEGRAM_API_TOKEN = "VOTRE_TOKEN_TELEGRAM_BOT"

async def poster_alerte_asynchrone(session: aiohttp.ClientSession, chat_id: str, message: str):  
    # Division du message en segments de 4096 caractères pour respecter les limites de l'API  
    taille_max = 4096  
    parts_message = [message[i:i + taille_max] for i in range(0, len(message), taille_max)]  
      
    url_target = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendMessage"  
      
    for fragment in parts_message:  
        payload = {  
            'chat_id': chat_id,  
            'text': fragment,  
            'parse_mode': 'Markdown'  
        }  
        try:  
            async with session.post(url_target, json=payload) as reponse:  
                if reponse.status == 429:  
                    # En cas de retour 429 (Too Many Requests), pause adaptative  
                    donnees_retour = await reponse.json()  
                    delai_attente = donnees_retour.get("parameters", {}).get("retry_after", 1)  
                    await asyncio.sleep(delai_attente)  
                    # Nouvelle tentative de transmission  
                    await session.post(url_target, json=payload)  
        except Exception as e:  
            pass # Enregistrement de l'exception réseau dans les journaux d'erreurs locaux

async def distribuer_notifications_multi_utilisateurs(liste_utilisateurs: list, texte_message: str):  
    # Restriction stricte de l'API de Telegram à 30 requêtes par seconde  
    connecteur_limite = aiohttp.TCPConnector(limit=30)  
      
    async with aiohttp.ClientSession(connector=connecteur_limite) as session:  
        taches = [  
            poster_alerte_asynchrone(session, identifiant_chat, texte_message)  
            for identifiant_chat in liste_utilisateurs  
        ]  
        # Exécution concurrente globale de l'ensemble des requêtes de notification  
        await asyncio.gather(*taches)

## **5. Intégration Fiscale et Obligations Réglementaires en France**

L'automatisation des opérations financières par un résident fiscal français est soumise à un cadre juridique et déclaratif rigoureux43. L'ignorance de ces mécanismes expose l'investisseur à des pénalités financières lourdes43.

### **Obligation Déclarative des Comptes Détenus à l'Étranger (Annexe 3916)**

En vertu de l'article 1649 A du Code général des impôts, tout compte bancaire ou de valeurs mobilières ouvert, détenu, utilisé ou clos hors du territoire français au cours de l'année de perception des revenus doit faire l'objet d'une déclaration spécifique jointe à la déclaration annuelle de revenus44.

* **Trade Republic** : Bien qu'il dispose d'une succursale française enregistrée, les liquidités associées aux comptes titres des clients sont hébergées auprès d'établissements partenaires situés en Allemagne6. Le compte doit être déclaré chaque année via le formulaire Cerfa n° 3916 (Annexe 3916)44.  
* **Trading 212 / Scalable Capital** : Ces intermédiaires financiers n'ayant pas de domiciliation bancaire en France, l'inscription d'un compte auprès de leurs services déclenche l'obligation de dépôt du formulaire Cerfa n° 39166.  
* **Pénalités Applicables** : Le manquement à cette obligation déclarative expose à une amende forfaitaire de ![][image19] euros par compte non déclaré43. Cette amende grimpe à ![][image20] euros si la valeur totale des avoirs détenus sur le compte étranger a dépassé le seuil critique de ![][image21] euros à un moment quelconque de l'année fiscale écoulée43.

### **Fiscalité des Plus-Values de Valeurs Mobilières**

Par défaut, les gains issus de la cession d'actifs (plus-values de cession d'actions, d'ETF ou de produits dérivés) au sein d'un Compte-Titres Ordinaire (CTO) sont assujettis au Prélèvement Forfaitaire Unique (PFU)43.  
Pour l'exercice d'imposition en cours, le taux global du PFU s'établit à ![][image22] des gains nets réalisés43, décomposé comme suit :

* Une part représentative de l'impôt sur le revenu fixée au taux forfaitaire de ![][image23]43.  
* Une part représentative des prélèvements sociaux fixée au taux réévalué de ![][image24] depuis le 1er janvier 202643.

Il convient de souligner que les plus-values latentes (gains théoriques sur des actifs non encore revendus) ne sont jamais soumises à l'impôt44. Seul le débouclage d'une position (la vente réelle de l'actif) constitue un fait générateur d'imposition44. Les moins-values subies au cours de l'année de négociation sont directement déductibles des plus-values de même nature réalisées durant le même exercice, ou reportables sur les dix années fiscales suivantes44.

### **La Problématique de l'Imprimé Fiscal Unique (IFU)**

La gestion administrative de la déclaration d'impôt varie de manière spectaculaire selon le courtier sélectionné6 :

* **Courtiers Fournissant l'IFU (ex: Trade Republic, Fortuneo)** : Trade Republic fournit chaque année à ses usagers français un Imprimé Fiscal Unique (IFU) complet précalculé selon les normes nationales6. L'IFU indique avec précision les montants exacts des gains nets imposables et désigne explicitement les cases correspondantes de la déclaration de revenus n° 2042 (cases 3VG et 3VH pour les plus-values et moins-values)6.  
* **Courtiers Étrangers sans Support National (ex: Trading 212, DeGiro, Revolut)** : Ces courtiers ne fournissent aucun document d'Imprimé Fiscal Unique conforme à la législation fiscale française6. L'investisseur reçoit uniquement un relevé d'activité annuel global rédigé dans les devises d'origine des places de négociation46.

La déclaration des gains réalisés auprès de ces plateformes s'avère d'une extrême complexité technique46 :

* Chaque transaction de vente doit donner lieu au calcul de la plus-value ou moins-value sous-jacente en devise d'origine44.  
* Toutes les opérations libellées en devises étrangères (USD, GBP) doivent être converties en euros au taux de change officiel en vigueur à la date exacte de chaque opération43.  
* L'investisseur doit remplir manuellement la déclaration complexe des revenus de capitaux mobiliers n° 2047 et calculer son gain net sur le formulaire n° 2074-CMV44.

Pour surmonter cet obstacle administratif et éviter les redressements fiscaux, l'intégration d'un outil de consolidation automatisé tiers, tel que FlashFiscal ou DeclarAid, est fortement recommandée44. Ces applications importent les historiques d'exécution bruts générés par les robots de trading, gèrent les conversions de devises au jour le jour, calculent le prorata exact des retenues à la source sur les dividendes étrangers, et éditent des formulaires Cerfa d'annexe fiscale prêts à l'emploi44.

## **6. Synthèse d'Architecture Système Recommandée**

Pour concrétiser le projet d'une suite logicielle d'analyse et d'aide à la décision à la pointe de la technologie sans s'exposer à des blocages opérationnels ou fiscaux, l'architecture globale recommandée s'organise selon un schéma fonctionnel découplé.  
Cette architecture est conçue pour s'intégrer directement sur la base d'une implémentation classique de bot de trading en Python (telle que celle initiée lors du hackathon de l'école dans TradeBot.git), en remplaçant les scripts synchrones par des boucles asynchrones modernes basées sur l'écosystème open source détaillé dans ce rapport1.

┌──────────────────────────────────────────────────────────────────────────────────────────────┐  
│                                ARCHITECTURE LOGICIELLE DU SYSTEME                            │  
└──────────────────────────────────────────────────────────────────────────────────────────────┘

 1. COUCHE D'INGESTION DES DONNÉES (Asynchrone)  
    │  
    ├──► Flux Temps Réel : API WebSocket Finnhub (Gratuit) ou Alpaca (Demo/Paper) [cite: 31, 48, 49]  
    └──► Flux d'Actualités : Polling HTTP REST Alpha Vantage ou Finnhub News [cite: 28, 31]  
      
 2. COUCHE D'ANALYSE TECHNIQUE ET GÉOMÉTRIQUE (Local GPU/CPU)  
    │  
    ├──► Lissage du Signal : Calcul EMA local via pandas-ta-classic [cite: 17, 22]  
    ├──► Détection Analytique : Points Pivots (PIPs) & Algorithme Zigzag  
    └──► Détection Visuelle : Modèle YOLOv8s avec mss (Screen capture) ou image-rendering  
         (Vérification systématique de l'absence de "time gap" > 10 min)

 3. COUCHE D'EVALUATION SÉMANTIQUE PAR L'IA LOCALE (Local CPU/GPU)  
    │  
    └──► Modèle Ollama local (qwen3) contraint par schéma JSON Pydantic  
         (Calcul du score de sentiment, pertinence par ticker, extraction des risques)

 4. COUCHE DE NOTIFICATION ET DE ROUTAGE D'ALERTE (Asynchrone)  
    │  
    └──► Moteur d'envoi Telegram asynchrone (aiohttp / limite de débit à 30 req/s)  
         (Envoi simultané du résumé d'analyse rédigé et du graphique de signal annoté)

 5. EXECUTION DES ORDRES ET INTÉGRATION FISCALE  
    │  
    ├──► Choix A (Sécurisé & Simple) : Exécution manuelle Trade Republic (via alerte mobile)  
    │    └──► Fiscalité : IFU officiel de Trade Republic fourni clé en main  
    │  
    └──► Choix B (Totalement Automatisé) : API Trading 212 / Scalable Capital  
         └──► Fiscalité : Consolidation obligatoire via FlashFiscal / DeclarAid

En adoptant cette topologie modulaire, le développeur s'assure d'exécuter un système de pointe qui exploite le plein potentiel de l'intelligence artificielle locale35 et des algorithmes de reconnaissance visuelle de formes18, tout en maintenant un contrôle strict sur la sécurité de ses avoirs et la parfaite conformité de ses obligations fiscales sur le territoire français43.

#### **Sources des citations**

1. Outil open-source pour extraire vos transactions Trade Republic - Finary, [https://community.finary.com/t/outil-open-source-pour-extraire-vos-transactions-trade-republic/21566](https://community.finary.com/t/outil-open-source-pour-extraire-vos-transactions-trade-republic/21566)  
2. nborrmann/pytr: Unoffical Python Interface for the Trade Republic API - GitHub, [https://github.com/nborrmann/pytr](https://github.com/nborrmann/pytr)  
3. t212-api 0.1.0 on PyPI - Libraries.io - security & maintenance data for open source software, [https://libraries.io/pypi/t212-api](https://libraries.io/pypi/t212-api)  
4. Trading Web API | IBKR API, [https://www.interactivebrokers.com/campus/ibkr-api-page/web-api-trading/](https://www.interactivebrokers.com/campus/ibkr-api-page/web-api-trading/)  
5. Exploring Finance APIs with Python (Colab), [https://pythoninvest.com/long-read/exploring-finance-apis](https://pythoninvest.com/long-read/exploring-finance-apis)  
6. Quels courtiers fournissent l'IFU en 2026 ? ⚠️ À lire avant de s'inscrire - Meilleurs Brokers, [https://www.meilleursbrokers.com/blog/2025/07/03/quels-courtiers-fournissent-ifu/](https://www.meilleursbrokers.com/blog/2025/07/03/quels-courtiers-fournissent-ifu/)  
7. pytr-org/pytr: Use TradeRepublic in terminal and mass download all documents - GitHub, [https://github.com/pytr-org/pytr](https://github.com/pytr-org/pytr)  
8. ScalableCapital/scalable-cli - GitHub, [https://github.com/ScalableCapital/scalable-cli](https://github.com/ScalableCapital/scalable-cli)  
9. Scalable Capital - GitHub, [https://github.com/ScalableCapital](https://github.com/ScalableCapital)  
10. Trading 212 Public API, [https://docs.trading212.com/](https://docs.trading212.com/)  
11. Positions - Trading 212 API, [https://docs.trading212.com/api/positions](https://docs.trading212.com/api/positions)  
12. trading212-connector - PyPI, [https://pypi.org/project/trading212-connector/](https://pypi.org/project/trading212-connector/)  
13. New Equity Trading API in Beta - Try it Out in Practice Mode! - Page 9, [https://community.trading212.com/t/new-equity-trading-api-in-beta-try-it-out-in-practice-mode/61788?page=9](https://community.trading212.com/t/new-equity-trading-api-in-beta-try-it-out-in-practice-mode/61788?page=9)  
14. New Equity Trading API in Beta - Try it Out in Practice Mode! - Page 12, [https://community.trading212.com/t/new-equity-trading-api-in-beta-try-it-out-in-practice-mode/61788?page=12](https://community.trading212.com/t/new-equity-trading-api-in-beta-try-it-out-in-practice-mode/61788?page=12)  
15. Interactive Brokers API Tutorial 2026: Connect, Trade, Stream - Quantt, [https://www.quantt.co.uk/resources/interactive-brokers-api-tutorial](https://www.quantt.co.uk/resources/interactive-brokers-api-tutorial)  
16. Getting Started with the Interactive Brokers Native API, [https://www.interactivebrokers.com/campus/ibkr-quant-news/getting-started-with-the-interactive-brokers-native-api/](https://www.interactivebrokers.com/campus/ibkr-quant-news/getting-started-with-the-interactive-brokers-native-api/)  
17. Detecting & Trading Technical Chart Patterns w/ Python - Alpaca, [https://alpaca.markets/learn/algorithmic-trading-chart-pattern-python](https://alpaca.markets/learn/algorithmic-trading-chart-pattern-python)  
18. foduucom/stockmarket-pattern-detection-yolov8 - Hugging Face, [https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8](https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8)  
19. Automating Chart Patterns: A Deep Dive into Three Essential Algorithms | by Nilay Parikh, [https://blog.nilayparikh.com/automating-chart-patterns-a-deep-dive-into-three-essential-algorithms-9ecfafc710fa](https://blog.nilayparikh.com/automating-chart-patterns-a-deep-dive-into-three-essential-algorithms-9ecfafc710fa)  
20. tysoncung/crypto-chart-patterns: Machine learning for cryptocurrency chart pattern detection and technical analysis using Python and deep learning models - GitHub, [https://github.com/tysoncung/crypto-chart-patterns](https://github.com/tysoncung/crypto-chart-patterns)  
21. GitHub - zeta-zetra/chart_patterns: Automate the detection of chart patterns, [https://github.com/zeta-zetra/chart_patterns](https://github.com/zeta-zetra/chart_patterns)  
22. xgboosted/pandas-ta-classic: Technical Analysis Indicators - Pandas TA Classic is an easy to use Python 3 Pandas Extension with 200+ Indicators and Candlestick Patterns - GitHub, [https://github.com/xgboosted/pandas-ta-classic](https://github.com/xgboosted/pandas-ta-classic)  
23. Detecting the Engulfing Pattern Using pandas-ta | Adnan's Random bytes, [https://blog.adnansiddiqi.me/detecting-the-engulfing-pattern-using-pandas-ta/](https://blog.adnansiddiqi.me/detecting-the-engulfing-pattern-using-pandas-ta/)  
24. Using Deep Learning Neural Networks and Candlestick Chart Representation to Predict Stock Market - arXiv, [https://arxiv.org/pdf/1903.12258](https://arxiv.org/pdf/1903.12258)  
25. Stock Market Pattern Detection using YOLOv8 - GitHub, [https://github.com/abbasi0abolfazl/stock-market-pattern-detection](https://github.com/abbasi0abolfazl/stock-market-pattern-detection)  
26. ayhannbozkurt/candlestick-pattern-yolo - GitHub, [https://github.com/ayhannbozkurt/candlestick-pattern-yolo](https://github.com/ayhannbozkurt/candlestick-pattern-yolo)  
27. Candle Stick Pattern Detection with Python | by EAE - Medium, [https://medium.com/@elia.enrico/candle-stick-pattern-detection-with-python-bcd3733125e2](https://medium.com/@elia.enrico/candle-stick-pattern-detection-with-python-bcd3733125e2)  
28. Best Financial News API for Trading 2026: 5 Compared, [https://apitube.io/blog/post/best-financial-news-api-trading](https://apitube.io/blog/post/best-financial-news-api-trading)  
29. EODHD: Market Data API & Stock API | Real-Time Financial Data, [https://eodhd.com/](https://eodhd.com/)  
30. Top Free News API Comparison - GitHub, [https://github.com/free-news-api/news-api](https://github.com/free-news-api/news-api)  
31. Finnhub Python API Docs | dltHub, [https://dlthub.com/context/source/finnhub](https://dlthub.com/context/source/finnhub)  
32. Real-time Market News API - Finnhub, [https://finnhub.io/docs/api/market-news](https://finnhub.io/docs/api/market-news)  
33. Alpha Vantage: Free Stock APIs in JSON & Excel, [https://www.alphavantage.co/](https://www.alphavantage.co/)  
34. Best Stock News API for Traders and Developers in 2026 - NewsData.io, [https://newsdata.io/blog/best-stock-news-api/](https://newsdata.io/blog/best-stock-news-api/)  
35. Local Financial Analysis with DeepSeek R1, Ollama, and RAG - Sridhar Sampath's Tech, [https://sridhartech.hashnode.dev/guide-to-local-financial-analysis-with-deepseek-r1-llama-and-ollama-using-rag](https://sridhartech.hashnode.dev/guide-to-local-financial-analysis-with-deepseek-r1-llama-and-ollama-using-rag)  
36. Sorties structurées avec Ollama : JSON garanti et Pydantic en local - Stephane Robert, [https://blog.stephane-robert.info/docs/developper/programmation/python/ollama-structured-outputs/](https://blog.stephane-robert.info/docs/developper/programmation/python/ollama-structured-outputs/)  
37. Using Ollama for Real — Choosing Models, Writing Prompts, and Creating Modelfiles, [https://www.grandlinux.com/en/blogs/ollama-model-prompt.html](https://www.grandlinux.com/en/blogs/ollama-model-prompt.html)  
38. Structured Outputs - Ollama documentation, [https://docs.ollama.com/capabilities/structured-outputs](https://docs.ollama.com/capabilities/structured-outputs)  
39. JSON agents with Ollama & LangChain, [https://www.langchain.com/blog/json-based-agents-with-ollama-and-langchain](https://www.langchain.com/blog/json-based-agents-with-ollama-and-langchain)  
40. Send Alerts In A Telegram Channel Using Python - Tradehull, [https://tradehull.com/send-alerts-in-a-telegram-channel-using-python/](https://tradehull.com/send-alerts-in-a-telegram-channel-using-python/)  
41. How to send notifications to Telegram with Python | by Andrei Kushniarou - Medium, [https://andrewkushnerov.medium.com/how-to-send-notifications-to-telegram-with-python-9ea9b8657bfb](https://andrewkushnerov.medium.com/how-to-send-notifications-to-telegram-with-python-9ea9b8657bfb)  
42. Building a Push Notification service for Telegram Bots (PART 1) - Kanishk Singh, [https://kanishk.io/posts/telegram-push-notification-service/](https://kanishk.io/posts/telegram-push-notification-service/)  
43. Guide de la déclaration de revenus - Nicolas Cheron, [https://newsletter.nicolascheron.fr/p/impots-le-guide-complet](https://newsletter.nicolascheron.fr/p/impots-le-guide-complet)  
44. FlashFiscal Avis : Comment Déclarer ses Revenus sans Erreur ? (+ 5min Chrono) - Seqooia, [https://www.seqooia.com/post/avis-flashfiscal](https://www.seqooia.com/post/avis-flashfiscal)  
45. Impôt Compte Trading 212 - Fiscalité - Forum Finance n°1 pour les investisseurs en France, [https://community.finary.com/t/impot-compte-trading-212/14005](https://community.finary.com/t/impot-compte-trading-212/14005)  
46. Comment obtenir votre Imprimé Fiscal Unique (IFU) ? - RevenusEtDividendes, [https://revenusetdividendes.com/ifu-imprime-fiscal-unique/](https://revenusetdividendes.com/ifu-imprime-fiscal-unique/)  
47. Déclaration impôts bourse : dossier fiscal en 1 clic - DeclarAid, [https://www.declaraid.com/declaration-impots-bourse/](https://www.declaraid.com/declaration-impots-bourse/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAZCAYAAAB3oa15AAACRElEQVR4Xu2WT6hOQRjGX7mKUmJBNnStSbGS7CiSu9BdKGVhZaEshJL0FbbKn4VkwQp1U3aWspAoK7orG+neWLCyQP48PzNv9z3jfN+Zs9V56lfnm/fMzPvMvDPnMxs06L/TRnFarC4D43RO/K7kcO5zVXwP7ffFVI6VWiZu2tK79KP/OB0RP8XeMtCl89ZM0rVc7BHvLJmNuiU+ikWxpYi5psUL8UM8tckryyLMWcrjehHrlO/EoTKQxcrcLtpuiIvilzhZxFyz4oL4at0GWITX4puYFxua4clqM7BO7MjPDE7CsVT4vV+8tfbkVloql4NWZ+CouGJpF1gUxq5Wm4GdlpJETMwEJOUixjsjS6u2K8TQVnHK0jtdBliYu5bGYB5y4exwhqrkBo5Zugk2W1q9e/GlQm7AEyzr9qylhGoMsMOPxJr8zLnqVUZu4LN4Lz5Yug1qDKwST6w5IYlcyrEaA6z6KD/7Ye5VRm0lROc74Tfl01ZCyLedw47oezw/dxmI5ePqXUZtBuIZQGfEgfA7GuC6ZNf4JqywdC3ThroMUDL0XbC0+/DJUj7VZdRmIMo/SNtDWzTg8S9iRlzObajLQCwfV+8y6jLAaj6zdMBd0QBiIiZ8mZ9dkwxQkg/t3xsM9SqjcV9iRPKPrZkAh/OB2Jd/Iw4uX91XYm1on2Rgt3gjNhXtqOo2OmFL9eZ4LfpN5O1+I10r3ue9bTnGTo7y83rx3JpjcMtxPRMj8ThO3DXyiv+3GIOx6Ddo0KBBgwb91R8P0LGNhSYz7gAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAxCAYAAABnGvUlAAAF4ElEQVR4Xu3dX4htVR0H8BUl2F8pM43+iIEPSWF/MBKrB6kospCIChV68MGwntIKIum++Bao1YOIohZYUQQRoRHIBYOiXosCjTS0qDBf7KGkbH3de81Zs+6emTM6M56rnw/8mLPXPnNmn70P7O/9rb3PLQUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABOGi+rdeo4eAReWOuV4yAA8Pz1vnHgJPHBWpfWesG8fEa37qDcXuu0cfAI5D1dM//cBK8t0/YAAHt4uNZ93fLptR4s08k03lnr31trV75U68laHx1XVO+p9WitN48rNtRLyxSiLuvGrqj131ov78aeqQSlG7vlF9e6qdYjZdrPR+WX48AavlKm492kW/fHWl+el/Mzyzn2vQTTfH7a56n3w7L9NQGAHTxU63i3/KpaXy2rE2+CxFW1Prb1jEnGlgJbptx+VOuJWvcM6zZRQtT/an1qGM+05fFhbDffHAcWJNT8dBws0/hRBrbLy/6nRrONLVzlM/KOWm8qq/ed9Z8sU9h/3TwWXyvTZ2wpsF1fptD2oXEFALDdGNgSHDKd+bZu+Zxa3631onksISdjS4EtQe+2Wr+p9diw7rDkerBsx5njijW8t0zvPwGtl+UfDGO7WSewJaylUzU6jMCWfZF9snSt3Pm1LhgH99AHtmxr3m/2UcJfW5+/l/D76Xks23CsLAe2fJYuLNPvf6tszjQtAGyknEwz9ffnWv8sJwaHtpwT8efnx5+Yfy4FtnTXEuYS3NJly7TfOq4r0zbsVOnqjE6p9ftaZ8/LCQpvKVOHcB3pBD1YDmbqdp3A9rdy4pRhHGRgy774bZn2TSRo5xj0+yRBq01lrqsFthyLfF7u3L56K7AlfLWgfqxMoW0psLWgl+CWz5YuGwDsou+w5cSe4JBuR7t2qwWJdMx+NT+nXYe1FNjSXctJOFNu+Z0EqF4LEgfh42XVzYls6zVl6tz08jez7a8fxvP8f5W9pwdfMg6UqXOVENIq77tfPn311C0JO0vBbLfAlmPRv25fS92zW8t0XWKT104YyjV6TQJbpiNHudmi/91e32HL6+XvRNuGFthaUI9Mh8ZSYLuje5zXTdDr5brCveQ6utYJBoDntHFKNDJddsP8uAWJBK90Tq7txsbAluCT7lqTk3d/HVu6LTt1sxIixkDSV07Oo2x7H1ryvH90y71MA47TbrkG66/lxOnQuLmsuoO5Xm8v63TYnk5g268ck357c3x+UrZ3OnfqsL2i7Hx3bx/YmuzPFrRaYIt0YhOk235dCmzHuse5ji1dtt43huUlef3xHwwA8Jw03iWablS6Tu1E+IGyOtnnJNymsmIMbHeUE0NRf5JfJ/jsR7o8bdvSPcuJP1N15249Y2/vqvWdsuroJLTcslr9VMB8dbe8k3UC2+/K8j5I2NnvNWU7ub+spq5zLP5QpmPUdxfz+P3d8jrGu0Tjx2UV9r9e68r5cTqsCcLNGNguqvXGbrkF5wT6yP7eKdj3BDYAnhcS1HISbtcmpfI4gS0dn4S5tj5yzVfroOV6t4zneqa31vrcvHx3mcLPa2r9eh7Lcz9bpkB1kM4q03tIwEroShDJNmeqdD8eqPX3Wp+p9fNa53XrElDHELpkncB2e9neOcp+avsx9Zdu3dN1ca0/lWmKNtPA+R65HIe+05nOZ38n515y7PptzD7Occ9yAvLV3foWBO8q0/v73jz++Pw419f1n6n4z7yc18znJtvXpuTT8cvx7auFOYENAA7Bz8r0/Wb9tOmmu7fW22t9ZFwxuGQcWJDr2n4xDj4L8p42VbpzCfZvKFOA3o3ABgCH4Hg5+e4G/H6tL5SDu1ki32e37p2zhyXX522qdDNzU0vuat3tf4P4cK1vl+lLgL84rAMAeMZy48a7x8EjkGnGpbtDAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADbT/wFbbfTUyp1elQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAaCAYAAABYQRdDAAABYklEQVR4Xu3UTysFURjH8Uf+REgiUWTCBq+AKMnCUthbsmWFJBtLhY2NhJ2yldgQxcI7YCEl9pIFhe/TmblmnrlmJmzU/dWnZs5z7+mcM+cckUISUoUBjKMTxX57JZr958zpwiWesYcZ7OIY3TjEUO7XKSnFAl4xi4poWfrxhHvJOFLtcANvGDO1IOU48OlzaqbwgTkUmVo4O5i3jfnSgQfcoMXUbDYl43ouiRvlsmnPlxpxS5UY3TaneJeMI8iSJtzhEW2mlhZde92zsQSdKn1Oin7M3tB7j7idoLONpBZXkt5pHbbQEGrTQ7Eeeo9kRdyaDtuCH52mdhDsXw9ruBX3PbRW5tdyacU1ztBoanqqFjEt0f1bjxNxd8K38XCBF2xjQtxozjEo8QPRhyNUm/ZY9I8eRnzt8nUz2UxKwnr+JCXYx6i4E6iz+nV0RqviPrDeanrK/iTasa6nXetC/ks+AQp1NPx3+T9VAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAaCAYAAABsONZfAAAA1klEQVR4Xu3RPQtBURzH8b9QTAaPhWQw2I2KTAxmyguw2AwGXonIaKAMSl6CzAZlsFgtyqb4nnvO5eAVKL/61O3/O/ec+yDyz68mjjrKCLxX30lgjgWa6OOEgumTyJlrJxnsMITfzHyYYo0gBiiazilHonfNukOTHi6oYoKQW+RxFv1oagM7LVxF39CwC/XSd7TtoYnbLUU/4jM1U6gFn1GzG0qfRRoHdKyZBxXs5bVhChFrjbPgiBnG2KCLKFbYiv6K6iu/xYsYwqJPsufqBPdX/PMjeQBZzSE9o6921QAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA9CAYAAAAQ2DVeAAACUElEQVR4Xu3dPWvVUBjA8YgKFcWXxVbopB9ARNTFRQfRoYt0E/wKgg4FJ2cVRCoOviBOLuImOEs/gIOgCC5SEATp1ElEn0OiOffALem9rYnw+8Gf5jzJBzjkpklVAQAAAAAAAAAAAAAAAAAAAAAAAAAAAACDsDN6GC1Ge4pzAAD0bHe0HB2L1qP3o6cBAOjbUvSrOZ6PVqPZ9jQAAENyNvoRzZQnAAAYhmfRWjkEAGAYTkenyiEAAMOx0vzdFR3MTwAA0K+56G30pelbVW/aAAAAAAAAAAAAAACAfyh9BupddDX6FO2o6m93pm95AgDQs8vRz2x9MroR3c5m00qfmOozAID/2uvoa7Y+Gr2I7mczAAB69Dl6nq2PRHeydS79dLq4QWfaSwEA2CpPojfN8Xz0sqo3cHv/XgEAwCDMZsfpE1D7svVWOVe1z5Wlf2pI9mezV81sWieiC+UQAIBu0t27a9GDbHY9upKtx7lbDgrpJ9mFaD1aKs4BANDRvap+Tm41mz2OjmfrcZbLwRg2bAAAU0ivDUnWogPNcdeNWNfrbNgAACaU7qKll/ImF6v69SHpWbZxd9fStelu3J+eFuuZ9tIRNmwAABPKn1NLd9fSXbbz0aFsnjscPcr6WKwvtZeOsGEDAJhQ/r635EO1uS8R+EkUAGCbla/tuBV9L2Yb2cyG7WY5BABg+3XdsAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAPfgPdzWRGr+f+OAAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABHElEQVR4XmNgGAWUAkcgfg3E/6F4BxBzIsnzAfEuJHkQXgfE3EhqUAAjEM8C4l9A/BOILVGlwSAIiNcwoFqEFQgC8UIgzmeA2DyFAWIBMigC4mg0MaxAH4j7gVgSiK8D8RMgVkSSZwHi2VB1BAHIxnQou4EB4rocuCwDgwgDxOUgHxAEfUBsDGXrAPF7ID4BxPxQMRsgngxl4wWw8ALZDgIgLy0H4n9A7AEVA7mapPBCDnCQISDDQIaCYo+s8IIBkPdA3gR514mByPACuQYUFqboEkAQwwCJiGtA3IkmhxWghxcyEGeAJBOQgUSFF8gLoKzBhS4BBQ1A/BaINdHEUYALEH9hQOQ1UBbyRlEBAaBkAsqrBMNrFIyCIQMA260zNBT6yKgAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAaCAYAAABozQZiAAABBElEQVR4Xu3RP0tCURjH8SesLREpCMElaWlyiAaHwlfQ4KQvIJpak6Shpa1AHNsaXKq9wSVSaHDuBQRRU7Q5+/15nlPnRr2B8gcfLs9zz797rtk8/yk5bGEXS0l/AXl/KhuoJvVscBcdPOAiviB7+LAwYRVPeEMlDqjjBAUM0bevlXsWJmiiekd4tWTyPjZRwwRN7xcxtuxiZdxZWCyTU7xg3Wst+I6DOMB755Z8s7KMe9xi0XsNCyfZ9lrRqVpJPUsJzxYuLabtPb1TtKh2jSf7TLzJM6/1B64te7M7OLZvR47RMfWNNxjgECM84gqXFv7Ir9GOaxbuQNEuK+7HHef5+5kCasIlUVci1xkAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAbCAYAAABIpm7EAAAAuUlEQVR4XmNgGAWDFbACsTgQS6JhAWRFIMANxBOA+C8Q/8eC90DVgAE/VOA8ENsCsTwQLwXit0BsyQCxgQemmJEBYvIVIBaDCQKBMRB/AmJPJDEw0GSAmJSDJm4DxL+B2BdNnHQNIIFvQGyKJp7OADEIZCAKAGl4yADxGAxwAvEOIJ4DxCxI4mCgA8Q3oDQIgAKhDIgvMkBCCwOAFGQB8RkgngXEB4B4JhALIanBCjgYIM4C0aNgkAMAUTIgX1iegXkAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAxCAYAAABnGvUlAAADQ0lEQVR4Xu3cTchmYxgH8GtkRERRJBbIZsqClLKgyIIFShYSpXwmEYqys1BKIZGvhSik7ISSMkmx0GDFgpKSkCU1Uz6ua+5zPPd75pl5zut95Wn8fvXvPec+H/c7s7q67nO/EQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwDrakbltOggAwPp4PPNH5o7phc6xmR8yZw3nVeTtzRz59x0LH2QunQ7O9H60eUZXZ56N5fMAAPwvVOF1Q+aYzJ/D8TKXZR6ZjPUF3DL3ZD6P9u65qnDs56n3r5oHAOCwdnbmpOG4iqVPumu9BzNXTMaqwFvV+dqZ2ROtQzdHvbOf57rM67F6nv9a/TtP7QIAsG3e7I6viVa0ndCNjb7NvJR5IfNG5sfM8f0NK9S9L2dOnF7onJb5OTbO89iGO9ZP/V99lrko82rmwlCwAQDbrP9urYqP6rBV4TZVhdzYPTolWkdprirWPo3WLTuUWnZ9MTbOsy6+ydw9Gavl5CczJw/n58eBXUgAgC2pZc6pKtqqOOsdnfllMjbH09E2IFRhM8fbmV3TwRmemA78Cy6OAzuKV2Z+685vj3/2+wMALHVXZl/muyWp78guX9y6//ip7nyV6pKdOR2cYVoojl6J1tF7L1pB+W60TtZYcNZS6+imaMVVdeeeGcZqI8URmdeivef7aF3Eh4brVw3j9T3fZrp652S+Go6rKP2iuwYAsGVfRivMDpa3ou3u/Kkbu2T/k9vvxljMUztC+3lqs0F9G3ZtLDZEXBBtJ+m4LNsXbNX1KrWr9Otoz1Vhd9zws3yYOX04r/fXd3x1XxV2taw5VxVpd0b73m53HPr7PACAtdLvlpxm7vJob9wlel6052/NPJd5YLheBVsVZGUs2KrYfGc4rmeOikXBtjva7zKeV7dx3IV6xvATAOCwVh2ng2UsrDajlmPrj+feG63YqiXNhzO/RyuwPsrcnzk383Hm0Xoo2hLnfdGWPuv7str4cH3m18zzw/nN0Yq7ev8tsbnNFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALBFfwH/GnhRogFeUAAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAAAaCAYAAADcx/BtAAADRElEQVR4Xu2XS6hNURjHP6E8Q+QRpesxEEVkoJQJSh5JSChFSlLej4xMhIHkMRERkshECkm5ZWiAASMKiSKUuhPv/9+3Pmeftfc9zt5n7X0M1q/+3bPXuud8e631rf/6lkgkEolE/lcWQL+a1A73nRCMgF5IOkaW7kH99Wst0664f7gAfYNme+09oOnQK2iZ1xeCxaIDOuh3gKHQNeiy6HuEpPK4Q6CH0EtodH3XX85Cc/3GAHCQHOwSv8PByTjhNwag8rhToS/QdaiXa+PfaVBv93zc/V9IuBW5Jd9B4xLtE6Fh7jMHG9KmSFvirhFd1b2JNmY0s9c8iVYxuNYdBA6QA+2EBrg2bs9jUltQLjQVkrbE5WR+h5ZCo6Cx0GnJ9quQ2KF7SjQutU508LSwsqg8rvnxD+gN9Bp6757L8OAk5ouMx7iMz7hBvTCDyuPOgLqk3o8Hip6u5lc9XZtBC1kvrdkHt2mnpH1xv6h9GYzL+H2hDdAhaKN7LkLeuPOhFdAY0aTjbs+N+XHS5FlLnoT6uOdV0GbR0uYKdBN6JrrNimK+6Neih6FJ7nOHqJX1g45AK0W9k5/5HpYUecgTlwvJc4rzw0y/Co10/9M0fOHzkl0fG1z5S6KBDWb/Y2ltklk6dVenGrtFF5iTeRE6J5pdHDgtroh/5olLNomOtzDmx7wBMXuzWCua1cmivNEks+Qb7v42wnyRh1AWE6C7kn4vZtctaKfXXlZcTvIiUZtg5ZH7cpLlxwa30j7oAzTL62s0yUdFB3HAa0/SnS8SZuo86Lmkf2M19Ag6I+nrbllxt0O7RK2Tvr1HmpxoGvhb0Zcyv7HKgvqa6Lsj6UOm0SRvg36Kep7VoAYz7YHU/76d8NSnRPtnaIp+LcUW0d8flGirIu5M6KnUW2dpNJpkwlsTLcbPtqLQAnjojXfPvI1xQv0SM3RcO4+Y4YTj5qK05NHN8q9Jpr1wW4WCp/5HqVVAvH3y2aoBI3RcLtoNqd0CuahPJH1OBIW+xJP3tmhFQi/3DxCuPquAyV57KzCT6blbRaug+6KWkfTGMuISVhn05eWifr6wrrdNMLvm+I2B4OWHu8f3XFJFXLs3RCKRSCQSSfIbD5jm2Sfw+TMAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAaCAYAAADi4p8jAAACtElEQVR4Xu2WS8hNURTH/0IRQkRCHpFEEUUirxgYkJQSSqZMpDwnlJSJPFMikkxkqr6+DL5SSgxlIAqJoihl4O3/t/Zy9t333M+5555b0vnXr3vP2vecddZej32BWrVq/UtaR34WZG+4pwqNJ8/Q7COPO2SY3VZe18hXsiyxDyALyAuyKVmrQuthQRxPF6gx5Ca5AXuP0hpNHpDnZGLj0h9dJmtSYwVSYApwQ7oQpA04mxrb1Tzykdwig4JNn/PJ4HB9JvyuSqnsVH5vyPTIPpOMDd8VYMetsQ22iwcimzKprHntqzxHZcuVSEEpuD4yPNhUiqeQbaY2WXQkBfKNbCQTyBRyEfl9UaV8wJ2H+RU7YAGrbSqR99938oq8JG/DdTd6Lpb3n/zJr/zLb8c9F2sh+YTG/hsBm17eFwODTZoEe7ELZDnKTzeVZB+a++8wrGVc8iv/LvnbgzY23/svbmSdUefIkHC9heyCBXeEDIUNgidka/hNu/L+S8+4E2R2+D4N1j7y51oMS4iGz1+l3biK/PPPpZ2+DnOmh2ra+gBQJtMXLCodC63OP9c+2Oa6RpJj5CEKBuj9p38UylqetsOyqc1QIEuRHR3qldvIMi1pbVz47E/efxo0eZpBepG9l/zvJItgpV0owLz+cymYg+QdWZKsSZPJIzQf0CdhL340scdq1X+S+m0teYrGZ6hqlE2/t98A1aCvkf3Xiyeo+BKt9aCxByRl5zTZjOYhowHwA1a6fra5lNm7aHy+T1DxPrJ/IHPttt/P0fDRZ6EAO5GCk7PV4VrDJs2+/oWorMv0Zp5WkfuwwO7BNugxyg+4llK2dsN2T4fyLHIo2GOppPcntqokv/rj35UMroSVs5eRUDZjqYSukDmJvQpNJZfIZ9gAajWguiqdYStSY61atWr9N/oFrTibE82iczgAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAZCAYAAADAHFVeAAAB+UlEQVR4Xu2UPSiFURjHH6HI16B8hIFIMiAhRZnEoCSLGKRkoXwMPkoxGEjKoGS6MjAIA4OPiYnJaDGQshkUm/j/7znnvuc97+11ze6/ft173nPe87zP/3nOEUnKUzYoAKnuhCXO5bgP/6o+cAMWwBEo8k9HxUBzYNqdsFUMFsEOWAaV/mkpA/egTY+HwBtYAlWi3u8CF+BclANx1QwuQTuoA2fgG8yAFL2mGzyBUj2uABugHPSDQdArKuMGvSagTHACRsSrQz64Ax+gUT+bFRWMGVD83RV/BmOSgH3c5F1UVkasC7MzL7NedjBmuA0y9LgW7EmIfVQ62BTls9mIYiYMxl+qGjyAJj1mfab0f7oTkRD7wpQGDsEX6NDPWDvW8FqU5afi1Y/Zh9oXphZR9WJnMnNb7NIekKvHzOZAPPvowBoYkOC7AeWBK1H+ZzlzrhiAgYx97OpbPR4Fk+J1c0D8EhadLc06/CbbPlq/D1b0mO9zrxI99skEmhfvCNSAztgKv1z7CsGjeA1FTYh3dGIyxWd32Wnz3LDlXbn2UeYI2cHGQas1jm4+DD7BC3i24HVkridbtI4fYosXL+9NE4z7rotyJybzRTxTLq+iriVbzCYi8WvKQMeiSsKuZVnirUtIvC22JHhJG3HjVVEZ8q6t908nldS/1A/8cFWR9iYg/wAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,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>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABaklEQVR4Xu2SvStGYRjGL0mRr8TIIvkohZj1DnplkWIQo8GiJMVKMlAWo4mk/AMmhrds7BakJCaJsihxXe7nnPM8p/Pa3u386lfnuc99388nkFNJmugyPaCbtJtWBRlGD92F5c3RuvB3wgAt0QJtoYv0i64ibDxNb+ggbaBb9Jw2ezkx+/SbTrqxGl/TV9rnYh30ls67sYjylrxYzB79oQtu3Egv6QdsF0LNPumwGwvt4gS2S608oIa20Wo37qdvCJO1m3RTcURfaGcqHqAL0+yPsLOLUHG5plnxP+rpKazZPR1HsnKttoTs4n+b+vTSZ3oMm0xeILs4s6nOc8Wp74h12OWpSGwjoxj2/4m2+0ElKTldoGQ11QUJPTc9u7E4A6ilZ059x0Tv7xDJI26lV7AXMJSKbbix6IKtctaLxUzQO7oDS9DM7y7uM0If6BqdgT181fjHFqDlF2DJoyifqEsr0inYLnNyKs0v7h9KJFrzeeQAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAZCAYAAACsGgdbAAACgElEQVR4Xu2VTahNURTHl6IIScpHkZeUJEW+UlJPGCGFEmWkSCgUkYGSRCkZKB8lA2MjIwYXJWEoBpRIKWIglOTj/7P2fm+fde65TyZS51+/unvtdfdee6291zFr1er/0DhxRcyNE9IscVpcFFvEqOr0b2FjDp8TYmp1ekDDxBJxTpwXy5NtSOF0WHwWC8LcBvFEzBNjxHFx0/xQWZPFbXHWPLg14oVYVvig0eaJuC5miznikVhVOjWJk32wepDTxDOxtbCNFw/F7jQeLi6Lp2JSdjLP5n0bPAyJ4BB3Ctt+8VMcSuNG8YdL5mWKQRJctLHZNdExzywZeV+Ms9aKH2JlGrMGa+0Y8PCsc5gZha0mNuQ0lJTTxIC4N9GGroo35ovnzTtWD7LM0jHxzfwK4DdFjExzPbVYnBEjrHuQBBNt0T5UkFwFgrlh7ndSXDBPDldpl/V4OJSZEvelcQySDTvBllUGyTrcvbtibOFDGQkS37xWDpp7jPrFR7EujSsi8n1iU2GLQfISbwVbVsww1+Wd+QNEfeYPKQZZ3lFEyV+aZ7lW+vk2WOasGCSKwTTZOTRt57n5prQYHki+k00HzkECvyvaKV4FPpkv+ta8dBPNSxYXRgT52pobNiLIMnOUOa7VM8hu6pZJ7sp3q5YoP4KyRHvMDzY9jcksTbvsk5vFV6s2+J7l7qYj4otYVNgmiAfm7SNrpnkW2TSLzBLA0jSma9CiygdBZR5btXH3W4+HU4osUWJKDWTunvmiaKH5J+6g2Gj+tTll1fvMJnw6t4uj5ofYZvXWsiLN0Yb2pt8HrO73V+LirxbrzT+V3cQnkd6IH/5N+pO1WrVq1epf6hdX0aCw1LRU5gAAAABJRU5ErkJggg==>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB8AAAAZCAYAAADJ9/UkAAAB/UlEQVR4Xu2UQUgVURSG/0cJRokEUQhF5CIRWhShUZirkCQKwUTQNtEicRsouZU2boQSBA2kXAgWuGqVYLsEwZXlSigRXKmbAlEw//+dub5778zgg6BN74ePmTn3zvzn3jn3ABX9r7pP3pNrpC7iAqlK5nWRO+QMKZDz5Am5now7nSI9ZIKMkIZwONQg+ZPDFmkkJ8lsxvgHUouSdP+ZDMOSVGLfSac3J9BbMgfL1DFJfpAB2CrdvBWyTj6SB+REMuakhSyRs16sl6zCdjGQshsn56L4LTIN20KnN+Sm9xxLhjJ+F8WbyC/yKIoXPy4jbauT/qVWdtmLSceZ6/foN8Xmeuc3eRXFU1ISr0l3PECNkVGyTDbIV3LDG3cmeeZxPKW7sILxi8hpirxE6T+r0rdJc/L8EFaEsUlZ5lr1DGzlWapBWGAXYTugd/RuO/7CXOd8B1ad5Uh94CdZg1VynklePNBzWObavlhPyQHp82LOXOi+nmwibeLMh6J4IJ3jfdISD6DUiHxzt+1fYEdW6P4TqT6aBdwje8k1U5qsl5Rh1nG6Dat212olnYhdhN1LRagmdCV5LsC63SKyi7io02Qe+eb6yAuyQJ7BilJnuj8Zc1Jyalqa1wEz/oZ0/09JTUVbHrdMX5dgH21D/kqUzFXymLQi3K2KKqro3+oQVfdpz3noVoQAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAZCAYAAACsGgdbAAACSklEQVR4Xu2VT0hVQRjFj1igUIgoipCQEoaLoAgNw1pFGFKEGUkGIUKE20CxrbhpE6QYhBAuJMi2rgQzWhQELiJrJZIEkVCrAhG0znnfHZ079933IAVd3AM/3rtzZ+535s83H5Ap08HXcdITNka6Sc6TI6SE1JA75LTfiSont8kz8oicjL/OSePPkSdkgnSS0liPQM1kgMyTTTIVf53TIfKS/A2YIRVeP/2fIyOwyWgCn8kNr48MDpI3pIFUkWnYpA57/WKSyeuwVfqG/CalSfKJrJJXyD/7IfKBVHptveQLqY2ez5IfpH27B9BIvpIOry2v6mAd00yOwQKkScZkMBzfQn6Ta9HzKCyO4jkdJW/Jc9hKp2q3JrUjP5EcrzF/YObKyCySJnU0FpDchYSKmRwnj8ki7Fi8I2e8985MON5vd2bSTIbtCRUzqa0Yxs45VGb/Iq3R81VYMoXjfZMuRmhmz0zq3PiJcgy2oi9g2X8FxU0qeZaRNLNnJkO5/gqq4Pu+3X1ki9z32sKt0zXyHcnxzuRD2Irr+grNOJPKcO1YqgqZ1P2nrfRNuu1egAVxgZS9ymKnS2Qj+pX0Ld0Cug2cqskSrAIVlDOp278keNcGy26/Itwi64hXEyWTLvuG6FnfUfV5j53KdAI2Ob/8XiBrsDh5pRlqkEqiK3e6fD+SU1EfBXtAXpN+2Iy1Giqn/oQ0iadRP1UxGdQKhfW9i6yQe+QurHSG3/pv1cOCX0a8ZvtSoCbSTS4ivR6rZuvaEvqfKVOmTAdN/wCwpY+Q1EuhAgAAAABJRU5ErkJggg==>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAZCAYAAACLtIazAAACxklEQVR4Xu2WS8gOURjH/18ocktEQh9SsqKICCuJRMJCLCyEklIWlBKSBXaUJBElkbIixeJzSS4LFJJLLilRlhaSy//vmWPOzDyn90jJYn71650578wzz3PmnDMHaGlp+R8ZS1fR0bQX7Usn0/XFccxEup8ehd3Tr/p3kkF0M+y+HXRk9e9fdNEZ9CA9TBfB8qmTE6vBLPqF/oj8DHtIzHL6hE6hA+geeoUOji9y6KYP6TpYpy2kz+j06BoVuJVeo+PoUHoaVkif6LqcWC5TYck/pQ/objR7Zwx9TldHbUPoPbopaqvTmx6j54vjwF56GeVIUA4f6OzfVwDj6Ru6oDjPjeWiBxyqN9ZQcXq7ujag3ldv98DerIcSfU+31dqXoRpPiaqguHMH0hv0BOxZubFccorUPPECnYQ9WAl4zKPf0UxsMWxaqPM07C6iWaQ6rgc2WjRqcmIlUeIX6Cn6gr6lO1F9/SomVaTXHggJpBJTeygmVWRoz4mVRAneoROKc036uygnfXiYV0ynIvVgL4E4MRWgQjoVmRMriQrpX2vbDltxZxb/XYVfTKcit8BPIE5sBH2JzkXmxPojQq8psEgVk2oPpBL4p8NVQ1MT+xEdHrXXh4ZWP68YFfkOtpHw0CfhK5oJhMS0MupzoM9CqkitsFppc2K5hPlQL1LDVTcuKc71+w22wgXCqijrO6PAKPoatjrHbKCf6KTiXInH52IYfYzy3txYDdSLR2DbqYB2MNdh8zDsZsJitKs4F1qo9BZXRm11umA7o9soY2kNOEfPoPyoe7Hm0I+wdUHkxnLphg2JA3QtvU9voTkEp9FXsO3XCtgw34fqtstDCV2iZ2HbsOP0JprxNdwUX3vmNbBd2EZYcYHcWC5KdC4seb12b2MstNLOp0thW71cFE/zWfH1m4qvEaM5JnXskRurpaWlpaXlb/gJEMXG5uRLTIIAAAAASUVORK5CYII=>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAZCAYAAACLtIazAAADCElEQVR4Xu2WT6hNURjFl1DkXyJS5CpEFELKQC+FlH/5lzJQJB4D8QZKSQYKITwDYfAkfwbKSJTJLTPKUAqJMEMUSYm1+va+55zPufedd+/t1tVd9at39j7vnLP2Xvv7LtBRR63QUDKRDPMTTqPIYD/YDppKHpMj5CFZnp2uaCm5Q0b6iTyNJgfIFXKczCSDMnck0qptJSU3Xq82kNOp6yHkGjkRrqeTN+QuzNQkMpdcJO/JgnBfTc0jZdJFxpI95BfpQWJ0OFlLLpAP5DtZGOYaUYm8JtdTY4qoxjaGa5k+DzO2kuyALcxJcijc06+0Ir/JunAto0/JJzI7jMnkathCHENzTOrMacf+IGtSz9XztahRvWE8ag65gYIxlc7CXrQrXOsg6zx8g+2y12E0x+QW2LsVubTJyWEsmlSalKD0gvehYEyjtKLjkVQoxeILLMJ5K9UMkyXY+Z9F3iJrUtX0HqzoSIrvZSTfouNUOKZ5UgG6Sd6R+W4uqlGTWtQzZAmsiHiT0iLynOwnt5AcJe1e4WrqNQL2zzKnQ78K1XtPoyY3wXZCMaxmUtKCK7KqrpKM6RtjTGNHOApL4oCkCH2EHWyZ92rEpPrfVTImXNcy6aWFiTHV/z+AVVolIv3MQtIKK7IqRnvdnFSvSbUClX19VFRRkz6m28gTWCeQ9iGJ9D/S+TgY0N9RMuJLe1S9JifAqraORESJ0XvUl3Wdt6g+ppJaTznMSYthRnMVe5L/aJnTy9VDvWqZ1DmWmf5+b0YV2cl0TKN0fxmJSX1LT2XWaQp5Ces7MdPjYFFQG8nrRTL5A7Z6Xuq1WpzbsHj2p9gTdTx0TLz0/j5Yb0xL7aWMxKSiursymyP9knlFTsGyfp98DeNRsfp+hpmIKG7nUvepGv6EVWjtUjXF5ymm8Vl6djquMqDip183Xtq5F2QG7JhdQv59GSleXWQzWYbs+Ryo9KxeWBNvRDvJdj8YpF1fQ56RR6Q7jLVM02DNvkhc21IqPDK4wk/8T9I5XI8WR6ejjjpqH/0FfuiaO/7Ct7AAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAZCAYAAACLtIazAAADPElEQVR4Xu2WWahOURTHl1BkjgwhMwkPMpWUElLG8CCECBmiKErEi5I8mJJQrmQIJU/GdMMTkmR6IJFShBdKSfx/9t73nG9/wznfg1tu379+dc/e++7vrOGstcxqqqkx1FJ0E63ijUjtRPN48X9QH3FXbBPXxKTC7QaNF+dF23ijkvqKBfGiF96aLo545ovWBSfya4jYK46JxaJNaq+FOCF2++eB4o24ZM6oHmK4OCjei5H+XEUNFWvFbfFLnCrc/itS55DYKQaIleK7eGrO69Vonrhg7nd7iQ3igSX3kKKvxVz/jNH7zRk2VSwVc8QescmfyRQ/xj/hJTxTyshp4pbomVojAr/NeZ0XySMMIP0wLqiZOCx2+edR5hw4Mxww52DWg4aJ01ZlmiLS4K2VNnKrJQYF8aI4Ba/z8nnEi74Ug6J17g93h3uDkTjhgLlgID6ROsuZprEqGTlaPBarUmvhPPB3HvUXH8QrMdGvdRA3xCz/TDW9bK7oIBx41JKorbYq0jRWJSNLaYL4ae6Fssp8EFHZYi4r4KS4Ijb7vSCc+lysE2ctcQDRq7qaplWNkRQiXvCrGBvtZYkqTeUMhn4SU6zQSNTeXMpSXRGGYWBIU/Y3ih2ii1/LVDVGUiFJu8nxRoYwZL259KTQEUUMpaovSZ0rJVI0pCkpftVcpR0njvu1TOU1ksg9EmPijRwiYrSd3v6ZqC40V01fWPkCFqcpvfy+6OSfaYEhpSsqj5EYeM+SFKJ10Eo6NpyoLCooTTwWLeqLFbaJoDhNEffU+z2EwzE0U1lG0qyZOtLNn2+BqYX5ERGZrla+EHF3mGTSom08tKRNpJVO0yDuqbfESJxD8cpUMPKMFReB7uKOOW+/S/HR3PQShoEV5r6xc6m1tEipZ+bGxyB+a5m5Khr/D9Grs+LxkfZSb4mR3MsUVlYUD5ovH3+oeN/EEzHCnwnDQCnSkaEa/jA3IJTqnUSaFvJZbBeLxEVx3Zwj08IAphqmm1jpoYJKz8RU6tw/E6nKGFauiKDO5hzCkD/YijMHLTfnhFLi/AxzBfCmWOPXGk39xD4rTr0mI9IRA2kVTVZ8h7OtkVOnpppq+n/0B0nknhokt1S2AAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAZCAYAAACLtIazAAADRUlEQVR4Xu2WW6hNQRjHP6HILZFLkUMuCQ9C4oESotzCgxAiuUZRlCglhTy4FSGX5BJK8uBW2uUNSXIrJKU8iCeKB/H/9c2cPXvZZ+29PZw6p/2v38OambXW/Of75psxq6uu5lB70Vt0yHZk1EW0zTa2BA0QD8UOcUdMKe1u1ERxRXTOduSpQSzKNiYaJg6Ik2Kp6FTaXZPSby2wYjTaidNib3geLD6I6+am+oqR4oj4JEaHcbkaLtaLB+K3OF/a3SgmctV8fD+xSTw2X/VaRBruNH93jBgkbom1oZ8UfS/mh2dMHzI3Nl0sF/PEPrEljKkoJs1LrBIrU84kPyZtMBfVRhwTu5O2aoSZt+ZZg2aKP1b8L8Z/iNnhGR0N7VEjxAWrMU0RafDRypvkB2/EkEz7dvPUqlb9zQ2SalHdzCM7PjyzkCx2NMliHjYPBuoozlmVaZpVnklS6rN4JyaHNiZ3T8yJg6oQ+52oLTGvmvwzGw3ab5gXHUQWnbDiuDVWQ5pmlWeS1dxmPkE4K26KraGvWhFB3t9jnm4bxDOx33yvRo0Vr0L/JSsuJNGruZqmyjOJqH5UvGj0i5hmtZnk27xbsOJE2QJkCQuWqqt5ylJdEeMxGNOU/s1il+gZ2ioqzyRGNpqnJwWKKDJZqvGyZFwlRZOkXBSTL4iXlj9ZUjSmKVvltnmlZS+fCm0VlWeSiL0wLxyIqC42r4KvzfdNNaJIYTKtnNEk30oraKpsmrK3H4nu4ZkjsKrakGeSyaUVMWqG+GZNTy4rIlGryWyaIuZTCH1onLnRisozSVu8gaSi3D+xYnknwr2s6fvmBPHLvLpGVUrXNE2jmE/BiiZZnOyeLqto8qL9W0xIBSbRkLQxZoV59eNWglaZR+py0paKM47bDZGI/U0VHkT0zpm/l4rjpWBFk8xvdWNvGU01P3wpIrFyfhfPxagwhghxhHw1P7iJxDVxV/QJYxBp+NP8WsailRNR53g4Y74o/Oe4lR4hCAMcM9xuskovJ7zHzavcuP9SD3MjC8VQ+zfiiFTlGpZXjJjYJMv/zkorTetUjJ8lnor7Yl1oazYNFAetfLq2CpHWGOTIabViH861Zk6duuqqq+XoL1DCoXNol2YvAAAAAElFTkSuQmCC>