# Surveillance des plans d'eau par imagerie satellite — Industrialisation Vision par Ordinateur & MLOps

Application **full-stack** de monitoring des barrages/plans d'eau tunisiens,
combinant un **modèle de segmentation entraîné** (deep learning), une
**analyse spectrale réelle via Google Earth Engine** (NDVI/NDCI/NDTI +
pluviométrie CHIRPS), un **agent IA** de synthèse, et une carte interactive
permettant de créer n'importe quel nouveau site surveillé.

## Les deux chemins d'analyse (ne pas confondre)

| | Upload d'image | Analyse satellite (temps réel) |
|---|---|---|
| Entrée | Une image satellite RGB fournie par l'utilisateur | Rien — juste les coordonnées du site |
| Modèle utilisé | **U-Net entraîné** (`models/model.pt`), le livrable deep learning du projet | **Google Earth Engine** : dernière image Sentinel-2 + CHIRPS |
| Sortie | Masque d'eau + % de surface + heuristique couleur RGB | Vrais indices scientifiques NDVI/NDCI/NDTI + pluviométrie + explication croisée niveau/météo |
| Endpoint | `POST /predict` | `GET /live/{site}` ou `POST /analyze_location` |

Le chemin "upload" existe pour démontrer le modèle de deep learning
lui-même ; le chemin "satellite" ne l'utilise pas du tout, il s'appuie sur
de vraies bandes spectrales Sentinel-2.

## Pipeline

```
                      ┌─ Upload image ──→ [U-Net entraîné] ──→ masque + % eau + couleur RGB
Site (carte/clic/dessin) │
                      └─ Analyse live ──→ [Google Earth Engine] ──→ NDVI/NDCI/NDTI + pluie (CHIRPS)
                                                  │
                                                  ▼
                                   [Historique SQLite] + [Règle d'alerte]
                                                  │
                                                  ▼
                     [Prévision] GRU +1/+2/+4 sem. (repli : tendance + pluie Open-Meteo)
                                                  │
                                                  ▼
                 [API FastAPI] /predict /live /analyze_location /sites /history /forecast /alerts /briefing
                                                  │
                                                  ▼
                 [Dashboard web] carte + dessin de sites + alertes + briefing IA + prévision + historique
```

## Dataset & entraînement

[Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies)
(Kaggle, Francisco Escobar) — 2841 images Sentinel-2 + masques d'eau binaires
(NDWI). Entraînement sur **Kaggle (GPU gratuit)** —
[notebooks/kaggle_train_segmentation.ipynb](notebooks/kaggle_train_segmentation.ipynb),
tracé avec MLflow (SQLite). Le poids entraîné est rapatrié dans
`models/model.pt` et servi en **CPU** localement.

Protocole d'entraînement (v2) : split train/val/test à graine fixe (70/15/15),
U-Net ResNet34 pré-entraîné ImageNet **entièrement fine-tuné** (encodeur non
gelé), perte Dice + BCE, 30 epochs avec cosine LR, augmentations (flips,
rotations 90°, bandes no-data noires étiquetées « terre »), sauvegarde du
meilleur epoch sur la validation, seuil de décision optimisé sur la validation
puis score final sur le test jamais vu, écrit dans `reports/train_metrics.json`
et dans le checkpoint lui-même (le seuil est relu par l'API).

## Évaluation sur les vrais barrages

Le score Kaggle mesure le modèle sur les images Kaggle ; ce qui compte ici est
sa performance sur **nos** barrages. `data/eval/` contient 18 tuiles
Sentinel-2 réelles (6 barrages × saison humide / printemps / sèche),
étiquetées par NDWI sur la même scène (même règle que les masques Kaggle ; le
modèle ne voit que le RGB, la référence est donc indépendante). Nuages,
ombres et zones hors champ sont marqués « ignorer » et ne comptent ni pour ni
contre le modèle. Reconstruction : `python -m src.build_eval_set` (GEE).

```bash
dvc repro            # lance src/evaluate.py (venv activé), écrit reports/metrics.json
dvc metrics show
```

Résultat du modèle servi (v3, seuil 0,3) :

| IoU (jeu entier) | Précision | Rappel | Erreur moyenne sur le % d'eau |
|---|---|---|---|
| 0.758 | 0.856 | 0.868 | 0.72 point |

Détail par tuile dans `reports/eval_per_image.json`. Prétraitement identique
à l'API (redimensionnement simple, seuil du checkpoint, pixels no-data exclus).

### Ablation du transfer learning

`notebooks/kaggle_transfer_ablation.ipynb` entraîne le même U-Net quatre fois
(encodeur aléatoire, ImageNet gelé, ImageNet fine-tuné, ImageNet fine-tuné avec
learning rate réduit sur l'encodeur) à recette identique, plus une courbe
d'efficacité des données (10 % / 25 % / 100 % des images). Ensuite
`python -m src.compare_variants` ajoute le score de chaque variante sur nos
barrages dans `reports/ablation.md`. **Pas encore exécutée sur Kaggle** : le
notebook est testé sur données factices, les résultats restent à produire.

## Prévision par apprentissage profond

Un petit GRU (~8 000 paramètres, 5 graines moyennées) lit les 12 dernières
semaines d'un barrage (surface d'eau mesurée comme dans l'application, fraîcheur
de l'observation, pluie CHIRPS, évaporation et température ERA5-Land, saison)
et prédit le niveau à +1, +2 et +4 semaines. Données : `data/forecast/weekly.csv`,
2 502 semaines-barrages depuis décembre 2018 (`python -m src.build_forecast_dataset`).
Entraînement et évaluation reproductibles : `dvc repro train_forecaster`.

Évaluation sur des semaines jamais vues (juillet 2025 à novembre 2026,
425 fenêtres), erreur absolue moyenne en points de pourcentage :

| Méthode | +1 sem. | +2 sem. | +4 sem. |
|---|---|---|---|
| Aucun changement | 2.12 | 2.76 | 3.97 |
| Tendance linéaire (ancienne méthode de l'application) | 2.30 | 3.06 | 4.70 |
| Moyenne saisonnière | 2.13 | 2.72 | 3.62 |
| Régression ridge | 2.70 | 3.14 | 4.04 |
| **GRU (historique météo)** | **2.00** | 2.55 | **3.44** |

Le GRU est meilleur que toutes les références à tous les horizons, avec des
intervalles de confiance (bootstrap par semaines) qui excluent zéro ; **l'ancienne
tendance linéaire est pire que « aucun changement »**. Limites : les gains sont
modestes (0.1 point à une semaine, 0.5 à quatre), le modèle n'améliore pas les
semaines de grosse variation, et la prévision de pluie des deux semaines à venir
n'apporte rien (le modèle servi ne l'utilise donc pas). Détail, intervalles et
résultats par barrage : [reports/forecast_eval.md](reports/forecast_eval.md).
L'application affiche cette prévision (`/forecast/{site}`, calculée via Earth Engine
puis mise en cache 6 h) avec l'erreur mesurée, et un seuil d'alerte qui ignore les
baisses plus petites que l'erreur typique du modèle.

## Du modèle au taux de remplissage officiel (calibration)

L'indice de surface (pixels d'eau / étendue maximale JRC) n'est pas le taux de
remplissage publié par l'ONAGRI : un barrage a un profil en cuvette. Les
bulletins hebdomadaires 2024-01 à 2025-10 ont été collectés
(`src/fetch_bulletins.py`, `src/official_fill.py`) et une courbe
surface → taux officiel est ajustée **par barrage** (régression isotone ancrée
en (0,0) et (100,100), `src/calibration.py`). Validation par blocs temporels
contigus, et la courbe n'est utilisée que si elle bat une estimation constante :
erreur typique de **1,6 à 3,6 points** de taux de remplissage (contre 10 à 35
points sans calibration). Nebhana n'est pas retenu (gain négatif sur son
validation croisée) : l'application n'affiche alors pas de taux calibré.
Détail : [reports/calibration.md](reports/calibration.md).

## Benchmark mondial et réentraînement contrôlé

Pour ne pas juger le modèle sur 6 barrages seulement, un benchmark de **90 plans
d'eau tirés au hasard par biome** (HydroLAKES, étiquettes NDWI, tuiles de qualité
filtrées, `src/build_world_set.py`, `src/world_benchmark.py`) est calculé avec
IoU poolé et intervalle de confiance par bootstrap par blocs.

| Modèle | IoU Tunisie (18 tuiles) | IoU monde (90 tuiles) | réservoirs | lacs naturels |
|---|---|---|---|---|
| v1 (Kaggle, 10 epochs) | 0,723 | 0,449 | 0,726 | 0,413 |
| v2 (Kaggle, augmenté) — **rejeté** | 0,563 | 0,466 | 0,777 | 0,424 |
| v3 (v1 affiné sur 306 tuiles mondiales) — **servi** | 0,758 | 0,566 | 0,773 | 0,539 |

La v3 est issue de `src/finetune_world.py` (augmentation de couleur et de
zoom, perte masquée) sur des sites **disjoints** du benchmark (contrôle de
fuite spatiale). Elle ne remplace le modèle servi que si la **porte de
promotion** (`src/promote.py`) l'accepte : pas de régression en Tunisie
(tolérance 0,02) et gain d'au moins 0,02 sur le benchmark mondial. Ce
mécanisme a rejeté la v2 et promu la v3 ; ses décisions sont dans le Model
Registry (voir MLOps).

## Radar Sentinel-1 : mesurer quand le ciel est couvert

`src/build_sar_dataset.py` + `src/sar_model.py` + `src/train_sar.py` : un
U-Net ResNet18 sur les bandes radar VV/VH, supervisé par les masques optiques
d'images claires du même jour (supervision inter-modalités), comparé à un
seuillage classique et à un GBM sur caractéristiques manuelles, évalués sur des
régions jamais vues.

| Méthode | IoU monde (test) | IoU Tunisie (test) |
|---|---|---|
| seuil VV | 0,591 | 0,671 |
| seuil VV + médian | 0,611 | 0,732 |
| GBM (caractéristiques radar) | 0,647 | 0,804 |
| U-Net, init. aléatoire | 0,518 | 0,804 |
| **U-Net, init. ImageNet (transfer learning)** | **0,746** | **0,872** |

Le transfer learning compte surtout hors de la zone d'entraînement : +0,23 d'IoU
monde par rapport à l'initialisation aléatoire. Utilisé seul pour estimer le
taux de remplissage (`src/sar_gapfill.py`, 416 dates dont 95 sans image optique
exploitable), le radar atteint **2 à 7,7 points** d'erreur selon le barrage après
recalage par barrage (contre 1,6 à 3,6 pour l'optique) : utile pour combler les
semaines nuageuses, moins précis que l'optique quand elle est disponible.

## Lancer le projet

```bash
pip install -r requirements.txt
uvicorn src.serve:app --host 127.0.0.1 --port 8000
# ouvrir http://127.0.0.1:8000
```

Pile MLOps (chaque commande dans son terminal) :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db          # expériences et Model Registry : http://127.0.0.1:5000
prefect server start                                        # orchestrateur (venv .venv-orchestrator) : http://127.0.0.1:4200
python orchestration/flows.py continuous --force            # un cycle d'entraînement continu
python -m src.monitoring simulate-drift                     # démo de dérive, puis http://127.0.0.1:8000/monitoring.html
docker compose up                                           # API + Prometheus (9090) + Grafana (3000)
```

Prérequis pour l'analyse satellite (optionnel, dégrade proprement sinon) :
authentification Google Earth Engine une fois (`python -c "import ee; ee.Authenticate()"`)
et un projet Cloud enregistré (`gee.project` dans `params.yaml`).
Sans `models/model.pt`, l'upload d'image fonctionne quand même avec un
modèle non entraîné (placeholder), signalé explicitement dans l'interface.
Sans `GROQ_API_KEY`, le briefing IA bascule sur un résumé par règles.

## Fonctionnalités

- **Segmentation par deep learning** (U-Net + ResNet34 pré-entraîné, entièrement fine-tuné) — analyse ponctuelle d'une image uploadée, non ajoutée à l'historique du barrage (unité différente : % de la photo, pas % de l'étendue max)
- **Analyse spectrale réelle** (GEE) : NDVI (végétation/irrigation), NDCI (algues), NDTI (turbidité), NDWI
- **Gestion des nuages** : choix de l'image où le réservoir lui-même est le plus visible (bande SCL, pixel par pixel), nuages/ombres exclus du calcul ; si moins de 70 % du réservoir est visible sur toutes les images, aucune mesure n'est produite plutôt qu'une fausse baisse
- **Pluviométrie réelle** (CHIRPS) et explication croisée : une baisse malgré la pluie est signalée comme anormale
- **Carte interactive** : sites prédéfinis + création de nouveaux sites par clic ou dessin de zone (polygone)
- **Alertes automatiques** sur chute anormale du niveau d'eau, bannière visible en haut du dashboard
- **Interface en quatre onglets** : Suivi (liste des barrages avec taux de remplissage, carte satellite, mesure, courbe, prévision), Tester une image (l'eau détectée superposée à l'image, avec curseur), Précision dans le monde (carte du benchmark), Comment ça marche. Le briefing de l'agent a été retiré de l'interface ; le code de l'agent (`src/agent.py`, endpoint `/briefing`) reste disponible mais n'est plus affiché
- **Prévision par apprentissage profond** (GRU, +1/+2/+4 semaines) affichée après chaque analyse live avec son erreur mesurée sur des semaines jamais vues ; repli sur l'extrapolation linéaire + prévision de pluie Open-Meteo si le modèle ou les images récentes ne sont pas disponibles
- **Rate limiting** sur les endpoints déclenchant un appel Earth Engine (protège le quota gratuit)

## L'industrialisation MLOps

Chaque point ci-dessous se montre en interface **et** dans le code ; le détail
(commandes, fichiers, ce qui est vérifié ou non) est dans
[docs/MLOPS.md](docs/MLOPS.md).

| # | Exigence | Mise en oeuvre | Où le voir |
|---|---|---|---|
| 1 | Versionnement code et données | Git pour le code ; **DVC** pour les modèles et jeux de données (remote local), 5 étapes reproductibles, `dvc.lock` | `dvc.yaml`, `*.dvc`, `.dvc/config` |
| 2 | Expériences MLflow (params, métriques) | tous les entraînements et évaluations journalisent paramètres, métriques, courbes par epoch, commit git | `src/tracking.py`, `mlflow ui` (port 5000) |
| 3 | Model Registry | 3 modèles enregistrés, versions, alias `champion` / `previous` / `rejected`, historique des promotions | `src/registry.py`, onglet Models |
| 4 | Orchestration + gestion d'erreurs | **Prefect** : flux d'entraînement et d'entraînement continu, retries avec délais croissants, timeouts, hooks d'échec, UI | `orchestration/flows.py`, `prefect server start` |
| 5 | CI : tests exécutés | GitHub Actions : lint (ruff), 190 tests avec couverture (69 %), validation DVC, build et test du conteneur | `.github/workflows/ci.yml` |
| 6 | CD et CT | CD : image publiée sur ghcr.io après tests ; CT : flux Prefect déclenché par la dérive ou la perte de performance, planifié chaque semaine | `.github/workflows/cd.yml`, `ct.yml`, `orchestration/deploy.py` |
| 7 | Supervision système et modèle | Prometheus (`/metrics`) : trafic, latence, erreurs, CPU, mémoire ; dérive des données (PSI + Kolmogorov-Smirnov) ; qualité des prévisions dans le temps ; page `/monitoring.html`, Grafana | `src/monitoring.py`, `src/serve_metrics.py`, `monitoring/` |

Autres briques du projet :

| Brique | Rôle | Fichier |
|---|---|---|
| Segmentation | U-Net ResNet34, modèle servi `models/model.pt` (v3) | `src/model.py`, `src/finetune_world.py` |
| Radar | U-Net Sentinel-1, estimation du remplissage sans optique | `src/sar_model.py`, `src/sar_gapfill.py` |
| GEE | Sentinel-2/CHIRPS/ERA5, indices spectraux | `src/gee_client.py` |
| Calibration | indice de surface vers taux officiel ONAGRI | `src/calibration.py` |
| Qualité d'eau (RGB) | heuristique couleur sur image uploadée | `src/water_quality.py` |
| Historique + alertes | SQLite, règle de détection de chute | `src/alerts.py` |
| Agent IA | synthèse en langage naturel (Groq) | `src/agent.py` |
| Prévision deep learning | GRU, jeu de données, entraînement, service | `src/forecast_*.py`, `src/train_forecaster.py` |
| API et dashboard | FastAPI, Leaflet, vue monde | `src/serve.py`, `static/index.html` |
| Tests | 190 tests pytest | `tests/` |

## Limites connues (honnêteté avant l'oral)

- **Validé et corrigé contre des données réelles les 2026-11-21 et
  2026-11-24** (voir [reports/validation_vs_official.md](reports/validation_vs_official.md)),
  en trois temps :
  1. Le `water_area_pct` initial (rescaling linéaire du NDWI moyen) donnait
     une tendance **inversée** par rapport à l'officiel ONAGRI/DGRE sur
     Sidi Salem — remplacé par une surface d'eau réelle (pixels classifiés)
     normalisée par l'étendue maximale historique **JRC Global Surface
     Water**.
  2. Les 6 coordonnées de barrage (`src/sites.py`) se sont révélées mal
     centrées (0 à 8.1 km d'écart de l'eau réelle). Corrigées pour 4 des 6
     sites via une recherche de plus grande composante connexe d'eau
     (JRC) dans une fenêtre de 15 km.
  3. Les 2 sites restants (Sidi El Barrak, Mellègue) avaient chacun une
     cause distincte : Sidi El Barrak fusionnait avec la Méditerranée
     (fixé par un masque terre/mer, pays `USDOS/LSIB_SIMPLE/2017`) ;
     Mellègue avait son vrai réservoir hors de la fenêtre de 15 km (fixé
     en l'élargissant à 35 km). Les 6 sites ont désormais des surfaces en
     hectares cohérentes avec la taille réelle connue des barrages, mais
     Sidi El Barrak et Mellègue sont une vérification de **plausibilité**,
     pas une validation officielle (pas de point ONAGRI/DGRE collecté pour
     ces deux sites).
  La tendance est maintenant correcte et la géométrie d'échantillonnage
  n'est plus le facteur limitant ; l'écart absolu restant (sur Sidi
  Salem/Nebhana, les 2 sites validés officiellement) est attribué à la
  nature même de la grandeur (indice de **surface** vs taux officiel de
  **volume/capacité** — un barrage a un profil en cuvette, les deux ne
  sont pas linéairement liées sans la courbe hauteur-surface-volume propre
  à chaque site, non publique). Diagnostic complet dans le rapport de
  validation.
- **Modèle de segmentation** : le modèle servi est la v3 (v1 affiné sur des
  tuiles mondiales, IoU 0,758 sur les barrages tunisiens et 0,566 sur le
  benchmark mondial, intervalle de confiance 95 % [0,49 ; 0,64]). Il ne voit que
  le RGB (pas le proche infrarouge) : il tend à classer en eau les zones très
  sombres (ombres de nuages, forêt dense), et reste moins bon sur les lacs
  naturels (0,54) que sur les réservoirs (0,77). Toutes les évaluations
  (Tunisie comme monde) prennent le **NDWI comme référence** : c'est une vérité
  terrain approchée, pas une annotation manuelle, donc un IoU de 1,0 ne serait
  pas atteignable ni souhaitable. La v2 (Kaggle) a été rejetée par la porte de
  promotion : bonne sur le jeu Kaggle (IoU 0,81), mais 0,563 sur les barrages à
  cause d'une scène claire de saison sèche (Nebhana, sol nu classé en eau).
- **Prévision** : entraînée et évaluée sur les 6 grands barrages seulement. Sur un
  site dessiné par l'utilisateur elle répond dès qu'il y a de l'eau historique
  dans la zone, sans garantie que ce soit un réservoir ; la cible est l'indice
  de surface (pas le taux de remplissage officiel) et le bruit de mesure
  limite la précision atteignable.
- Les seuils NDCI/NDTI/pluviométrie restent des points de départ issus de la
  littérature, non calibrés non plus sur des mesures tunisiennes réelles.
- `requirements.txt` est figé à des versions exactes ; pas encore de lockfile
  complet des dépendances transitives.
- Rate limiting en mémoire (par process) — ne protège pas un déploiement
  multi-worker.

## Limites MLOps (à dire clairement)

- Le remote DVC est un dossier local (`C:/Users/HP/dvc-remote/water-monitoring`) :
  le versionnement est réel et vérifié (`dvc status -c` synchronisé), mais un
  collègue ou le CI ne peut pas faire `dvc pull` tant que le remote n'est pas
  un stockage partagé (DagsHub, S3...). `dvc remote modify storage url ...` suffit.
- Le conteneur et la pile Prometheus/Grafana (`docker-compose.yml`) sont écrits
  et leur syntaxe est validée, mais n'ont pas été exécutés sur cette machine
  (le démon Docker n'était pas démarré). Les métriques Prometheus ont, elles,
  été vérifiées directement sur l'API.
- Les workflows GitHub Actions n'ont pas encore tourné sur GitHub : leurs étapes
  ont été exécutées localement (lint, tests avec couverture, `dvc dag`,
  validation YAML).
- L'entraînement continu tourne dans Prefect, sur la machine qui détient les
  données ; le workflow `ct.yml` vérifie seulement que sa mécanique fonctionne.
- La supervision des prévisions ne devient informative qu'après plusieurs
  semaines : une prévision doit attendre la semaine réelle pour être comparée.
  La dérive des images est démontrée par une simulation (`python -m
  src.monitoring simulate-drift`), pas par un incident réel.

## Prochaines étapes

- Lancer `notebooks/kaggle_transfer_ablation.ipynb` sur Kaggle puis
  `python -m src.compare_variants` pour mesurer l'apport du transfer learning
  sur l'optique (le radar l'a déjà mesuré : +0,23 d'IoU monde).
- Remplacer le remote DVC local par un stockage partagé et activer le secret
  `DVC_REMOTE_URL` pour que le CD embarque les poids dans l'image.
- Collecter un point ONAGRI/DGRE officiel pour Nebhana (calibration non retenue).
- Rapport PDF exportable par site ; notifications réelles (email/SMS).
- Sécurisation production (authentification des endpoints, au-delà du rate limiting).
