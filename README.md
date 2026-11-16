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
                        [API FastAPI] /predict /live /analyze_location /sites /history /alerts /briefing
                                                  │
                                                  ▼
                 [Dashboard web] carte + dessin de sites + alertes + briefing IA + historique
```

## Dataset & entraînement

[Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies)
(Kaggle, Francisco Escobar) — 2841 images Sentinel-2 + masques d'eau binaires
(NDWI). Entraînement sur **Kaggle (GPU gratuit)** —
[notebooks/kaggle_train_segmentation.ipynb](notebooks/kaggle_train_segmentation.ipynb),
tracé avec MLflow (`/kaggle/working/mlruns`). Le poids entraîné est rapatrié
dans `models/model.pt` et servi en **CPU** localement.

## Lancer le projet

```bash
pip install -r requirements.txt
uvicorn src.serve:app --host 127.0.0.1 --port 8000
# ouvrir http://127.0.0.1:8000
```

Prérequis pour l'analyse satellite (optionnel, dégrade proprement sinon) :
authentification Google Earth Engine une fois (`python -c "import ee; ee.Authenticate()"`)
et un projet Cloud enregistré (`gee.project` dans `params.yaml`).
Sans `models/model.pt`, l'upload d'image fonctionne quand même avec un
modèle non entraîné (placeholder), signalé explicitement dans l'interface.
Sans `ANTHROPIC_API_KEY`, le briefing IA bascule sur un résumé par règles.

## Fonctionnalités

- **Segmentation par deep learning** (U-Net + ResNet34, transfer learning + décodeur réellement entraîné)
- **Analyse spectrale réelle** (GEE) : NDVI (végétation/irrigation), NDCI (algues), NDTI (turbidité), NDWI
- **Pluviométrie réelle** (CHIRPS) et explication croisée : une baisse malgré la pluie est signalée comme anormale
- **Carte interactive** : sites prédéfinis + création de nouveaux sites par clic ou dessin de zone (polygone)
- **Alertes automatiques** sur chute anormale du niveau d'eau, bannière visible en haut du dashboard
- **Agent IA** (Claude) qui synthétise l'état de tous les sites en langage naturel, avec repli par règles si aucune clé API
- **Rate limiting** sur les endpoints déclenchant un appel Earth Engine (protège le quota gratuit)

## L'industrialisation MLOps

| Brique | Rôle | Fichier |
|---|---|---|
| Segmentation | modèle deep learning entraîné | `src/model.py` |
| GEE | récupération Sentinel-2/CHIRPS, indices spectraux | `src/gee_client.py` |
| Qualité d'eau (RGB) | heuristique couleur sur image uploadée | `src/water_quality.py` |
| Sites | registre seed + sites créés dynamiquement | `src/sites.py`, `src/site_store.py` |
| Historique + alertes | SQLite, règle de détection de chute | `src/alerts.py` |
| Agent IA | synthèse en langage naturel | `src/agent.py` |
| Rate limiting | protection quota GEE | `src/rate_limit.py` |
| API | backend FastAPI, logging structuré | `src/serve.py` |
| Dashboard | frontend web (carte, dessin, alertes, briefing) | `static/index.html` |
| Tests | 43 tests pytest (modèle, GEE, alertes, agent, API, rate limit) | `tests/` |
| Docker / CI | conteneurisation + tests auto à chaque push | `Dockerfile`, `.github/workflows/ci.yml` |

## Limites connues (honnêteté avant l'oral)

- Les seuils (NDCI > 0.15, NDTI > 0.05, pluviométrie, -20% d'alerte) sont des
  points de départ plausibles issus de la littérature, **pas calibrés** sur
  des mesures réelles de barrages tunisiens — à présenter comme preuve de
  concept, pas comme validé.
- Le module de preprocessing CV classique (`src/preprocess.py` : netteté,
  déduplication, GrabCut) est testé unitairement mais **pas encore branché**
  dans le pipeline d'entraînement ou de service réel.
- `requirements.txt` est figé à des versions exactes ; pas encore de lockfile
  complet des dépendances transitives.
- Rate limiting en mémoire (par process) — ne protège pas un déploiement
  multi-worker.

## Prochaines étapes

- Calibrer les seuils contre des relevés officiels réels de barrages (ONAGRI/DGRE).
- Historique long terme (12-24 mois) directement depuis l'archive Sentinel-2, pas seulement les lectures accumulées depuis le lancement.
- Bilan hydrique : croiser le niveau du barrage (offre) avec l'évolution des zones irriguées via NDVI (demande).
- Alertes à paliers (vigilance / alerte / critique) au lieu d'un seuil binaire.
- Prévision simple de tendance (extrapolation) pour anticiper un seuil critique.
- Notifications réelles (email/SMS) en plus de l'alerte dans le dashboard.
