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
                              [Prévision] tendance + pluie 7j (Open-Meteo)
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
Sans `GROQ_API_KEY`, le briefing IA bascule sur un résumé par règles.

## Fonctionnalités

- **Segmentation par deep learning** (U-Net + ResNet34, transfer learning + décodeur réellement entraîné)
- **Analyse spectrale réelle** (GEE) : NDVI (végétation/irrigation), NDCI (algues), NDTI (turbidité), NDWI
- **Pluviométrie réelle** (CHIRPS) et explication croisée : une baisse malgré la pluie est signalée comme anormale
- **Carte interactive** : sites prédéfinis + création de nouveaux sites par clic ou dessin de zone (polygone)
- **Alertes automatiques** sur chute anormale du niveau d'eau, bannière visible en haut du dashboard
- **Agent IA** (Groq) qui synthétise l'état de tous les sites et fusionne tous les signaux d'une analyse (segmentation, couleur, indices GEE, pluie, bilan hydrique) en une recommandation unique, avec repli par règles si aucune clé API
- **Prévision à 7 jours** : extrapolation linéaire de la tendance historique du site, croisée avec une prévision de pluie réelle (Open-Meteo, sans clé API) — affichée après chaque analyse live, avec le niveau de risque projeté et un avertissement qualitatif quand la pluie pourrait changer la tendance (jamais intégrée numériquement, faute de courbe pluie→surface calibrée par site)
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
| Prévision | tendance + pluie 7j (Open-Meteo) | `src/forecast.py` |
| Rate limiting | protection quota GEE | `src/rate_limit.py` |
| API | backend FastAPI, logging structuré | `src/serve.py` |
| Dashboard | frontend web (carte, dessin, alertes, briefing, prévision) | `static/index.html` |
| Tests | 69 tests pytest (modèle, GEE, alertes, agent, prévision, API, rate limit) | `tests/` |
| Docker / CI | conteneurisation + tests auto à chaque push | `Dockerfile`, `.github/workflows/ci.yml` |

## Limites connues (honnêteté avant l'oral)

- **Validé et corrigé contre des données réelles le 2026-11-21** (voir
  [reports/validation_vs_official.md](reports/validation_vs_official.md)),
  en deux temps :
  1. Le `water_area_pct` initial (rescaling linéaire du NDWI moyen) donnait
     une tendance **inversée** par rapport à l'officiel ONAGRI/DGRE sur
     Sidi Salem — remplacé par une surface d'eau réelle (pixels classifiés)
     normalisée par l'étendue maximale historique **JRC Global Surface
     Water**.
  2. Les 6 coordonnées de barrage (`src/sites.py`) se sont révélées mal
     centrées (0 à 8.1 km d'écart de l'eau réelle). Corrigées pour 4 des 6
     sites via une recherche de plus grande composante connexe d'eau
     (JRC) dans une fenêtre de 15 km — confirmé par des surfaces en
     hectares désormais cohérentes avec la taille réelle connue des
     barrages. Sidi El Barrak et Mellègue gardent leurs coordonnées
     approximatives, la même recherche y donnant un résultat implausible
     (fusion avec la mer / réservoir hors fenêtre).
  La tendance est maintenant correcte et la géométrie d'échantillonnage
  n'est plus le facteur limitant ; l'écart absolu restant est attribué à
  la nature même de la grandeur (indice de **surface** vs taux officiel de
  **volume/capacité** — un barrage a un profil en cuvette, les deux ne
  sont pas linéairement liées sans la courbe hauteur-surface-volume propre
  à chaque site, non publique). Diagnostic complet dans le rapport de
  validation.
- Les seuils NDCI/NDTI/pluviométrie restent des points de départ issus de la
  littérature, non calibrés non plus sur des mesures tunisiennes réelles.
- `requirements.txt` est figé à des versions exactes ; pas encore de lockfile
  complet des dépendances transitives.
- Rate limiting en mémoire (par process) — ne protège pas un déploiement
  multi-worker.

## Prochaines étapes

- **Finir la correction géométrique** pour Sidi El Barrak (masque terre/mer
  pour éviter la fusion avec la Méditerranée) et Mellègue (élargir la
  fenêtre de recherche au-delà de 15 km) — les 4 autres sites sont déjà
  corrigés (voir le rapport de validation).
- **Calibrer surface→volume** par site, si une courbe hauteur-surface-volume
  officielle devient accessible (DGRE), pour que `water_area_pct` se
  rapproche du taux de remplissage officiel plutôt que de rester un indice
  de surface.
- Rapport PDF exportable par site ; vue nationale agrégée (tous les sites
  en un coup d'œil).
- Notifications réelles (email/SMS) en plus de l'alerte dans le dashboard.
- Sécurisation production (authentification sur les endpoints, au-delà du
  rate limiting déjà en place).
