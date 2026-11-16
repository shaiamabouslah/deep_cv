# Surveillance des plans d'eau par imagerie satellite — Industrialisation Vision par Ordinateur & MLOps

Application **full-stack** de monitoring : à partir d'une image satellite, un
modèle de **segmentation sémantique** (deep learning) détecte la surface
d'eau d'un barrage/lac, calcule son évolution dans le temps, et **déclenche
une alerte** si le niveau chute anormalement — un scénario directement
utile en Tunisie, pays en stress hydrique chronique (ex. barrage Sidi
Salem).

## Pipeline complet

```
Image satellite (Sentinel-2)
        │
        ▼
[Preprocessing CV classique]  nettoyage, netteté, standardisation (OpenCV)
        │
        ▼
[Segmentation U-Net]  encodeur ResNet34 pré-entraîné (ImageNet) + décodeur
        │                entraîné sur le dataset "Satellite Images of Water Bodies"
        ▼
Masque d'eau  →  % de surface couverte par l'eau
        │
        ▼
[Historique (SQLite)]  une lecture par site et par date
        │
        ▼
[Règle d'alerte]  chute > 20% vs. moyenne historique du site → alerte
        │
        ▼
[API FastAPI]  /predict  /history/{site}  /alerts
        │
        ▼
[Dashboard web]  upload d'image, masque affiché, courbe d'évolution, alertes
```

## Où est le deep learning, concrètement

- **Modèle** : U-Net (segmentation sémantique) avec un **encodeur ResNet34
  pré-entraîné sur ImageNet** (transfer learning), dont le **décodeur est
  réellement entraîné** sur les paires image/masque du dataset — donc,
  contrairement à une approche zero-shot, il y a un vrai entraînement ici.
- **Entraînement** : fait sur **Kaggle (GPU gratuit)**, notebook fourni dans
  [notebooks/kaggle_train_segmentation.ipynb](notebooks/kaggle_train_segmentation.ipynb).
  Le poids entraîné (`model.pt`) est ensuite rapatrié localement.
- **Service (inférence)** : tourne très bien en **CPU** localement — une
  seule image à traiter par requête.

## Dataset

[Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies)
(Kaggle, Francisco Escobar) — 2841 images Sentinel-2 + masques d'eau binaires
(calculés par NDWI). Utilisé comme preuve de concept ; le pipeline se
rebranche directement sur de vraies images Sentinel-2 de Tunisie une fois
l'accès Copernicus/Google Earth Engine mis en place (prochaine itération).

## Lancer le projet

```bash
pip install -r requirements.txt

# 1. Entraîner sur Kaggle (voir notebooks/kaggle_train_segmentation.ipynb),
#    puis déposer le fichier téléchargé dans models/model.pt

# 2. Lancer l'API + le dashboard
uvicorn src.serve:app --reload
# ouvrir http://127.0.0.1:8000
```

Sans `models/model.pt`, l'API démarre quand même avec un modèle **non
entraîné** (placeholder) pour que tout le reste (dashboard, historique,
alertes, tests) reste testable — l'interface l'indique clairement.

## L'industrialisation MLOps

| Brique | Rôle | Fichier |
|---|---|---|
| Preprocessing CV | nettoyage/qualité des images (classique, OpenCV) | `src/preprocess.py` |
| Segmentation | extraction de la surface d'eau (deep learning) | `src/model.py` |
| Historique + alertes | stockage SQLite, règle de détection de chute anormale | `src/alerts.py` |
| API | backend FastAPI (`/predict`, `/history`, `/alerts`) | `src/serve.py` |
| Dashboard | frontend web (upload, masque, courbe, alertes) | `static/index.html` |
| DVC | pipeline d'évaluation reproductible | `dvc.yaml`, `params.yaml` |
| MLflow | tracking des expériences (utilisé côté notebook Kaggle) | `notebooks/` |
| Tests | pytest sur modèle, preprocessing, alertes, API | `tests/` |
| Docker | conteneurisation du service complet | `Dockerfile` |
| CI/CD | tests + build auto à chaque push | `.github/workflows/ci.yml` |

## Pour l'oral : argumentaire d'industrialisation

1. **Transfer learning** (encodeur pré-entraîné) + **entraînement réel du
   décodeur** sur GPU Kaggle : rapide, reproductible, peu coûteux.
2. **Preprocessing CV classique** avant tout deep learning : montre une
   vraie maîtrise des deux approches (classique + apprentissage profond).
3. **Séparation calcul lourd (Kaggle, offline) / service léger (CPU,
   temps réel)** : pattern standard en MLOps.
4. **Pipeline complet mesurable** : DVC pour la reproductibilité, MLflow
   pour le suivi, tests pour la fiabilité, CI/CD pour la non-régression.
5. **Boucle fermée produit** : détection → métrique → alerte →
   visualisation, pas juste un modèle isolé dans un notebook.

## Prochaines étapes

- Brancher de vraies images Sentinel-2 de Tunisie via Copernicus/Google
  Earth Engine (accès déjà expérimenté sur un autre projet).
- Ajouter un job planifié (cron) qui récupère une nouvelle image par site
  chaque semaine et appelle `/predict` automatiquement.
- Notifications réelles (email/SMS) en plus de l'alerte affichée sur le
  dashboard.
