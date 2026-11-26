# MLOps : les 7 points, de la preuve à la démonstration

Pour chaque point : ce qui existe, où est le code, ce qu'on montre dans une
interface, et ce qui a réellement été exécuté ou pas. Les chemins sont
relatifs à la racine du dépôt.

## Préparer la démonstration (une fois)

```bash
# terminal 1 : l'application (la base de supervision de démonstration contient du trafic simulé)
MONITORING_DB=data/monitoring_demo.db uvicorn src.serve:app --port 8000
# terminal 2 : MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# terminal 3 : Prefect (environnement séparé)
.venv-orchestrator\Scripts\prefect server start
```

Interfaces : application http://127.0.0.1:8000, supervision
http://127.0.0.1:8000/monitoring.html, MLflow http://127.0.0.1:5000, Prefect
http://127.0.0.1:4200.

---

## 1. Versionnement du code et des données

| | |
|---|---|
| Code | Git. Chaque run MLflow enregistre le commit, la branche et si l'arbre de travail était modifié (`git_dirty`). |
| Données et modèles | **DVC** : `data/eval`, `data/world`, `data/world_train`, `data/sar`, `data/sar_gap`, `data/forecast/weekly.csv` et les 4 checkpoints de segmentation sont suivis par des fichiers `.dvc` (petits, dans git) ; leur contenu est dans le cache DVC et dans le remote. |
| Pipeline | `dvc.yaml` : 5 étapes (`evaluate`, `train_forecaster`, `calibration`, `world_benchmark`, `train_sar`) ; `dvc.lock` fige les empreintes des dépendances et des sorties. |
| Lien code, données, résultat | `dvc status` détecte qu'un fichier source a changé ; `dvc repro` relance seulement ce qui est périmé ; `dvc metrics show` compare les métriques entre commits. |

Démonstration :

```bash
dvc dag                  # graphe : jeux de données et modèles vers étapes
dvc status               # "Data and pipelines are up to date."
dvc status -c            # cache et remote synchronisés
dvc metrics show
```

Vérifié : 1910 fichiers poussés, cache et remote synchronisés, pipeline à jour.
Le flux d'entraînement Prefect se termine par `dvc commit` et `dvc push`.
Limite : le remote est un dossier local (`C:/Users/HP/dvc-remote/water-monitoring`),
donc ni le CI ni un collègue ne peuvent faire `dvc pull` ; remplacer par un
stockage partagé avec `dvc remote modify storage url <url>`.
Convention de versions proposée : un tag git par modèle promu (`model-v3`),
cohérent avec la version du Model Registry.

## 2. Expériences MLflow : paramètres et métriques

Code : `src/tracking.py` (`log_experiment`), appelé par chaque script
(`train_forecaster`, `train_sar`, `finetune_world`, `calibration`,
`world_benchmark`, `evaluate`, `promote`). Un échec de MLflow ne casse jamais
l'entraînement (testé). Les expériences qui précèdent l'instrumentation ont été
reportées par `src/backfill_mlflow.py` ; ces runs portent le tag
`source=backfill` et leurs valeurs sont relues des fichiers de résultats ou
recalculées depuis les checkpoints, rien n'est inventé.

Expériences : `segmentation-water`, `segmentation-sar-radar`,
`forecasting-water-level`, `fill-rate-calibration`, `world-benchmark`,
`model-promotion`, `model-registry`.

À montrer dans MLflow : onglet Experiments, `segmentation-sar-radar`, cocher
les runs `unet_imagenet_init` et `unet_random_init`, bouton Compare : mêmes
paramètres sauf l'initialisation, courbes `val_iou` par epoch, métriques
`test_world.iou` 0,746 contre 0,518. Dans `forecasting-water-level`, trier par
`h4.mae` : le GRU bat l'absence de changement (3,46 contre 3,97 points).

## 3. Model Registry

Code : `src/registry.py`. Trois modèles enregistrés :

| Modèle | Versions et alias |
|---|---|
| `water-segmentation` | v1 `previous` (ancien modèle servi), v2 `rejected` (refusé par la porte de promotion), v3 `champion` (servi) |
| `sar-water-segmentation` | v1 `champion` |
| `water-forecaster` | v1 `champion` ; une version v2, ré-entraînée par le flux d'entraînement continu, n'a pas été promue car elle n'est pas meilleure |

Gestion : les alias (et non les « stages » dépréciés de MLflow) disent quel
rôle joue chaque version ; chaque version porte en tags les métriques sur
lesquelles elle a été jugée (`tunisia_iou`, `world_iou`...) et une description
qui raconte pourquoi elle a été promue ou rejetée. Évolution : `src/promote.py
--apply` enregistre le candidat, le promeut (l'ancien champion devient
`previous`, retour arrière immédiat) ou le marque `rejected`.
`registry.promote_if_better` fait de même pour le radar et le prévisionniste.
Le chargement par alias est testé : `models:/water-segmentation@champion`.

À montrer : onglet Models, `water-segmentation`, les 3 versions avec leurs
alias et tags ; page d'une version, onglet de lignée vers le run qui l'a produite.

## 4. Orchestration, interface et gestion des erreurs

Code : `orchestration/flows.py` (Prefect, dans `.venv-orchestrator` car Prefect
impose des versions de dépendances incompatibles avec l'API).

Flux :

- `training-pipeline` : validation des données, entraînement du prévisionniste,
  calibration, évaluation, benchmark mondial, versionnement DVC.
- `continuous-training` : lit la supervision, décide, puis appelle le précédent ;
  option `retrain_segmentation` pour affiner et soumettre un candidat à la porte.
- `resilience-demo` : une étape qui échoue deux fois puis réussit, une qui échoue
  définitivement, pour montrer la gestion d'erreurs en direct.

Gestion d'erreurs (expliquée dans le code, en tête de `flows.py`) :

| Mécanisme | Où |
|---|---|
| Une commande qui sort en erreur lève `CommandFailed` avec la fin de son stderr : l'échec est visible, jamais avalé | `run_command` |
| Retries avec délais croissants (10 s, 60 s, 300 s) pour les tâches réseau | `refresh_data`, `RETRY_DELAYS` |
| Retries limités pour l'entraînement, aucun pour les étapes déterministes | paramètres `retries` de chaque `@task` |
| Délai maximal par tâche, un job bloqué ne bloque pas le flux | `timeout_seconds` |
| Hooks `on_failure` et `on_completion` : trace durable dans `reports/orchestration.log` | `notify_failure` |
| La validation des données arrête le flux avant tout entraînement | `validate_data` |
| Un candidat refusé par la porte est un résultat normal : le flux se termine `Completed`, le modèle servi n'est pas touché | `continuous_training` |
| Planification : lundi 3 h (contrôle), 1er du mois (réentraînement complet avec rafraîchissement) | `orchestration/deploy.py` |

Démonstration :

```bash
set PREFECT_API_URL=http://127.0.0.1:4200/api
.venv-orchestrator\Scripts\python orchestration\flows.py resilience     # échec visible, retries
.venv-orchestrator\Scripts\python orchestration\flows.py continuous     # décision pilotée par la supervision
.venv-orchestrator\Scripts\python orchestration\deploy.py               # planifications dans l'UI
```

Dans l'UI Prefect : Runs (états, durées, nombre de tâches), ouvrir
`resilience-demo` : la tâche « flaky step » montre ses 3 tentatives, la tâche
« always failing step » l'échec définitif. Vérifié : trois runs réels dans
l'UI (entraînement continu et pipeline `Completed`, démonstration de résilience
`Failed` volontairement) ; 8 tests (`tests/test_orchestration.py`) couvrent
retries, hooks, décision de réentraînement, rejet par la porte, arrêt sur
données invalides.

## 5. CI : tests et exécution dans le pipeline

`.github/workflows/ci.yml`, quatre jobs : `lint` (ruff), `test` (pytest avec
couverture, seuil 65 %, rapports JUnit et couverture en artefacts et dans le
résumé du job), `pipeline` (`dvc dag` valide `dvc.yaml`), `docker` (construit
l'image, la démarre et vérifie `/health`, `/metrics`, `/monitoring`). Cache pip,
annulation des runs obsolètes.

Vérifié localement : ruff sans erreur, **206 tests passent** (8 de plus
s'exécutent dans l'environnement Prefect), couverture **70 %**. Les tests
couvrent le modèle, l'évaluation, l'API (dont `/metrics` et la journalisation
des prédictions), la calibration, le benchmark, le suivi MLflow, le registre, la
dérive, les prévisions. Limite : les workflows n'ont pas encore tourné sur GitHub.

## 6. CD et CT

**CD** (`.github/workflows/cd.yml`) : après un CI réussi sur `main`, ou sur un
tag `v*`, l'image est construite, démarrée et testée (`/health`, `/metrics`)
avant d'être poussée sur ghcr.io (tags `main`, SHA court, version, `latest`).
Si le secret `DVC_REMOTE_URL` existe, `dvc pull` embarque les vrais poids ;
sinon l'image sert le modèle de repli et `/health` le signale. Non exécuté sur
GitHub ; le `Dockerfile` et `docker-compose.yml` sont validés syntaxiquement
mais le démon Docker n'était pas démarré ici.

**CT** : l'entraînement continu est le flux Prefect `continuous-training`.
Déclencheurs : dérive des données d'entrée (alerte PSI et Kolmogorov-Smirnov),
prévisionniste qui ne bat plus « aucun changement », exécution planifiée,
déclenchement manuel (`--force`). Il réentraîne puis, pour la segmentation, ne
remplace le modèle servi que si la porte de promotion accepte le candidat.
Vérifié : exécution réelle déclenchée par une dérive simulée (6 minutes,
Completed). `.github/workflows/ct.yml` exécute chaque lundi les tests de cette
mécanique ; il ne peut pas réentraîner (le runner n'a pas les données).

## 7. Supervision : système et modèle

Code : `src/serve_metrics.py` (Prometheus), `src/monitoring.py` (dérive et
performance), `static/monitoring.html`, `monitoring/` (Prometheus, règles
d'alerte, Grafana), `docker-compose.yml`.

Niveau système (`GET /metrics`) : requêtes par route et statut, latence
(histogramme), requêtes en cours, durée d'inférence du réseau, CPU et mémoire du
service et de la machine, disque. Les routes sont étiquetées par modèle
(`/history/{site}`), jamais par URL réelle, donc un utilisateur ne peut pas
faire exploser le nombre de séries (testé).

Niveau modèle :

| Quoi | Comment |
|---|---|
| Qualité des prédictions | confiance moyenne des pixels et part d'eau prédite, en histogrammes |
| Dérive des données | luminosité, contraste, netteté, couleurs, confiance et part d'eau des dernières prédictions comparées à un échantillon de 300 tuiles d'entraînement : PSI (seuils 0,1 et 0,25) **et** test de Kolmogorov-Smirnov, une variable n'est signalée que si les deux concordent |
| Dérive de concept | chaque prévision est enregistrée ; quand la semaine réelle est connue, l'erreur du GRU est comparée à celle de « aucun changement » (gain par horizon) ; `degraded` si le modèle fait pire sur la fenêtre récente |
| Décision | `retrain_needed` et ses raisons, lus par le flux Prefect, exposés en métrique et dans la page |

À montrer : `/monitoring.html` (carte de décision, ressources, une barre PSI par
variable avec moyenne de référence et actuelle, prévisions, dernières
prédictions) et `/metrics`. Démonstration de dérive :

```bash
python -m src.monitoring simulate-drift     # 80 images normales puis 80 images assombries
```

Mesuré dans la démonstration : luminosité 72,6 vers 55,1 (PSI 0,37), contraste
PSI 0,71, netteté PSI 0,66, alerte et réentraînement recommandé ; la confiance
et la part d'eau restent stables. Un test vérifie l'absence de fausse alerte
sur un trafic normal. Limites : la dérive est ici simulée, pas un incident réel ;
la supervision des prévisions ne devient informative qu'après plusieurs semaines
d'observations ; Prometheus et Grafana (`docker compose up`) sont configurés
mais n'ont pas été lancés ici.

---

## Ordre de présentation conseillé (10 minutes)

1. `dvc dag` puis `dvc status` (point 1), 1 min.
2. MLflow : comparer deux runs radar, puis Models (points 2 et 3), 3 min.
3. Prefect : lancer `resilience`, montrer retries et échec, puis le run
   d'entraînement continu (point 4 et CT), 3 min.
4. `simulate-drift` puis `/monitoring.html` et `/metrics` (point 7), 2 min.
5. Fichiers des workflows et résultat local des tests (points 5 et 6), 1 min.
