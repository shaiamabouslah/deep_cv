"""Promotion gate: may a candidate segmentation model replace the live one?

The decision rests on two held-out test sets the candidate was never
trained or tuned on, scored with the same preprocessing as the API:
  - the six Tunisian dams (data/eval): the place the app is deployed
  - the world benchmark (data/world): clean tiles from 100+ lakes and
    reservoirs worldwide, the check on generalisation

Rules (all must hold):
  1. Tunisia: candidate IoU >= live IoU - MAX_TUNISIA_LOSS  (no regression at home)
  2. World:   candidate IoU >= live IoU + MIN_WORLD_GAIN    (a real gain abroad)
Both models' bootstrap intervals on the world benchmark are reported so a
"gain" inside the noise is visible, but the gate uses the point estimate
only with a margin; this is a guard against regressions, not a significance
test (90 tiles cannot prove small differences).

    python -m src.promote <candidate.pt> [--apply]
--apply copies the candidate over models/model.pt (the live model is kept
as models/model_previous.pt) only if every rule holds.
"""
import json
import os
import shutil
import sys

from src import world_benchmark as wb
from src.evaluate import score_checkpoint

LIVE_PATH = "models/model.pt"
MAX_TUNISIA_LOSS = 0.02
MIN_WORLD_GAIN = 0.02
REPORT_PATH = "reports/promotion.json"


def evaluate_model(path, tunisia_dir="data/eval", world_dir="data/world"):
    tunisia, _ = score_checkpoint(path, tunisia_dir)
    _, rows = score_checkpoint(path, world_dir)
    clean = [r for r in rows if wb.tile_quality(r) is None]
    return {"tunisia_iou": tunisia["iou"], "world": wb.pooled(clean, 1000), "world_reservoirs":
            wb.pooled([r for r in clean if r["type"] == "reservoir"], 1000),
            "world_lakes": wb.pooled([r for r in clean if r["type"] == "lake"], 1000)}


def decide(live, candidate):
    rules = {
        "no regression in Tunisia": candidate["tunisia_iou"] >= live["tunisia_iou"] - MAX_TUNISIA_LOSS,
        "gain on the world benchmark": candidate["world"]["iou"] >= live["world"]["iou"] + MIN_WORLD_GAIN,
    }
    return all(rules.values()), rules


def run(candidate_path, apply=False, live_path=LIVE_PATH, report_path=REPORT_PATH, evaluator=evaluate_model, track=False):
    live, candidate = evaluator(live_path), evaluator(candidate_path)
    promoted, rules = decide(live, candidate)
    report = {"candidate": candidate_path, "live": live, "candidate_scores": candidate,
              "rules": rules, "promoted": promoted and apply, "would_promote": promoted}
    if promoted and apply:
        shutil.copy2(live_path, os.path.join(os.path.dirname(live_path), "model_previous.pt"))
        shutil.copy2(candidate_path, live_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    if track:
        log_decision(report, live_path, candidate_path, apply)
    return report


def log_decision(report, live_path, candidate_path, apply):
    """Records the gate decision as an MLflow run. With --apply, the candidate
    also becomes a registry version: champion (old champion kept as `previous`)
    if promoted, `rejected` if the gate refused it."""
    from src import registry, tracking

    tracking.log_experiment(
        "model-promotion", f"gate {os.path.basename(candidate_path)}",
        params={"candidate": candidate_path, "live": live_path, "applied": apply,
                "max_tunisia_loss": MAX_TUNISIA_LOSS, "min_world_gain": MIN_WORLD_GAIN},
        metrics={"live": report["live"], "candidate": report["candidate_scores"],
                 "rules": {k: bool(v) for k, v in report["rules"].items()}, "promoted": report["promoted"]},
        artifacts=[REPORT_PATH], tags={"source": "promote", "decision": "promote" if report["would_promote"] else "reject"})
    if not apply:
        return
    scores = report["candidate_scores"]
    previous = registry.alias_version(registry.SEGMENTATION, "champion")
    version = registry.register_segmentation(
        candidate_path, f"Candidate {os.path.basename(candidate_path)}: "
        f"{'promoted' if report['would_promote'] else 'rejected'} by the gate. "
        f"Tunisia IoU {scores['tunisia_iou']}, world IoU {scores['world']['iou']}.",
        {"tunisia_iou": scores["tunisia_iou"], "world_iou": scores["world"]["iou"],
         "world_reservoir_iou": scores["world_reservoirs"]["iou"], "world_lake_iou": scores["world_lakes"]["iou"]})
    if report["would_promote"]:
        registry.promote_segmentation(version, previous_champion=previous)
    else:
        registry.reject_segmentation(version)
    print(f"registry: {registry.SEGMENTATION} v{version} -> {'champion' if report['would_promote'] else 'rejected'}")


if __name__ == "__main__":
    run(sys.argv[1], apply="--apply" in sys.argv, track=True)
