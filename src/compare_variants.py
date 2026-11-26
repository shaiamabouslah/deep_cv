"""Compares the transfer-learning variants trained by
notebooks/kaggle_transfer_ablation.ipynb.

Kaggle gives two numbers per variant (test IoU on the Kaggle dataset, best
epoch) and the data-efficiency curve; this script adds the one that matters
here: each variant's score on the real tiles of our own dams (data/eval/),
with exactly the serving preprocessing. Both go in one table.

Put the notebook outputs in place first:
    variants/model_<name>.pt     ->  models/variants/
    ablation_metrics.json        ->  reports/ablation_kaggle.json
then:
    python -m src.compare_variants
"""
import glob
import json
import os

from src.evaluate import score_checkpoint

VARIANT_DIR = "models/variants"
KAGGLE_PATH = "reports/ablation_kaggle.json"
OUT_JSON = "reports/ablation.json"
OUT_MD = "reports/ablation.md"
LABELS = {
    "scratch": "From scratch (random init)",
    "frozen": "ImageNet, encoder frozen",
    "finetune": "ImageNet, fine-tuned",
    "finetune_discriminative": "ImageNet, fine-tuned (encoder LR x0.1)",
}


def _fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def render_markdown(rows, fractions):
    lines = [
        "## Transfer-learning ablation",
        "",
        "Same data split, augmentation, loss and epochs for every variant; only the initialization / "
        "which layers train changes. Kaggle test = held-out part of the Kaggle dataset; dams = the 18 real "
        "tiles of our six dams (`data/eval/`).",
        "",
        "| Variant | Kaggle test IoU | Best epoch | Dams IoU | Dams precision | Dams recall | Dams water-% error (pts) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {LABELS.get(r['variant'], r['variant'])} | {_fmt(r['kaggle_test_iou'])} | "
            f"{r['best_epoch'] if r['best_epoch'] is not None else '-'} | {_fmt(r['dam_iou'])} | "
            f"{_fmt(r['dam_precision'])} | {_fmt(r['dam_recall'])} | {_fmt(r['dam_water_pct_mae'], 2)} |"
        )
    if fractions:
        sizes = sorted({size for curve in fractions.values() for size in curve}, key=float)
        lines += ["", "### Data efficiency (Kaggle test IoU when training on a fraction of the data)", "",
                  "| Variant | " + " | ".join(f"{float(s):.0%}" for s in sizes) + " |",
                  "|---|" + "---|" * len(sizes)]
        for name, curve in fractions.items():
            lines.append(f"| {LABELS.get(name, name)} | " + " | ".join(_fmt(curve.get(s)) for s in sizes) + " |")
    return "\n".join(lines) + "\n"


def compare(variant_dir=VARIANT_DIR, kaggle_path=KAGGLE_PATH, out_json=OUT_JSON, out_md=OUT_MD):
    kaggle = {}
    if os.path.exists(kaggle_path):
        with open(kaggle_path, encoding="utf-8") as f:
            kaggle = json.load(f)

    rows = []
    for path in sorted(glob.glob(os.path.join(variant_dir, "model_*.pt"))):
        name = os.path.basename(path)[len("model_"):-len(".pt")]
        metrics, _ = score_checkpoint(path)
        run = kaggle.get(f"{name}@1.0", {})
        rows.append({
            "variant": name,
            "kaggle_test_iou": run.get("test_iou"), "kaggle_val_iou": run.get("val_iou"),
            "best_epoch": run.get("best_epoch"), "threshold": metrics["threshold"],
            "dam_iou": metrics["iou"], "dam_precision": metrics["precision"],
            "dam_recall": metrics["recall"], "dam_water_pct_mae": metrics["water_pct_mae"],
        })
        print(rows[-1], flush=True)

    fractions = {}
    for key, run in kaggle.items():
        name, fraction = key.rsplit("@", 1)
        fractions.setdefault(name, {})[fraction] = run.get("test_iou")
    fractions = {name: curve for name, curve in fractions.items() if len(curve) > 1}

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"variants": rows, "data_fraction_test_iou": fractions}, f, indent=2)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(rows, fractions))
    print(render_markdown(rows, fractions))
    return rows


if __name__ == "__main__":
    compare()
