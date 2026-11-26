"""Fine-tunes the segmentation model on worldwide tiles (data/world_train/)
with colour and scale augmentation, then writes a candidate checkpoint.

Why: the live model was trained on one dataset and validated on Tunisian
dams; on the world benchmark it falls from IoU 0.72 to 0.45, mostly on
natural lakes, turbid or dark water and unusual lighting. This trains it on
tiles rendered and labelled exactly like the benchmark but from different
lakes, with brightness / contrast / gamma / channel-gain jitter and random
zoom (tiles cover 4 to 24 km, so ground resolution varies a lot), and
ignores invalid pixels in the loss. The benchmark and the Tunisian tiles are
never touched here: model selection uses a validation split of the TRAINING
sites only. Whether the candidate may replace the live model is decided by
src/promote.py.

    python -m src.finetune_world [--epochs N] [--no-color]
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.model import build_segmentation_model, load_model
from src.preprocess import nodata_mask
from src.sar_model import masked_loss

TRAIN_DIR = "data/world_train"
OUT_PATH = "models/model_v3_world.pt"
BASE_PATH = "models/model.pt"
IMAGE_SIZE = 256
MIN_VALID_FRACTION = 0.5
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_tile(directory, entry):
    image = Image.open(os.path.join(directory, "images", entry["file"])).convert("RGB")
    label = np.array(Image.open(os.path.join(directory, "masks", entry["file"])).convert("L"))
    valid = ((label < 64) | (label > 192)) & ~nodata_mask(np.array(image))
    return image, (label > 192).astype(np.float32), valid.astype(np.float32)


def usable_entries(directory):
    with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    keep = []
    for entry in manifest:
        _, _, valid = load_tile(directory, entry)
        if valid.mean() >= MIN_VALID_FRACTION:
            keep.append(entry)
    return keep


def split_by_site(entries, val_frac=0.12, seed=0):
    sites = sorted({e["site_id"] for e in entries})
    random.Random(seed).shuffle(sites)
    val_sites = set(sites[: max(1, round(len(sites) * val_frac))])
    return [e for e in entries if e["site_id"] not in val_sites], [e for e in entries if e["site_id"] in val_sites]


def color_jitter(tensor):
    """tensor: 3xHxW in [0, 1]. Simulates other exposure stretches, haze and sensors."""
    brightness = random.uniform(0.6, 1.5)
    contrast = random.uniform(0.7, 1.4)
    gamma = random.uniform(0.7, 1.5)
    gains = torch.tensor([random.uniform(0.9, 1.1) for _ in range(3)]).view(3, 1, 1)
    out = tensor.clamp(0, 1) ** gamma
    mean = out.mean()
    out = ((out - mean) * contrast + mean) * brightness * gains
    return out.clamp(0, 1)


class TileDataset(Dataset):
    def __init__(self, directory, entries, augment, color=True):
        self.directory, self.entries, self.augment, self.color = directory, entries, augment, color

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        image, water, valid = load_tile(self.directory, self.entries[i])
        x = transforms.functional.to_tensor(image)
        y, m = torch.tensor(water)[None], torch.tensor(valid)[None]
        if self.augment:
            x, y, m = self._geometry(x, y, m)
            if self.color:
                x = color_jitter(x)
        size = [IMAGE_SIZE, IMAGE_SIZE]
        x = transforms.functional.resize(x, size, antialias=True)
        y = transforms.functional.resize(y, size, interpolation=transforms.InterpolationMode.NEAREST)
        m = transforms.functional.resize(m, size, interpolation=transforms.InterpolationMode.NEAREST)
        return transforms.functional.normalize(x, MEAN, STD), y, m

    @staticmethod
    def _geometry(x, y, m):
        h, w = x.shape[-2:]
        scale = random.uniform(0.45, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        top, left = random.randint(0, h - ch), random.randint(0, w - cw)
        x, y, m = (t[..., top:top + ch, left:left + cw] for t in (x, y, m))
        if random.random() < 0.5:
            x, y, m = x.flip(-1), y.flip(-1), m.flip(-1)
        if random.random() < 0.5:
            x, y, m = x.flip(-2), y.flip(-2), m.flip(-2)
        k = random.randint(0, 3)
        return tuple(torch.rot90(t, k, (-2, -1)) for t in (x, y, m))


def pooled_iou(model, loader, threshold=0.5):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for x, y, m in loader:
            pred = torch.sigmoid(model(x)) > threshold
            truth, valid = y.bool(), m.bool()
            tp += int((pred & truth & valid).sum())
            fp += int((pred & ~truth & valid).sum())
            fn += int((~pred & truth & valid).sum())
    return tp / max(tp + fp + fn, 1)


def tune_threshold(model, loader, grid=(0.3, 0.4, 0.5, 0.6, 0.7)):
    return max(grid, key=lambda t: pooled_iou(model, loader, t))


def finetune(train_entries, val_entries, directory=TRAIN_DIR, base_path=BASE_PATH, epochs=12, lr=1e-4,
             batch_size=8, color=True, seed=0, log=print):
    torch.manual_seed(seed)
    random.seed(seed)
    if base_path and os.path.exists(base_path):
        model, _, _ = load_model(base_path)
    else:
        model = build_segmentation_model("resnet34", encoder_weights=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(TileDataset(directory, train_entries, True, color), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TileDataset(directory, val_entries, False), batch_size=batch_size)

    best, best_state, curve = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        for x, y, m in loader:
            optimizer.zero_grad()
            masked_loss(model(x), y, m).backward()
            optimizer.step()
        scheduler.step()
        val_iou = pooled_iou(model, val_loader)
        curve.append(round(val_iou, 4))
        if val_iou > best:
            best, best_state = val_iou, {k: v.clone() for k, v in model.state_dict().items()}
        log(f"epoch {epoch + 1}/{epochs} val_iou={val_iou:.4f} best={best:.4f}")
    model.load_state_dict(best_state)
    return model, tune_threshold(model, val_loader), curve


def run(epochs=12, color=True, directory=TRAIN_DIR, out_path=OUT_PATH, track=False):
    entries = usable_entries(directory)
    train_entries, val_entries = split_by_site(entries)
    print(f"{len(train_entries)} training tiles, {len(val_entries)} validation tiles "
          f"({len({e['site_id'] for e in train_entries})} / {len({e['site_id'] for e in val_entries})} sites)", flush=True)
    model, threshold, curve = finetune(train_entries, val_entries, directory, epochs=epochs, color=color)
    torch.save({
        "state_dict": model.state_dict(), "encoder_name": "resnet34", "image_size": IMAGE_SIZE,
        "threshold": float(threshold),
        "metrics": {"val_iou_curve": curve, "color_augmentation": color, "n_train_tiles": len(train_entries),
                    "init": "models/model.pt"},
    }, out_path)
    print("saved", out_path, "threshold", threshold)
    if track:
        from src import tracking

        tracking.log_experiment(
            "segmentation-water", "finetune (worldwide tiles)",
            params={"init": BASE_PATH, "epochs": epochs, "lr": 1e-4, "batch_size": 8, "loss": "masked dice+bce",
                    "color_augmentation": color, "n_train_tiles": len(train_entries), "n_val_tiles": len(val_entries)},
            metrics={"best_val_iou": max(curve), "threshold": float(threshold)}, curves={"val_iou": curve},
            tags={"source": "finetune_world", "candidate": out_path})
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()
    run(args.epochs, not args.no_color, out_path=args.out, track=True)
