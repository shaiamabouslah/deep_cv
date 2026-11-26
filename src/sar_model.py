"""Water segmentation from Sentinel-1 radar (VV, VH), learned from optical
masks (see src/build_sar_dataset.py), with the classical baselines it has to
beat. Radar needs no clear sky, so this is what can measure a reservoir in
the weeks optical imagery is clouded.

Everything is scored on valid pixels only (optical cloud, shadow, snow and
no-data pixels carry no label), with overlap pooled over the whole split.

Baselines, in increasing strength:
  - threshold: the single best VV (or VH) cut-off found on the training
    patches, optionally on a 3x3 median-filtered image. The standard
    classical method for radar water mapping.
  - gbm: gradient-boosted trees per pixel on VV, VH, their difference and
    local means / standard deviations (5x5), i.e. strong hand-made features.
The U-Net has to beat both, including on regions it never saw.
"""
import json
import os
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader, Dataset

SAR_DIR = "data/sar"
VV_RANGE, VH_RANGE = (-40.0, 10.0), (-50.0, 0.0)


def load_patch(path):
    d = np.load(path)
    return (d["vv"].astype(np.float32), d["vh"].astype(np.float32),
            d["water"].astype(np.float32), d["valid"].astype(np.float32))


def to_input(vv, vh):
    """3 channels roughly centered and unit-scaled: VV, VH and their difference."""
    vv, vh = np.clip(vv, *VV_RANGE), np.clip(vh, *VH_RANGE)
    return np.stack([(vv + 15) / 7.5, (vh + 22) / 8.0, (vv - vh - 8) / 5.0]).astype(np.float32)


def split_sites(manifest, seed=0, val_frac=0.15, test_frac=0.15):
    """Site-level split of the world sites (all dates of a site stay together,
    so no lake leaks across splits). Tunisian dams are never trained on: they
    are the held-out region the model is meant to serve."""
    world_sites = sorted({r["site_id"] for r in manifest if r["kind"] == "world"})
    random.Random(seed).shuffle(world_sites)
    n_val, n_test = round(len(world_sites) * val_frac), round(len(world_sites) * test_frac)
    groups = {"val": world_sites[:n_val], "test_world": world_sites[n_val:n_val + n_test]}
    held = {s for g in groups.values() for s in g}
    out = {"train": [r for r in manifest if r["kind"] == "world" and r["site_id"] not in held]}
    out["val"] = [r for r in manifest if r["site_id"] in groups["val"]]
    out["test_world"] = [r for r in manifest if r["site_id"] in groups["test_world"]]
    out["test_tunisia"] = [r for r in manifest if r["kind"] == "tunisia"]
    return out


class SarDataset(Dataset):
    def __init__(self, rows, sar_dir=SAR_DIR, augment=False):
        self.paths = [os.path.join(sar_dir, "patches", r["file"]) for r in rows]
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        vv, vh, water, valid = load_patch(self.paths[i])
        x, y, m = torch.tensor(to_input(vv, vh)), torch.tensor(water)[None], torch.tensor(valid)[None]
        if self.augment:
            if random.random() < 0.5:
                x, y, m = x.flip(-1), y.flip(-1), m.flip(-1)
            if random.random() < 0.5:
                x, y, m = x.flip(-2), y.flip(-2), m.flip(-2)
            k = random.randint(0, 3)
            x, y, m = (torch.rot90(t, k, (-2, -1)) for t in (x, y, m))
        return x, y, m


def build_model(encoder="resnet18", pretrained=True):
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def masked_loss(logits, target, valid):
    """Dice + BCE over valid pixels only."""
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    bce = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum()
    return dice + bce / valid.sum().clamp(min=1)


def counts(pred, truth, valid):
    pred, truth, valid = pred.astype(bool), truth.astype(bool), valid.astype(bool)
    return (int((pred & truth & valid).sum()), int((pred & ~truth & valid).sum()), int((~pred & truth & valid).sum()))


def scores(total):
    tp, fp, fn = (int(v) for v in total)  # plain Python numbers: NumPy scalars in a checkpoint break torch.load's safe mode
    return {"iou": round(tp / max(tp + fp + fn, 1), 4), "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4)}


def predict_masks(model, rows, sar_dir=SAR_DIR, threshold=0.5):
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            vv, vh, water, valid = load_patch(os.path.join(sar_dir, "patches", r["file"]))
            prob = torch.sigmoid(model(torch.tensor(to_input(vv, vh))[None]))[0, 0].numpy()
            out.append((prob > threshold, water, valid))
    return out


def evaluate_model(model, rows, sar_dir=SAR_DIR, threshold=0.5):
    total = np.zeros(3, dtype=int)
    for pred, water, valid in predict_masks(model, rows, sar_dir, threshold):
        total += counts(pred, water, valid)
    return scores(tuple(total))


def train(train_rows, val_rows, epochs=25, lr=3e-4, batch_size=8, seed=0, sar_dir=SAR_DIR, pretrained=True,
          encoder="resnet18", log=print):
    torch.manual_seed(seed)
    random.seed(seed)
    model = build_model(encoder, pretrained)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(SarDataset(train_rows, sar_dir, augment=True), batch_size=batch_size, shuffle=True)
    best, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        for x, y, m in loader:
            optimizer.zero_grad()
            masked_loss(model(x), y, m).backward()
            optimizer.step()
        scheduler.step()
        val_iou = evaluate_model(model, val_rows, sar_dir)["iou"] if val_rows else 0.0
        history.append(val_iou)
        if val_iou > best:
            best, best_state = val_iou, {k: v.clone() for k, v in model.state_dict().items()}
        log(f"epoch {epoch + 1}/{epochs} val_iou={val_iou:.4f} best={best:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def smooth(vv, vh=None):
    return ndimage.median_filter(vv, size=3)


class ThresholdBaseline:
    """Best single cut-off of one polarisation, tuned on the training patches."""

    def __init__(self, band="vv", median=False):
        self.band, self.median = band, median
        self.threshold = None

    def _channel(self, vv, vh):
        channel = vv if self.band == "vv" else vh
        return ndimage.median_filter(channel, size=3) if self.median else channel

    def fit(self, rows, sar_dir=SAR_DIR, grid=None):
        grid = grid if grid is not None else np.arange(-30, -5, 0.5)
        data = [(self._channel(*load_patch(os.path.join(sar_dir, "patches", r["file"]))[:2]),
                 *load_patch(os.path.join(sar_dir, "patches", r["file"]))[2:]) for r in rows]
        best = (-1.0, grid[0])
        for t in grid:
            total = np.zeros(3, dtype=int)
            for channel, water, valid in data:
                total += counts(channel < t, water, valid)
            iou = scores(tuple(total))["iou"]
            if iou > best[0]:
                best = (iou, float(t))
        self.threshold = best[1]
        return self

    def predict(self, vv, vh):
        return self._channel(vv, vh) < self.threshold


def pixel_features(vv, vh):
    channels = [vv, vh, vv - vh]
    for c in (vv, vh):
        mean = ndimage.uniform_filter(c, size=5)
        channels += [mean, np.sqrt(np.maximum(ndimage.uniform_filter(c * c, size=5) - mean * mean, 0))]
    return np.stack(channels, axis=-1)


class GbmBaseline:
    """Gradient-boosted trees per pixel on hand-made radar features."""

    def __init__(self, max_pixels=300_000, seed=0):
        self.max_pixels, self.seed = max_pixels, seed
        self.model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed)

    def fit(self, rows, sar_dir=SAR_DIR):
        xs, ys = [], []
        for r in rows:
            vv, vh, water, valid = load_patch(os.path.join(sar_dir, "patches", r["file"]))
            keep = valid.astype(bool)
            xs.append(pixel_features(vv, vh)[keep])
            ys.append(water[keep])
        x, y = np.concatenate(xs), np.concatenate(ys)
        if len(x) > self.max_pixels:
            idx = np.random.default_rng(self.seed).choice(len(x), self.max_pixels, replace=False)
            x, y = x[idx], y[idx]
        self.model.fit(x, y)
        return self

    def predict(self, vv, vh):
        features = pixel_features(vv, vh)
        return self.model.predict(features.reshape(-1, features.shape[-1])).reshape(vv.shape).astype(bool)


def evaluate_baseline(baseline, rows, sar_dir=SAR_DIR):
    total = np.zeros(3, dtype=int)
    for r in rows:
        vv, vh, water, valid = load_patch(os.path.join(sar_dir, "patches", r["file"]))
        total += counts(baseline.predict(vv, vh), water, valid)
    return scores(tuple(total))


def load_manifest(sar_dir=SAR_DIR):
    with open(os.path.join(sar_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)
