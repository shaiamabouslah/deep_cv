"""Metrics computed from a predicted water mask."""
import torch


def water_area_fraction(mask: torch.Tensor, valid: torch.Tensor = None) -> float:
    """Fraction of pixels classified as water, in [0, 1]. With `valid`, only
    those pixels count (no-data areas are neither water nor land)."""
    mask = mask.float()
    if valid is None:
        return mask.mean().item()
    valid = valid.float()
    total = valid.sum().item()
    return (mask * valid).sum().item() / total if total else 0.0


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred_mask.float()
    true = true_mask.float()
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()
