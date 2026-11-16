"""Metrics computed from a predicted water mask."""
import torch


def water_area_fraction(mask: torch.Tensor) -> float:
    """Fraction of pixels classified as water, in [0, 1]."""
    return mask.float().mean().item()


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred_mask.float()
    true = true_mask.float()
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()
