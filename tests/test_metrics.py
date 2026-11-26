import torch

from src.metrics import iou_score, water_area_fraction


def test_water_area_fraction_all_water():
    mask = torch.ones(1, 1, 4, 4)
    assert water_area_fraction(mask) == 1.0


def test_water_area_fraction_half():
    mask = torch.zeros(1, 1, 2, 4)
    mask[:, :, 0, :] = 1
    assert water_area_fraction(mask) == 0.5


def test_water_area_fraction_ignores_invalid_pixels():
    mask = torch.zeros(1, 1, 2, 4)
    mask[:, :, 0, :2] = 1
    valid = torch.zeros(2, 4, dtype=torch.bool)
    valid[0, :] = True
    assert water_area_fraction(mask, valid) == 0.5


def test_water_area_fraction_no_valid_pixels():
    assert water_area_fraction(torch.ones(1, 1, 2, 2), torch.zeros(2, 2, dtype=torch.bool)) == 0.0


def test_iou_perfect_match():
    mask = torch.ones(1, 1, 4, 4)
    assert iou_score(mask, mask) > 0.99


def test_iou_no_overlap():
    pred = torch.zeros(1, 1, 4, 4)
    pred[:, :, :2, :] = 1
    true = torch.zeros(1, 1, 4, 4)
    true[:, :, 2:, :] = 1
    assert iou_score(pred, true) < 0.01
