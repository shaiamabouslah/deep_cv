import torch

from src.model import build_segmentation_model


def test_segmentation_model_output_shape():
    model = build_segmentation_model("resnet34", encoder_weights=None)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)
