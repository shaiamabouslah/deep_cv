"""Shared config loading and image transforms for the water-monitoring pipeline."""
import yaml
from torchvision import transforms


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def build_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_normalize_transform():
    """ToTensor + normalize only, for an image already standardized to the
    model's input size (e.g. via src.preprocess.standardize) — avoids
    resizing twice and keeps the mask pixel-aligned with the input image."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
