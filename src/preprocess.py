"""Image preprocessing stage: cleans and standardizes the product catalog
before any embedding is computed. This is the explicit Computer Vision step
of the pipeline (classical CV techniques, no deep model involved here).
"""
import json
import os

import cv2
import imagehash
import numpy as np
from PIL import Image


def is_readable(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def sharpness_score(image: Image.Image) -> float:
    """Variance of the Laplacian: a classic blur-detection metric. Low
    values mean the image is blurry/low quality and should be dropped."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def remove_background(image: Image.Image) -> Image.Image:
    """Classical foreground segmentation (GrabCut) assuming the product is
    roughly centered on a plain background, as in typical catalog photos."""
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return image
    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    result = img * binary_mask[:, :, None] + 255 * (1 - binary_mask[:, :, None])
    return Image.fromarray(result.astype("uint8"))


def standardize(image: Image.Image, size: int) -> Image.Image:
    """Resizes preserving aspect ratio, then pads (letterbox) to a square."""
    image = image.convert("RGB")
    ratio = size / max(image.size)
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    resized = image.resize(new_size, Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def preprocess_catalog(images, params):
    """Takes a list of PIL images, returns the cleaned/standardized ones
    plus a report of what was dropped and why."""
    size = params["preprocess"]["image_size"]
    min_sharpness = params["preprocess"]["min_sharpness"]
    remove_bg = params["preprocess"]["remove_background"]

    kept, seen_hashes = [], set()
    report = {"input": len(images), "dropped_duplicate": 0, "dropped_blurry": 0, "kept": 0}

    for image in images:
        image = image.convert("RGB")

        phash = imagehash.phash(image)
        if phash in seen_hashes:
            report["dropped_duplicate"] += 1
            continue
        seen_hashes.add(phash)

        if sharpness_score(image) < min_sharpness:
            report["dropped_blurry"] += 1
            continue

        if remove_bg:
            image = remove_background(image)
        image = standardize(image, size)
        kept.append(image)

    report["kept"] = len(kept)
    return kept, report


def run(params_path="params.yaml"):
    from src.data import load_params
    params = load_params(params_path)
    os.makedirs("reports", exist_ok=True)
    # Placeholder entry point wired up once the catalog source is finalized
    # (called from src/build_catalog.py with the actual loaded images).
    with open("reports/preprocess_report.json", "w") as f:
        json.dump({"status": "run preprocess via src.build_catalog"}, f)


if __name__ == "__main__":
    run()
