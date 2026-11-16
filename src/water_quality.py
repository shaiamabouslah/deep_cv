"""Water usability analysis: beyond "is there water", estimate whether the
detected water looks usable, and for what purpose.

This reuses the segmentation mask: only the pixels classified as water are
analyzed, with classical color-based proxies (no spectral bands beyond
RGB are available from a normal image):

  - brightness: darker water often means more suspended sediment (turbidity)
  - greenness index: a high green share can indicate algae / eutrophication

These are illustrative heuristics, not a validated water-quality model —
thresholds would need calibration against real in-situ measurements
(turbidity sensors, chlorophyll-a) before any operational use. They exist
to demonstrate the decision-support layer end-to-end.
"""
import numpy as np


def extract_water_pixels(image_array: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
    """image_array: HxWx3 uint8 RGB. mask_array: HxW, truthy where water."""
    return image_array[mask_array.astype(bool)]


def compute_quality_metrics(pixels: np.ndarray):
    if pixels.size == 0:
        return None

    mean_rgb = pixels.mean(axis=0)
    r, g, b = mean_rgb
    brightness = float(mean_rgb.mean())
    greenness_index = float(g / (r + g + b + 1e-6))
    turbidity_proxy = float(1 - brightness / 255)

    return {
        "mean_rgb": [round(float(v), 1) for v in mean_rgb],
        "brightness": round(brightness, 1),
        "greenness_index": round(greenness_index, 3),
        "turbidity_proxy": round(turbidity_proxy, 3),
    }


def recommend_usage(metrics):
    if metrics is None:
        return {
            "label": "indéterminé",
            "detail": "Pas assez de pixels d'eau détectés pour analyser la qualité.",
        }

    if metrics["greenness_index"] > 0.40:
        return {
            "label": "traitement requis",
            "detail": "Teinte verte dominante : présence probable d'algues/eutrophisation. "
                      "Déconseillé pour l'irrigation ou la consommation sans traitement.",
        }

    if metrics["turbidity_proxy"] > 0.55:
        return {
            "label": "irrigation possible",
            "detail": "Eau trouble (forte turbidité apparente) : utilisable pour l'irrigation, "
                      "non recommandée pour la consommation sans traitement.",
        }

    return {
        "label": "qualité apparente correcte",
        "detail": "Eau claire en apparence : compatible avec l'irrigation ; la potabilité "
                  "doit être confirmée par une analyse physico-chimique réelle.",
    }


def analyze(image_array: np.ndarray, mask_array: np.ndarray):
    pixels = extract_water_pixels(image_array, mask_array)
    metrics = compute_quality_metrics(pixels)
    usage = recommend_usage(metrics)
    return {"metrics": metrics, "usage": usage}
