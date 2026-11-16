"""Real satellite analysis via Google Earth Engine: fetches actual Sentinel-2
spectral bands (not just RGB) for a site, and computes validated remote
sensing indices instead of the RGB color heuristic in water_quality.py.

Requires the user to have authenticated once on this machine:
    python -m ee.cli.eecli authenticate
or, in Python:
    import ee; ee.Authenticate(); ee.Initialize(project="<your-gcp-project>")
This opens a browser sign-in flow — Claude cannot do this step; the user
must run it themselves once, then credentials are cached locally.

Sentinel-2 bands used (10-20m resolution, surface reflectance):
  B2 Blue, B3 Green, B4 Red, B5 Red Edge 1, B8 NIR, B11 SWIR

Indices:
  NDWI  = (Green - NIR) / (Green + NIR)      McFeeters 1996 — open water
  NDVI  = (NIR - Red)   / (NIR + Red)        vegetation / irrigated land nearby
  NDCI  = (RedEdge - Red) / (RedEdge + Red)  Mishra & Mishra 2012 — chlorophyll-a proxy
  NDTI  = (Red - Green) / (Red + Green)      simple turbidity proxy
"""
import datetime

_EE_INITIALIZED = False


class GEENotAuthenticated(RuntimeError):
    pass


def _ensure_initialized(project=None):
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return
    import ee
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise GEENotAuthenticated(
            "Google Earth Engine n'est pas authentifié sur cette machine. "
            "Lance une fois dans un terminal : python -c \"import ee; ee.Authenticate()\" "
            "puis réessaie."
        ) from exc
    _EE_INITIALIZED = True


def fetch_sentinel2_bands(lat, lon, buffer_m=500, days_back=30, project=None):
    """Returns mean band reflectance over a buffer around (lat, lon), from
    the least cloudy Sentinel-2 scene in the last `days_back` days."""
    _ensure_initialized(project)
    import ee

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m)

    end = datetime.date.today()
    start = end - datetime.timedelta(days=days_back)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(str(start), str(end))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    image = collection.first()
    info = image.getInfo()
    if info is None:
        raise ValueError(
            f"Aucune image Sentinel-2 trouvée dans les {days_back} derniers jours pour ce site."
        )

    bands = image.select(["B2", "B3", "B4", "B5", "B8", "B11"])
    means = bands.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10).getInfo()

    date_ms = image.get("system:time_start").getInfo()
    acquisition_date = datetime.datetime.utcfromtimestamp(date_ms / 1000).date().isoformat()
    cloud_pct = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    return {
        "bands": means,
        "acquisition_date": acquisition_date,
        "cloud_pct": round(cloud_pct, 1) if cloud_pct is not None else None,
    }


def compute_indices(bands):
    b2, b3, b4, b5, b8 = bands["B2"], bands["B3"], bands["B4"], bands["B5"], bands["B8"]
    eps = 1e-6

    ndwi = (b3 - b8) / (b3 + b8 + eps)
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndci = (b5 - b4) / (b5 + b4 + eps)
    ndti = (b4 - b3) / (b4 + b3 + eps)

    return {
        "ndwi": round(ndwi, 3),
        "ndvi": round(ndvi, 3),
        "ndci": round(ndci, 3),
        "ndti": round(ndti, 3),
    }


def recommend_usage_from_indices(indices):
    """Thresholds are indicative starting points from the remote-sensing
    literature (Mishra & Mishra 2012 for NDCI; simple NDTI heuristics), not
    calibrated against in-situ measurements for Tunisian dams specifically —
    flag this clearly in any report generated from it."""
    if indices["ndci"] > 0.15:
        return {
            "label": "traitement requis",
            "detail": f"NDCI={indices['ndci']} : forte probabilité de chlorophylle-a élevée "
                      "(efflorescence algale). Traitement nécessaire avant usage.",
        }
    if indices["ndti"] > 0.05:
        return {
            "label": "irrigation possible",
            "detail": f"NDTI={indices['ndti']} : turbidité significative. Adapté à l'irrigation, "
                      "non recommandé pour la consommation sans traitement.",
        }
    return {
        "label": "qualité apparente correcte",
        "detail": f"NDCI={indices['ndci']}, NDTI={indices['ndti']} : pas de signe fort d'algues "
                  "ou de turbidité. Potabilité à confirmer par analyse physico-chimique.",
    }


def analyze_site_live(lat, lon, buffer_m=500, days_back=30, project=None):
    result = fetch_sentinel2_bands(lat, lon, buffer_m=buffer_m, days_back=days_back, project=project)
    indices = compute_indices(result["bands"])
    usage = recommend_usage_from_indices(indices)

    vegetation_note = (
        "zone agricole/végétalisée active autour du site" if indices["ndvi"] > 0.3
        else "peu de végétation détectée autour du site"
    )

    return {
        "acquisition_date": result["acquisition_date"],
        "cloud_pct": result["cloud_pct"],
        "indices": indices,
        "vegetation_note": vegetation_note,
        "usage": usage,
    }
