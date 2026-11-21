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
import socket

# Prevents a stuck/slow network call to Earth Engine from hanging the API
# request forever (e.g. right after registering a brand-new GCP project,
# propagation can be slow for the first few requests).
socket.setdefaulttimeout(30)

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


def fetch_sentinel2_bands(lat, lon, buffer_m=500, days_back=30, project=None, polygon_coords=None):
    """Returns mean band reflectance over the region of interest, from the
    least cloudy Sentinel-2 scene in the last `days_back` days.

    By default the region is a circular buffer around (lat, lon). If
    `polygon_coords` is given (a list of [lon, lat] pairs, as drawn by the
    user on the map), that exact polygon is used instead — a real area of
    interest rather than a fixed-radius point."""
    _ensure_initialized(project)
    import ee

    if polygon_coords:
        region = ee.Geometry.Polygon([polygon_coords])
    else:
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


def fetch_precipitation_mm(lat, lon, days_back=30, project=None, polygon_coords=None):
    """Total rainfall (mm) over the last `days_back` days around this site,
    from CHIRPS (UCSB-CHG/CHIRPS/DAILY) — already available through the
    same Earth Engine account, no extra credential needed. Explains WHY a
    water level moved: a drop despite heavy recent rain is abnormal (leak,
    extraction), while a drop with little/no rain is expected."""
    _ensure_initialized(project)
    import ee

    if polygon_coords:
        region = ee.Geometry.Polygon([polygon_coords])
    else:
        region = ee.Geometry.Point([lon, lat]).buffer(500)

    end = datetime.date.today()
    start = end - datetime.timedelta(days=days_back)

    daily_precip = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(str(start), str(end))
    total_image = daily_precip.sum()
    value = total_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=5000).getInfo()
    return round(value.get("precipitation") or 0.0, 1)


def interpret_rainfall(precipitation_mm, days_back=30):
    if precipitation_mm > 80:
        level = "forte"
    elif precipitation_mm > 20:
        level = "modérée"
    else:
        level = "faible"
    return {
        "mm": precipitation_mm,
        "days": days_back,
        "level": level,
        "note": f"Pluviométrie {level} : {precipitation_mm} mm cumulés sur les {days_back} derniers jours.",
    }


def explain_level_change(water_area_pct_change, rainfall):
    """Cross-references the observed water level trend with actual rainfall
    to flag whether a drop looks natural (no rain) or abnormal (dropping
    despite rain — possible leak, unauthorized extraction, evaporation
    well above normal)."""
    if water_area_pct_change is None:
        return None
    if water_area_pct_change < -5 and rainfall["level"] in {"modérée", "forte"}:
        return (
            "Baisse du niveau malgré une pluviométrie {} récente ({} mm) — "
            "anormal, à investiguer (fuite, surconsommation, évaporation excessive)."
        ).format(rainfall["level"], rainfall["mm"])
    if water_area_pct_change < -5:
        return (
            "Baisse du niveau cohérente avec la faible pluviométrie récente "
            "({} mm sur {} jours)."
        ).format(rainfall["mm"], rainfall["days"])
    if water_area_pct_change > 5 and rainfall["level"] in {"modérée", "forte"}:
        return (
            "Hausse du niveau cohérente avec la pluviométrie {} récente ({} mm)."
        ).format(rainfall["level"], rainfall["mm"])
    return None


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


def assess_water_balance(water_pct_change, ndvi_change):
    """Cross-references the dam's water level trend (supply) with the
    trend in surrounding vegetation/irrigated land, NDVI (a proxy for
    agricultural water demand). The interesting case is a mismatch: supply
    dropping while demand (NDVI) holds or grows means consumption isn't
    adjusting to the shortage — a genuine water-management concern, not
    just "the dam is low"."""
    if water_pct_change is None or ndvi_change is None:
        return None

    if water_pct_change < -5 and ndvi_change >= -0.02:
        return (
            "Offre en baisse mais demande (végétation/irrigation) stable ou en hausse — "
            "la consommation ne s'adapte pas à la baisse de disponibilité, risque de "
            "surexploitation."
        )
    if water_pct_change < -5 and ndvi_change < -0.02:
        return (
            "Offre et demande baissent ensemble — ajustement cohérent de la consommation "
            "à la ressource disponible."
        )
    if water_pct_change > 5 and ndvi_change > 0.02:
        return "Offre et demande augmentent ensemble — situation équilibrée."
    return None


def fetch_max_extent_area_m2(region, project=None):
    """Historical maximum observed water extent for this region, from JRC
    Global Surface Water (Pekel et al., 2016, 1984-2021 Landsat record) —
    a validated reference surface, used here as the "100% full" denominator
    instead of an arbitrary theoretical NDWI range."""
    _ensure_initialized(project)
    import ee

    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent")
    pixel_area = ee.Image.pixelArea()
    max_area = gsw.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=30, maxPixels=1e9, bestEffort=True
    ).get("max_extent")
    return max_area.getInfo() or 0.0


def fetch_calibrated_water_area(lat, lon, start=None, end=None, days_back=30, project=None,
                                 polygon_coords=None, buffer_m=6000, max_area_m2=None):
    """Real surface water area (classified NDWI>0 pixel count x pixel
    area), expressed as a percentage of this site's historical maximum
    observed extent (JRC Global Surface Water) — replaces the earlier
    `(mean_NDWI + 1) * 50` formula, which was an uncalibrated rescale with
    no physical meaning (see reports/validation_vs_official.md). This is
    still an approximation: it samples a circular buffer, not the dam's
    true polygon, so the absolute percentage can still diverge from the
    official volume-based fill rate — but the trend direction tracks real
    data much more reliably (validated 2026-11-21).

    `max_area_m2` lets a caller (e.g. backfill_monthly_history) compute the
    site's max extent once and reuse it across many months, since it
    doesn't change over time."""
    _ensure_initialized(project)
    import ee

    if polygon_coords:
        region = ee.Geometry.Polygon([polygon_coords])
    else:
        region = ee.Geometry.Point([lon, lat]).buffer(buffer_m)

    if start is None or end is None:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days_back)
        start, end = str(start_date), str(end_date)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    image = collection.first()
    if image.getInfo() is None:
        raise ValueError(f"Aucune image Sentinel-2 trouvée entre {start} et {end} pour ce site.")

    ndwi_img = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    water_mask = ndwi_img.gt(0)
    pixel_area = ee.Image.pixelArea()

    current_area = water_mask.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=10, maxPixels=1e9, bestEffort=True
    ).get("NDWI")

    if max_area_m2 is None:
        max_area_m2 = fetch_max_extent_area_m2(region, project=project)

    date_ms = image.get("system:time_start").getInfo()
    acquisition_date = datetime.datetime.utcfromtimestamp(date_ms / 1000).date().isoformat()
    current_m2 = current_area.getInfo() or 0.0

    pct = min(100.0, current_m2 / max_area_m2 * 100) if max_area_m2 else None

    return {
        "water_area_pct": round(pct, 1) if pct is not None else None,
        "current_area_ha": round(current_m2 / 10000, 2),
        "max_observed_area_ha": round(max_area_m2 / 10000, 2),
        "acquisition_date": acquisition_date,
        "max_area_m2": max_area_m2,
    }


def backfill_monthly_history(lat, lon, months_back=12, project=None, polygon_coords=None, buffer_m=6000):
    """Pulls one NDWI-based water-area estimate per month over the last
    `months_back` months, directly from the Sentinel-2 archive — so a
    site's history chart starts with real multi-month context instead of
    only the points accumulated from the moment someone starts clicking.
    Default is 12 months (not 6): a dam's level follows a seasonal cycle
    (refills in wet season, drawdown in dry season), so less than a full
    year risks showing only half that cycle and looking like a one-way
    trend when it isn't. Skips months with no usable image rather than
    failing the whole call. Several sequential GEE round trips — run this
    as a background task, not inline in a request handler."""
    _ensure_initialized(project)

    if polygon_coords:
        import ee
        region = ee.Geometry.Polygon([polygon_coords])
    else:
        import ee
        region = ee.Geometry.Point([lon, lat]).buffer(buffer_m)

    # Computed once and reused for every month: the site's historical max
    # extent doesn't change, so there's no reason to re-fetch it 12-24 times.
    max_area_m2 = fetch_max_extent_area_m2(region, project=project)

    today = datetime.date.today()
    results = []
    for i in range(months_back, 0, -1):
        month_end = today - datetime.timedelta(days=30 * (i - 1))
        month_start = today - datetime.timedelta(days=30 * i)
        try:
            point = fetch_calibrated_water_area(
                lat, lon, start=str(month_start), end=str(month_end),
                project=project, polygon_coords=polygon_coords, max_area_m2=max_area_m2,
                buffer_m=buffer_m,
            )
        except ValueError:
            continue  # no usable image that month
        if point["water_area_pct"] is None:
            continue
        results.append({"date": point["acquisition_date"], "water_area_pct": point["water_area_pct"]})

    return results


def analyze_site_live(lat, lon, buffer_m=500, days_back=30, project=None, polygon_coords=None,
                       area_buffer_m=6000):
    result = fetch_sentinel2_bands(
        lat, lon, buffer_m=buffer_m, days_back=days_back, project=project,
        polygon_coords=polygon_coords,
    )
    indices = compute_indices(result["bands"])
    usage = recommend_usage_from_indices(indices)

    vegetation_note = (
        "zone agricole/végétalisée active autour du site" if indices["ndvi"] > 0.3
        else "peu de végétation détectée autour du site"
    )

    precipitation_mm = fetch_precipitation_mm(
        lat, lon, days_back=days_back, project=project, polygon_coords=polygon_coords
    )
    rainfall = interpret_rainfall(precipitation_mm, days_back=days_back)

    # Calibrated water extent (real area / historical max extent) — see
    # fetch_calibrated_water_area's docstring and the validation report for
    # why this replaced a raw rescale of mean NDWI. Uses a wider buffer
    # than the spectral-quality indices above, since capturing real extent
    # needs more than a 500m circle around one point.
    water_area = fetch_calibrated_water_area(
        lat, lon, days_back=days_back, project=project,
        polygon_coords=polygon_coords, buffer_m=area_buffer_m,
    )

    return {
        "acquisition_date": result["acquisition_date"],
        "cloud_pct": result["cloud_pct"],
        "indices": indices,
        "vegetation_note": vegetation_note,
        "usage": usage,
        "rainfall": rainfall,
        "water_area": water_area,
    }
