"""Finds the lake or reservoir at a point of the world, from HydroLAKES
(Messager et al. 2016), so the analysis of a site picked anywhere uses the
water body's real outline instead of a fixed circle. Also returns the facts
HydroLAKES holds about it (area, type, country, mean depth, volume).

The outline is simplified (a few tens of metres) so a very detailed
polygon stays small enough to send to the browser and back.
"""
from src.gee_client import _ensure_initialized
from src.world_sites import HYDROLAKES, LAKE_TYPES

SEARCH_RADIUS_M = 1500
MAX_VERTICES = 600
MAX_ANALYSIS_KM2 = 2000
PROPS = ["Hylak_id", "Lake_name", "Country", "Continent", "Lake_type", "Lake_area", "Depth_avg", "Vol_total"]


def thin(ring, max_vertices=MAX_VERTICES):
    """Keeps at most max_vertices evenly spaced points of a closed ring."""
    if len(ring) <= max_vertices:
        return [list(p) for p in ring]
    step = len(ring) / max_vertices
    out = [list(ring[int(i * step)]) for i in range(max_vertices)]
    if out[0] != out[-1]:
        out.append(list(out[0]))
    return out


def outer_ring(geometry):
    """Main outline (the largest exterior ring) of a GeoJSON Polygon,
    MultiPolygon or GeometryCollection; lakes with islands can be any of them."""
    kind = geometry["type"]
    if kind == "Polygon":
        return geometry["coordinates"][0]
    if kind == "MultiPolygon":
        rings = [polygon[0] for polygon in geometry["coordinates"]]
    elif kind == "GeometryCollection":
        rings = [r for g in geometry["geometries"] if (r := outer_ring(g))]
    else:
        return None
    return max(rings, key=len) if rings else None


def describe(properties, ring):
    """Plain dict for the API from a HydroLAKES feature's properties and outline
    (None for lakes too large to analyse). HydroLAKES volumes are in million m3."""
    lake_type = LAKE_TYPES.get(int(properties.get("Lake_type") or 0), "other")
    return {
        "id": int(properties["Hylak_id"]),
        "name": properties.get("Lake_name") or None,
        "country": properties.get("Country"), "continent": properties.get("Continent"),
        "type": lake_type, "area_km2": round(float(properties.get("Lake_area") or 0), 2),
        "mean_depth_m": properties.get("Depth_avg"), "volume_mm3": properties.get("Vol_total"),
        "polygon": thin(ring) if ring else None,
    }


def lake_at(lat, lon, project=None):
    """The lake containing (lat, lon), else the nearest within SEARCH_RADIUS_M;
    None when there is no mapped water body there."""
    import ee

    _ensure_initialized(project)
    point = ee.Geometry.Point([lon, lat])
    nearby = ee.FeatureCollection(HYDROLAKES).filterBounds(point.buffer(SEARCH_RADIUS_M))
    ranked = nearby.map(lambda f: f.set("distance", f.geometry().distance(point, 30))).sort("distance")
    if ranked.size().getInfo() == 0:
        return None
    best = ee.Feature(ranked.first())
    props = best.toDictionary(PROPS).getInfo()
    area_km2 = float(props.get("Lake_area") or 1.0)
    if area_km2 > MAX_ANALYSIS_KM2:
        return describe(props, None)
    error_m = max(30.0, min(300.0, area_km2 ** 0.5 * 8))
    ring = outer_ring(best.geometry().simplify(error_m).getInfo())
    return describe(props, ring)
