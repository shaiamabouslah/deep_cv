from src import lakes


def _ring(n):
    return [[i * 0.001, (i % 7) * 0.001] for i in range(n)] + [[0.0, 0.0]]


def test_small_outlines_are_kept_as_they_are():
    ring = _ring(50)
    assert lakes.thin(ring) == ring


def test_big_outlines_are_thinned_and_stay_closed():
    thinned = lakes.thin(_ring(5000), max_vertices=100)
    assert len(thinned) <= 101
    assert thinned[0] == thinned[-1]


def test_description_names_the_type_and_keeps_the_hydrolakes_facts():
    info = lakes.describe(
        {"Hylak_id": 14809, "Lake_name": None, "Country": "Tunisia", "Continent": "Africa", "Lake_type": 2,
         "Lake_area": 24.834, "Depth_avg": 3.1, "Vol_total": 0.07},
        _ring(10),
    )
    assert info["type"] == "reservoir" and info["id"] == 14809 and info["name"] is None
    assert info["area_km2"] == 24.83 and info["volume_mm3"] == 0.07
    assert len(info["polygon"]) == 11


def test_a_lake_too_large_to_analyse_keeps_its_facts_but_has_no_outline():
    info = lakes.describe({"Hylak_id": 2, "Lake_type": 1, "Lake_area": 68000, "Country": "Uganda"}, None)
    assert info["polygon"] is None and info["area_km2"] == 68000


def test_outer_ring_handles_every_geometry_type_a_lake_with_islands_can_have():
    small, big = [[0, 0], [1, 0], [0, 0]], [[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]
    assert lakes.outer_ring({"type": "Polygon", "coordinates": [big, small]}) == big
    assert lakes.outer_ring({"type": "MultiPolygon", "coordinates": [[small], [big]]}) == big
    collection = {"type": "GeometryCollection", "geometries": [
        {"type": "Polygon", "coordinates": [small]}, {"type": "Polygon", "coordinates": [big]},
        {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}]}
    assert lakes.outer_ring(collection) == big
    assert lakes.outer_ring({"type": "Point", "coordinates": [0, 0]}) is None


def test_unknown_type_codes_do_not_crash():
    info = lakes.describe({"Hylak_id": 1, "Lake_type": 3, "Lake_area": 1}, _ring(4))
    assert info["type"] == "other"
