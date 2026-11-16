from src.gee_client import assess_water_balance, compute_indices, recommend_usage_from_indices


def _bands(b2, b3, b4, b5, b8):
    return {"B2": b2, "B3": b3, "B4": b4, "B5": b5, "B8": b8, "B11": 1000}


def test_compute_indices_clear_vegetated_water():
    # High NIR relative to red (vegetation nearby), green ~ red-edge (low chlorophyll)
    bands = _bands(b2=800, b3=900, b4=600, b5=650, b8=3000)
    indices = compute_indices(bands)
    assert indices["ndvi"] > 0.5
    assert -1 <= indices["ndwi"] <= 1
    assert -1 <= indices["ndci"] <= 1
    assert -1 <= indices["ndti"] <= 1


def test_recommend_usage_flags_algae_on_high_ndci():
    bands = _bands(b2=800, b3=900, b4=600, b5=950, b8=1200)
    indices = compute_indices(bands)
    usage = recommend_usage_from_indices(indices)
    assert usage["label"] == "traitement requis"


def test_recommend_usage_flags_turbidity_on_high_ndti():
    bands = _bands(b2=800, b3=700, b4=900, b5=920, b8=1100)
    indices = compute_indices(bands)
    usage = recommend_usage_from_indices(indices)
    assert usage["label"] in {"irrigation possible", "traitement requis"}


def test_recommend_usage_default_when_indices_unremarkable():
    bands = _bands(b2=800, b3=900, b4=850, b5=860, b8=1000)
    indices = compute_indices(bands)
    usage = recommend_usage_from_indices(indices)
    assert usage["label"] in {"qualité apparente correcte", "irrigation possible", "traitement requis"}


def test_water_balance_none_without_prior_data():
    assert assess_water_balance(None, 0.1) is None
    assert assess_water_balance(-10, None) is None


def test_water_balance_flags_mismatch_when_demand_holds_despite_drop():
    explanation = assess_water_balance(-10, 0.01)
    assert "ne s'adapte pas" in explanation


def test_water_balance_coherent_when_demand_drops_with_supply():
    explanation = assess_water_balance(-10, -0.05)
    assert "cohérent" in explanation


def test_water_balance_balanced_growth():
    explanation = assess_water_balance(8, 0.05)
    assert "équilibrée" in explanation


def test_water_balance_no_signal_for_small_change():
    assert assess_water_balance(1, 0.3) is None
