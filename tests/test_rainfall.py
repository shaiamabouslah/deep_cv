from src.gee_client import explain_level_change, interpret_rainfall


def test_interpret_rainfall_levels():
    assert interpret_rainfall(5.0)["level"] == "faible"
    assert interpret_rainfall(40.0)["level"] == "modérée"
    assert interpret_rainfall(120.0)["level"] == "forte"


def test_explain_drop_despite_rain_is_flagged_abnormal():
    rainfall = interpret_rainfall(50.0)
    explanation = explain_level_change(-10, rainfall)
    assert "anormal" in explanation


def test_explain_drop_with_little_rain_is_expected():
    rainfall = interpret_rainfall(2.0)
    explanation = explain_level_change(-10, rainfall)
    assert "cohérente" in explanation
    assert "anormal" not in explanation


def test_explain_rise_with_rain_is_coherent():
    rainfall = interpret_rainfall(60.0)
    explanation = explain_level_change(10, rainfall)
    assert "cohérente" in explanation


def test_no_explanation_for_small_change():
    rainfall = interpret_rainfall(60.0)
    assert explain_level_change(1.0, rainfall) is None


def test_no_explanation_without_prior_reading():
    rainfall = interpret_rainfall(60.0)
    assert explain_level_change(None, rainfall) is None
