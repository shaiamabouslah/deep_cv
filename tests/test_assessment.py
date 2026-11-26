from src import assessment


def _record(first, last, years=30):
    return [{"year": 1990 + i, "area_km2": first if i < years - 5 else last} for i in range(years)]


def test_level_follows_the_surface_tiers():
    assert assessment.assess(surface_pct=85)["level"] == "normal"
    assert assessment.assess(surface_pct=60)["level"] == "vigilance"
    assert assessment.assess(surface_pct=40)["level"] == "alerte"
    assert assessment.assess(surface_pct=12)["level"] == "critique"


def test_a_calibrated_fill_rate_takes_precedence_over_the_surface_index():
    result = assessment.assess(surface_pct=90, fill_pct=30)
    assert result["level"] == "alerte" and "Taux de remplissage" in result["reasons"][0]


def test_long_term_decline_is_measured_and_escalates_a_normal_level():
    record = _record(100, 60)
    assert assessment.long_term_change(record) == -40.0
    result = assessment.assess(surface_pct=80, record=record)
    assert result["level"] == "vigilance" and result["long_term_change_pct"] == -40.0
    assert any("sous la moyenne" in r for r in result["reasons"])


def test_a_stable_record_does_not_escalate():
    result = assessment.assess(surface_pct=80, record=_record(100, 98))
    assert result["level"] == "normal"


def test_a_short_record_gives_no_trend():
    assert assessment.long_term_change(_record(100, 60, years=10)) is None


def test_dry_rain_escalates_but_never_lowers_a_bad_level():
    dry = {"anomaly_pct": 40, "balance_mm": -120}
    assert assessment.assess(surface_pct=80, climate=dry)["level"] == "vigilance"
    assert assessment.assess(surface_pct=20, climate=dry)["level"] == "critique"


def test_without_a_measurement_the_result_is_context_not_a_status():
    calm = assessment.assess(climate={"anomaly_pct": 100})
    assert calm["basis"] == "contexte" and calm["level"] == "normal"
    worrying = assessment.assess(record=_record(100, 60), climate={"anomaly_pct": 100})
    assert worrying["basis"] == "contexte" and worrying["level"] == "vigilance"
    assert worrying["action"] == "Contexte à surveiller"


def test_a_collapsed_record_is_a_worrying_context_even_without_a_measurement():
    result = assessment.assess(record=_record(100, 30))
    assert result["level"] == "alerte" and result["action"] == "Contexte préoccupant"
