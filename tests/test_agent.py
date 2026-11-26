from src import agent
from src.agent import generate_briefing, generate_fused_recommendation


def _clear_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_falls_back_without_api_key(monkeypatch):
    _clear_groq_key(monkeypatch)
    sites = [{"name": "Sidi Salem", "latest_reading": {"water_area_pct": 60.0}, "alert_level": None}]
    result = generate_briefing(sites, [])
    assert result["ai_generated"] is False
    assert "1 site" in result["text"]


def test_rule_based_mentions_alerts(monkeypatch):
    _clear_groq_key(monkeypatch)
    alerts = [{"site": "Nebhana", "drop_pct": 25.0, "water_area_pct": 40.0, "baseline_pct": 53.0, "level": "alerte"}]
    result = generate_briefing([], alerts)
    assert "Nebhana" in result["text"]
    assert result["ai_generated"] is False


def test_fused_recommendation_falls_back_without_api_key(monkeypatch):
    _clear_groq_key(monkeypatch)
    signals = {
        "site": "Sidi Salem",
        "water_area_pct": 45.0,
        "source": "upload",
        "rgb_quality": {"usage": {"label": "irrigation possible", "detail": "eau trouble"}},
        "gee_indices": None,
        "rainfall": None,
        "water_balance": None,
        "alert": None,
    }
    result = generate_fused_recommendation(signals)
    assert result["ai_generated"] is False
    assert "45.0%" in result["text"]
    assert "irrigation possible" in result["text"]


def test_fused_recommendation_includes_gee_and_alert_in_fallback(monkeypatch):
    _clear_groq_key(monkeypatch)
    signals = {
        "site": "Nebhana",
        "water_area_pct": 30.0,
        "source": "satellite",
        "rgb_quality": None,
        "gee_indices": {"ndvi": 0.2, "ndci": 0.1, "ndti": 0.08},
        "rainfall": {"note": "Pluviométrie faible : 0.0 mm."},
        "water_balance": "Offre et demande baissent ensemble.",
        "alert": {"level": "critique", "drop_pct": 40.0},
    }
    result = generate_fused_recommendation(signals)
    assert "NDVI=0.2" in result["text"]
    assert "critique" in result["text"]
    assert "Offre et demande" in result["text"]


FILL = {"fill_pct": 41.3, "typical_error_pts": 1.6, "n_points": 61, "period": ["2024-01-03", "2025-10-08"]}


def test_rule_based_fusion_states_the_calibrated_fill_rate_with_its_error():
    text = agent._rule_based_fusion({"water_area_pct": 70.0, "source": "satellite", "fill_estimate": FILL})
    assert "41%" in text and "1.6" in text and "61 bulletins" in text


def test_llm_context_carries_the_fill_rate_so_the_answer_is_grounded_in_it():
    context = agent._format_fusion_context({"site": "Sidi Salem", "water_area_pct": 70.0, "source": "satellite", "fill_estimate": FILL})
    assert "Taux de remplissage estimé : 41%" in context
