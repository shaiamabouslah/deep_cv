import os

from src.agent import generate_briefing


def test_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sites = [{"name": "Sidi Salem", "latest_reading": {"water_area_pct": 60.0}, "alert_active": False}]
    result = generate_briefing(sites, [])
    assert result["ai_generated"] is False
    assert "Sidi Salem" not in result["text"] or "1 site" in result["text"]
    assert "1 site" in result["text"]


def test_rule_based_mentions_alerts(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    alerts = [{"site": "Nebhana", "drop_pct": 25.0, "water_area_pct": 40.0, "baseline_pct": 53.0}]
    result = generate_briefing([], alerts)
    assert "Nebhana" in result["text"]
    assert result["ai_generated"] is False
