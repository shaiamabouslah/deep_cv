import datetime

from src import place_report


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_geocode_returns_the_fields_the_interface_needs(monkeypatch):
    payload = {"results": [{"name": "Las Vegas", "admin1": "Nevada", "country": "Etats-Unis", "latitude": 36.1, "longitude": -115.1, "id": 1}]}
    monkeypatch.setattr(place_report.httpx, "get", lambda url, params=None, timeout=None: _Response(payload))
    assert place_report.geocode("las vegas") == [
        {"name": "Las Vegas", "admin1": "Nevada", "country": "Etats-Unis", "lat": 36.1, "lon": -115.1}]


def test_geocode_without_results_is_an_empty_list(monkeypatch):
    monkeypatch.setattr(place_report.httpx, "get", lambda url, params=None, timeout=None: _Response({}))
    assert place_report.geocode("zzzz") == []


def test_climate_compares_the_last_90_days_with_the_same_window_of_earlier_years(monkeypatch):
    today = datetime.date(2026, 7, 1)
    days = [(today - datetime.timedelta(days=90 - i)).isoformat() for i in range(97)]
    forecast = {"daily": {"time": days, "precipitation_sum": [1.0] * 90 + [2.0] * 7, "et0_fao_evapotranspiration": [4.0] * 97}}
    archive_days, archive_rain = [], []
    for year in range(2021, 2026):
        day = datetime.date(year, 1, 1)
        while day.year == year:
            archive_days.append(day.isoformat())
            archive_rain.append(2.0)
            day += datetime.timedelta(days=1)

    def fake_get(url, params=None, timeout=None):
        return _Response(forecast if "archive" not in url else {"daily": {"time": archive_days, "precipitation_sum": archive_rain}})

    monkeypatch.setattr(place_report.httpx, "get", fake_get)
    result = place_report.climate_summary(36.0, 9.0, today=today)

    assert result["precip_90d_mm"] == 90.0 and result["et0_90d_mm"] == 360.0
    assert result["balance_mm"] == -270.0 and result["rain_next_7d_mm"] == 14.0
    assert result["normal_years"] == 5 and abs(result["precip_normal_90d_mm"] - 2.0 * 91) < 4
    assert result["anomaly_pct"] == round(90.0 / result["precip_normal_90d_mm"] * 100)
