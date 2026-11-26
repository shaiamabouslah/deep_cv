"""Turns the measurements available for a place into one status and the
reasons behind it, the same way the precision-agriculture dashboard turns
soil moisture, weather and vegetation into "irrigate / wait / monitor".

Levels follow the project's alert vocabulary: normal, vigilance, alerte,
critique. The thresholds are conventions chosen for this project, not
official drought or water-stress classes; they are shown with the result.

Inputs, all optional:
  surface_pct   water surface now, as a percentage of the place's historical
                maximum extent (Sentinel-2 vs JRC), a surface index
  fill_pct      official-style fill rate, only for dams with a calibration
  record        yearly water surface [{"year", "area_km2", "pct_of_max"}]
  climate       {"precip_90d_mm", "precip_normal_90d_mm", "anomaly_pct", ...}
"""

LEVELS = ["normal", "vigilance", "alerte", "critique"]
SURFACE_TIERS = [(70, "normal"), (50, "vigilance"), (30, "alerte")]
FILL_TIERS = [(60, "normal"), (40, "vigilance"), (25, "alerte")]
ACTIONS = {"normal": "Situation normale", "vigilance": "Surveiller", "alerte": "Alerte", "critique": "Situation critique"}
LONG_TERM_DROP_PCT = 20
SEVERE_DROP_PCT = 50
DRY_ANOMALY_PCT = 60


def level_from(value, tiers):
    for floor, level in tiers:
        if value >= floor:
            return level
    return "critique"


def escalate(level, steps=1):
    return LEVELS[min(LEVELS.index(level) + steps, len(LEVELS) - 1)]


def long_term_change(record, recent_years=5, min_years=15):
    """Mean surface of the last `recent_years` of the record against the mean
    of all earlier years, in percent; None if the record is too short."""
    if not record or len(record) < min_years:
        return None
    ordered = sorted(record, key=lambda r: r["year"])
    recent = [r["area_km2"] for r in ordered[-recent_years:]]
    earlier = [r["area_km2"] for r in ordered[:-recent_years]]
    if not earlier or sum(earlier) == 0:
        return None
    return round((sum(recent) / len(recent)) / (sum(earlier) / len(earlier)) * 100 - 100, 1)


def assess(surface_pct=None, fill_pct=None, record=None, climate=None):
    reasons = []
    change = long_term_change(record)
    anomaly = (climate or {}).get("anomaly_pct")
    dry = anomaly is not None and anomaly < DRY_ANOMALY_PCT
    declining = change is not None and change <= -LONG_TERM_DROP_PCT

    if fill_pct is not None:
        base = level_from(fill_pct, FILL_TIERS)
        reasons.append(f"Taux de remplissage estimé à {fill_pct:.0f} % (courbe calibrée sur les bulletins officiels).")
    elif surface_pct is not None:
        base = level_from(surface_pct, SURFACE_TIERS)
        reasons.append(f"Surface d'eau à {surface_pct:.0f} % de son maximum historique.")
    else:
        base = None

    if base is None:
        level = "vigilance" if (declining or dry) else "normal"
        if change is not None and change <= -SEVERE_DROP_PCT:
            level = "alerte"
        status = "contexte"
    else:
        level = base
        status = "mesure"
        if base == "normal" and (declining or dry):
            level = "vigilance"
    if declining:
        reasons.append(f"Sur le long terme, la surface des 5 dernières années est {abs(change):.0f} % sous la moyenne des années précédentes.")
    elif change is not None:
        reasons.append(f"Sur le long terme, la surface est stable ({change:+.0f} % par rapport aux années précédentes).")
    if anomaly is not None:
        word = "déficitaire" if dry else "proche de la normale" if anomaly < 130 else "excédentaire"
        reasons.append(f"Pluie des 90 derniers jours : {anomaly:.0f} % de la normale ({word}).")
    if climate and climate.get("balance_mm") is not None:
        reasons.append(f"Bilan pluie moins évaporation sur 90 jours : {climate['balance_mm']:+.0f} mm.")

    if status == "contexte":
        headline, action = {
            "normal": ("Aucun signal défavorable dans le relevé long terme ni dans le climat.", "Contexte normal"),
            "vigilance": ("Tendance défavorable sur le long terme ou pluie déficitaire : à suivre.", "Contexte à surveiller"),
            "alerte": ("Le plan d'eau a perdu plus de la moitié de sa surface sur le long terme.", "Contexte préoccupant"),
        }[level]
    else:
        action = ACTIONS[level]
        headline = {
            "normal": "Le plan d'eau est dans sa plage habituelle.",
            "vigilance": "Niveau sous la normale ou tendance défavorable : à suivre de près.",
            "alerte": "Niveau bas : risque pour les usages (irrigation, eau potable).",
            "critique": "Niveau très bas : situation critique.",
        }[level]
    return {"level": level, "action": action, "headline": headline, "reasons": reasons, "basis": status,
            "long_term_change_pct": change, "thresholds": {"surface_pct": [t for t, _ in SURFACE_TIERS],
                                                           "fill_pct": [t for t, _ in FILL_TIERS]}}
