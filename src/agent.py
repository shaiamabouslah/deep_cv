"""AI agent layer, powered by Groq (fast open-model inference) via its
dedicated Python SDK — not to be confused with Grok (xAI); different
companies, similarly named.

Two roles:
  - generate_briefing: synthesizes every monitored site's status into a
    short, plain-language situation report (the kind a human analyst would
    write for decision-makers).
  - generate_fused_recommendation: combines ALL signals available for a
    single analysis (deep-learning segmentation result, RGB color quality
    heuristic, real GEE spectral indices when the site's coordinates are
    known, rainfall, water-balance diagnostic) into ONE coherent,
    LLM-written usability recommendation — replacing the separate
    hand-written template strings with a genuine fusion step.

Requires GROQ_API_KEY in the environment. Without it, both functions fall
back to a deterministic, rule-based summary built from the same inputs, so
the rest of the app stays fully usable.
"""
import os

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def _groq_api_key():
    return os.environ.get("GROQ_API_KEY")


def _groq_client():
    from groq import Groq
    return Groq(api_key=_groq_api_key())


def _ask_groq(prompt, max_tokens=400):
    client = _groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Multi-site briefing
# ---------------------------------------------------------------------------

def _rule_based_briefing(sites, alerts):
    if not sites and not alerts:
        return "Aucun site surveillé pour le moment."

    lines = [f"{len(sites)} site(s) surveillé(s), {len(alerts)} alerte(s) active(s)."]
    for alert in alerts[:5]:
        lines.append(
            f"- {alert['site']} : baisse de {alert['drop_pct']:.1f}% par rapport à la moyenne "
            f"historique ({alert['water_area_pct']:.1f}% actuel vs {alert['baseline_pct']:.1f}% "
            f"habituel), niveau {alert.get('level', 'inconnu')} — à surveiller en priorité."
        )
    if not alerts:
        lines.append("Aucune baisse anormale détectée sur les sites suivis.")
    return "\n".join(lines)


def _format_sites_context(sites, alerts):
    parts = []
    for site in sites:
        reading = site.get("latest_reading")
        status = f"{reading['water_area_pct']:.1f}% de surface d'eau" if reading else "pas encore de donnée"
        flag = f" [ALERTE {site['alert_level'].upper()}]" if site.get("alert_level") else ""
        parts.append(f"- {site['name']}: {status}{flag}")
    return "\n".join(parts) if parts else "Aucun site."


def generate_briefing(sites, alerts):
    if not _groq_api_key():
        return {
            "text": _rule_based_briefing(sites, alerts),
            "ai_generated": False,
            "note": "GROQ_API_KEY non configurée — résumé généré par règles simples, "
                    "pas par un agent IA.",
        }

    try:
        context = _format_sites_context(sites, alerts)
        text = _ask_groq(
            "Tu es un analyste des ressources hydriques en Tunisie, pays en stress hydrique "
            "chronique (le contexte est la RARETÉ de l'eau, pas les inondations). Voici l'état "
            "actuel des sites surveillés (barrages/plans d'eau) :\n\n" + context + "\n\n"
            "Rédige un briefing court (5-8 lignes max) en français, clair, actionnable et "
            "TERMINÉ (ne coupe pas la dernière phrase), pour un décideur : situation globale, "
            "sites préoccupants en priorité, et une recommandation concrète si une alerte est "
            "active. Pas de formules creuses.",
            max_tokens=600,
        )
        return {"text": text, "ai_generated": True, "note": None}
    except Exception as exc:
        return {
            "text": _rule_based_briefing(sites, alerts),
            "ai_generated": False,
            "note": f"Agent IA indisponible ({exc}), résumé de secours affiché.",
        }


# ---------------------------------------------------------------------------
# Single-analysis fusion (segmentation + RGB quality + GEE + rainfall + balance)
# ---------------------------------------------------------------------------

def _rule_based_fusion(signals):
    lines = [f"Surface d'eau estimée : {signals['water_area_pct']:.1f}% (source : {signals['source']})."]

    rgb = signals.get("rgb_quality")
    if rgb and rgb.get("usage"):
        lines.append(f"Analyse couleur (photo) : {rgb['usage']['label']} — {rgb['usage']['detail']}")

    gee = signals.get("gee_indices")
    if gee:
        lines.append(
            f"Indices satellite réels : NDVI={gee.get('ndvi')}, NDCI={gee.get('ndci')}, "
            f"NDTI={gee.get('ndti')}."
        )

    rainfall = signals.get("rainfall")
    if rainfall:
        lines.append(rainfall.get("note", ""))

    if signals.get("water_balance"):
        lines.append(f"Bilan hydrique : {signals['water_balance']}")

    if signals.get("alert"):
        lines.append(f"Alerte de niveau {signals['alert']['level']} active sur ce site.")

    if signals.get("fill_estimate"):
        lines.append(_fill_sentence(signals["fill_estimate"]))

    return "\n".join(line for line in lines if line)


def _format_fusion_context(signals):
    parts = [
        f"Site : {signals.get('site', 'inconnu')}",
        f"Surface d'eau estimée : {signals['water_area_pct']:.1f}% (source : {signals['source']})",
    ]
    rgb = signals.get("rgb_quality")
    if rgb and rgb.get("usage"):
        parts.append(f"Analyse couleur de la photo uploadée : {rgb['usage']['label']} ({rgb['usage']['detail']})")
    gee = signals.get("gee_indices")
    if gee:
        parts.append(f"Indices spectraux réels (Sentinel-2) : NDVI={gee.get('ndvi')}, NDCI={gee.get('ndci')}, NDTI={gee.get('ndti')}")
    if signals.get("rainfall"):
        parts.append(signals["rainfall"].get("note", ""))
    if signals.get("water_balance"):
        parts.append(f"Bilan hydrique offre/demande : {signals['water_balance']}")
    if signals.get("alert"):
        parts.append(f"Alerte active, niveau {signals['alert']['level']} (baisse de {signals['alert']['drop_pct']:.1f}%)")
    fill = signals.get("fill_estimate")
    if fill:
        parts.append(_fill_sentence(fill))
    return "\n".join(parts)


def _fill_sentence(fill):
    return (
        f"Taux de remplissage estimé : {fill['fill_pct']:.0f}% de la capacité (calibré sur "
        f"{fill['n_points']} bulletins officiels ONAGRI, erreur typique ± {fill['typical_error_pts']} points)."
    )


def generate_fused_recommendation(signals):
    """`signals` is a dict gathering everything known about one analysis:
    water_area_pct, source ("upload"/"satellite"), rgb_quality (from
    water_quality.analyze, if from an uploaded photo), gee_indices (dict
    with ndvi/ndci/ndti, if the site's coordinates let us fetch them),
    rainfall, water_balance, alert, fill_estimate (calibrated fill rate, only
    for dams with a validated calibration). Any of these may be None/missing."""
    if not _groq_api_key():
        return {
            "text": _rule_based_fusion(signals),
            "ai_generated": False,
            "note": "GROQ_API_KEY non configurée — diagnostic généré par règles simples.",
        }

    try:
        context = _format_fusion_context(signals)
        text = _ask_groq(
            "Tu es un conseiller en gestion de l'eau pour l'irrigation et la ressource hydrique "
            "en Tunisie. Voici toutes les données disponibles pour un plan d'eau (barrage) :\n\n"
            + context + "\n\n"
            "À partir de TOUTES ces données combinées, donne en français (6-10 lignes max) : "
            "1) l'état de la ressource (taux de remplissage s'il est donné, tendance, alerte), "
            "2) ce que les indices spectraux suggèrent de la qualité APPARENTE de l'eau pour "
            "l'irrigation, en précisant que c'est indicatif, 3) une recommandation concrète. "
            "RÈGLES : ne dis jamais qu'une eau est potable ou propre à la consommation, ces "
            "données satellite ne permettent pas de le savoir ; recommande des analyses de "
            "laboratoire pour toute décision de qualité. Sois direct, pas de formules creuses, "
            "et signale explicitement si les données sont insuffisantes pour trancher.",
            max_tokens=600,
        )
        return {"text": text, "ai_generated": True, "note": None}
    except Exception as exc:
        return {
            "text": _rule_based_fusion(signals),
            "ai_generated": False,
            "note": f"Agent IA indisponible ({exc}), diagnostic de secours affiché.",
        }
