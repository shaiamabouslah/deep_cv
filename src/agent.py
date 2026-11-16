"""AI agent layer: synthesizes the raw metrics (water levels, NDVI/NDCI/NDTI,
active alerts) across every monitored site into a short, plain-language
briefing — the kind a human analyst would write for decision-makers.

Uses the Claude API when ANTHROPIC_API_KEY is set; otherwise falls back to a
simple templated summary, so the rest of the app stays fully usable without
requiring that extra credential.
"""
import os


def _rule_based_briefing(sites, alerts):
    if not sites and not alerts:
        return "Aucun site surveillé pour le moment."

    lines = [f"{len(sites)} site(s) surveillé(s), {len(alerts)} alerte(s) active(s)."]
    for alert in alerts[:5]:
        lines.append(
            f"- {alert['site']} : baisse de {alert['drop_pct']:.1f}% par rapport à la moyenne "
            f"historique ({alert['water_area_pct']:.1f}% actuel vs {alert['baseline_pct']:.1f}% "
            "habituel) — à surveiller en priorité."
        )
    if not alerts:
        lines.append("Aucune baisse anormale détectée sur les sites suivis.")
    return "\n".join(lines)


def _format_context(sites, alerts):
    parts = []
    for site in sites:
        reading = site.get("latest_reading")
        status = f"{reading['water_area_pct']:.1f}% de surface d'eau" if reading else "pas encore de donnée"
        flag = " [ALERTE ACTIVE]" if site.get("alert_active") else ""
        parts.append(f"- {site['name']}: {status}{flag}")
    return "\n".join(parts) if parts else "Aucun site."


def generate_briefing(sites, alerts):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "text": _rule_based_briefing(sites, alerts),
            "ai_generated": False,
            "note": "ANTHROPIC_API_KEY non configurée — résumé généré par règles simples, "
                    "pas par un agent IA.",
        }

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        context = _format_context(sites, alerts)

        response = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    "Tu es un analyste des ressources hydriques en Tunisie. Voici l'état actuel "
                    "des sites surveillés (barrages/plans d'eau) :\n\n" + context + "\n\n"
                    "Rédige un briefing court (5-8 lignes max) en français, clair et actionnable, "
                    "pour un décideur : situation globale, sites préoccupants en priorité, et une "
                    "recommandation concrète si une alerte est active. Pas de formules creuses."
                ),
            }],
        )
        text = response.content[0].text
        return {"text": text, "ai_generated": True, "note": None}

    except Exception as exc:
        return {
            "text": _rule_based_briefing(sites, alerts),
            "ai_generated": False,
            "note": f"Agent IA indisponible ({exc}), résumé de secours affiché.",
        }
