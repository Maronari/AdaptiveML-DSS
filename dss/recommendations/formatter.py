from __future__ import annotations


def format_recommendation(rule: dict, scenario: list[str], facts: dict) -> dict:
    rationale = []
    if facts["strong_positive_factors"]:
        rationale.append(
            "Сильные положительные факторы: "
            + ", ".join(facts["strong_positive_factors"])
        )
    if facts["negative_factors"]:
        rationale.append("Сдерживающие факторы: " + ", ".join(facts["negative_factors"]))
    if not rationale:
        rationale.append("Явных доминирующих факторов не выявлено.")

    return {
        "risk_level": facts["risk_level"],
        "summary": rule["summary"],
        "actions": scenario,
        "rationale": rationale,
    }
