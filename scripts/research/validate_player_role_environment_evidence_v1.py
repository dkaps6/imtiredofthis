"""Fail-closed validator for PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PATH = Path("docs/research/player_role_environment_evidence_v1.json")
VERSION = "PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1"
POSITIONS = {"QB","RB","WR","TE"}
FORBIDDEN = {"sportsbook","oddsapi","vegas_line","market_prob","book_line"}


def load_contract(path: Path = PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(d: dict) -> dict:
    errors=[]

    if d.get("contract_version") != VERSION:
        errors.append("contract version drift")
    if d.get("status") != "FROZEN_EVIDENCE_SCHEMA_SOURCE_ENGINEERING_ONLY":
        errors.append("status drift")
    if set(d.get("target_positions", [])) != POSITIONS:
        errors.append("target positions drift")

    serialized=json.dumps(d).lower()
    for token in FORBIDDEN:
        if token in serialized and token != "sportsbook":
            errors.append(f"forbidden market token present: {token}")

    rules=" ".join(d.get("global_rules",[])).lower()
    for phrase in [
        "before target kickoff",
        "sportsbook data is prohibited",
        "target-game partial usage is prohibited",
        "depth chart is contextual evidence only",
        "qualitative statements are structured evidence",
    ]:
        if phrase not in rules:
            errors.append(f"missing global rule: {phrase}")

    sources=d.get("source_registry",{})
    required_sources={
        "CURRENT_AVAILABILITY",
        "STRICT_PRIOR_PLAYER_HISTORY",
        "HISTORICAL_ROSTER_TEAM_HISTORY",
        "CURRENT_DEPTH_CONTEXT",
        "COACH_PLAYCALLER_HISTORY",
        "AUTHORITATIVE_ROLE_STATEMENT",
        "TRANSACTION_EVENT_HISTORY",
        "CURRENT_TRENCH_PERSONNEL",
    }
    missing_sources=sorted(required_sources-set(sources))
    if missing_sources:
        errors.append(f"missing sources {missing_sources}")

    fields=d.get("evidence_fields",[])
    if len(fields) < 40:
        errors.append(f"evidence field inventory unexpectedly small: {len(fields)}")
    names=[x.get("name") for x in fields]
    dup=sorted(k for k,v in Counter(names).items() if v>1)
    if dup:
        errors.append(f"duplicate fields {dup}")

    required_names={
        "current_availability_state",
        "eligible_for_opportunity",
        "official_depth_rank",
        "team_changed_since_prior_season",
        "credible_competitors_departed",
        "vacated_prior_opportunity_share",
        "strict_prior_target_share",
        "primary_qb_changed",
        "primary_play_caller_changed",
        "ol_returning_starter_count",
        "official_workload_expansion_signal",
        "role_evidence_coverage",
        "role_evidence_confidence",
    }
    if not required_names.issubset(set(names)):
        errors.append(f"required evidence fields missing {sorted(required_names-set(names))}")

    for x in fields:
        label=x.get("name","<unnamed>")
        if not x.get("family"):
            errors.append(f"{label}: family missing")
        if not x.get("type"):
            errors.append(f"{label}: type missing")
        src=x.get("source")
        if src!="DERIVED" and src not in sources:
            errors.append(f"{label}: unknown source {src}")
        if "positions" in x and not set(x["positions"]).issubset(POSITIONS):
            errors.append(f"{label}: bad position scope")

    qualitative=d.get("qualitative_statement_grain",{})
    qreq=set(qualitative.get("required_fields",[]))
    for k in {"source_reference","published_at_utc","available_before_kickoff","authority_tier","direct_vs_interpreted","confidence"}:
        if k not in qreq:
            errors.append(f"qualitative provenance missing {k}")
    concepts=set(qualitative.get("normalized_role_concepts",[]))
    for k in {"LEAD_ROLE","COMMITTEE_ROLE","WORKLOAD_EXPANSION","WORKLOAD_CONTRACTION","PRIMARY_TARGET","PASS_DOWN_ROLE","GOAL_LINE_ROLE"}:
        if k not in concepts:
            errors.append(f"role concept missing {k}")

    boundary=d.get("output_boundary",{})
    for k in ["direct_projection_adjustment_authorized","predictive_coefficients_authorized","production_integration_authorized","market_inputs_authorized"]:
        if boundary.get(k) is not False:
            errors.append(f"boundary weakened: {k}")

    if errors:
        raise ValueError("\n".join(errors))

    return {
        "disposition":"PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1_VALIDATED",
        "field_count":len(fields),
        "families":dict(sorted(Counter(x["family"] for x in fields).items())),
        "sources":len(sources),
        "production_integration_authorized":False,
        "predictive_coefficients_authorized":False,
    }


def main() -> None:
    print(json.dumps(validate_contract(load_contract()), indent=2, sort_keys=True))


if __name__=="__main__":
    main()
