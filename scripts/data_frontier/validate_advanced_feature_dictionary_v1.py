"""Fail-closed validator for NFL_ADVANCED_FEATURE_DICTIONARY_V1.

This validates the engineering contract only. It does not calculate features,
run predictive experiments, touch production science, or access sportsbook data.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DICT_PATH = Path("docs/data_frontier/nfl_advanced_feature_dictionary_v1.json")

VERSION = "NFL_ADVANCED_FEATURE_DICTIONARY_V1"
STATUS = "FROZEN_ENGINEERING_CONTRACT_RESEARCH_ONLY"
TEMPORAL = {
    "PREGAME_HISTORICAL_DERIVABLE",
    "TARGET_GAME_POST_KICKOFF",
    "RETROSPECTIVE_VALIDATION_ONLY",
}
EXPECTED_SOURCES = {
    "BDB2021_ROUTE_GEOMETRY": {
        "slug": "nfl-big-data-bowl-2021",
        "hash": "55de76561799514779f3fd64b57c02a52430596f4f104c2b6060f3df8fd9e1b4",
    },
    "BDB2023_PROTECTION_GEOMETRY": {
        "slug": "nfl-big-data-bowl-2023",
        "hash": "1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182",
    },
    "BDB2026_THROW_WINDOW": {
        "slug": "nfl-big-data-bowl-2026-analytics",
        "hash": "228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07",
    },
}
REQUIRED_FIELD_KEYS = {
    "field_name",
    "feature_family",
    "source_key",
    "source_dataset",
    "source_kaggle_slug",
    "source_season",
    "source_hash_sha256",
    "source_hash_scope",
    "source_access_status",
    "raw_data_handling",
    "source_license_lineage",
    "canonical_source_checkpoint",
    "canonical_source_workflow",
    "grain",
    "deterministic_definition",
    "units",
    "temporal_availability",
    "confidence_abstention",
    "provenance_required",
    "semantic_limitations",
    "production_status",
    "predictive_experiment_status",
}
REQUIRED_PROVENANCE = {
    "source_dataset",
    "source_kaggle_slug",
    "source_hash_sha256",
    "source_hash_scope",
    "feature_contract_version",
    "algorithm_version",
    "game_id",
    "play_id",
    "generated_at_utc",
}


def load_contract(path: Path = DICT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(contract: dict) -> dict:
    errors: list[str] = []

    if contract.get("feature_contract_version") != VERSION:
        errors.append("feature_contract_version drift")
    if contract.get("contract_status") != STATUS:
        errors.append("contract_status drift")
    if contract.get("parent_handoff_commit") != "66e9d5d41e7e4e6187ecc2ab317f8a6f2d52c748":
        errors.append("parent handoff commit drift")

    registry = contract.get("source_registry", {})
    for key, expected in EXPECTED_SOURCES.items():
        src = registry.get(key)
        if not isinstance(src, dict):
            errors.append(f"source registry missing {key}")
            continue
        if src.get("kaggle_slug") != expected["slug"]:
            errors.append(f"{key} slug drift")
        if src.get("source_hash_sha256") != expected["hash"]:
            errors.append(f"{key} source hash drift")

    rules = " ".join(contract.get("global_semantic_rules", [])).lower()
    if "nearest defender" not in rules or "coverage responsibility" not in rules:
        errors.append("nearest-defender semantic firewall missing")
    if "pff_nflidblockedplayer" not in rules or "universal primary blocker-rusher" not in rules:
        errors.append("blocked-player semantic firewall missing")
    if "sportsbook" not in rules:
        errors.append("sportsbook separation rule missing")

    fields = contract.get("fields", [])
    if not isinstance(fields, list) or not fields:
        errors.append("fields missing/empty")
        fields = []

    names = [f.get("field_name") for f in fields if isinstance(f, dict)]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate field names: {duplicates}")

    class_counts = Counter()
    source_counts = Counter()

    for idx, feature in enumerate(fields):
        label = feature.get("field_name", f"index={idx}") if isinstance(feature, dict) else f"index={idx}"
        if not isinstance(feature, dict):
            errors.append(f"{label}: field record is not an object")
            continue

        missing = sorted(REQUIRED_FIELD_KEYS - set(feature))
        if missing:
            errors.append(f"{label}: missing keys {missing}")
            continue

        source_key = feature["source_key"]
        source_counts[source_key] += 1
        expected = EXPECTED_SOURCES.get(source_key)
        if expected is None:
            errors.append(f"{label}: unknown source_key {source_key}")
        else:
            if feature.get("source_kaggle_slug") != expected["slug"]:
                errors.append(f"{label}: source slug does not match registry")
            if feature.get("source_hash_sha256") != expected["hash"]:
                errors.append(f"{label}: source hash does not match registry")

        temporal = feature.get("temporal_availability", {})
        classification = temporal.get("classification")
        class_counts[classification] += 1
        if classification not in TEMPORAL:
            errors.append(f"{label}: invalid temporal classification {classification!r}")

        allowed = temporal.get("target_game_pregame_allowed")
        if classification == "PREGAME_HISTORICAL_DERIVABLE":
            if allowed is not True:
                errors.append(f"{label}: pregame historical field not explicitly allowed")
            if not feature["field_name"].startswith("hist_"):
                errors.append(f"{label}: pregame field must use hist_ prefix in V1")
            guard = temporal.get("leakage_guard", "").lower()
            if "target game excluded" not in guard:
                errors.append(f"{label}: pregame leakage guard does not explicitly exclude target game")
        elif allowed is not False:
            errors.append(f"{label}: non-pregame field incorrectly allowed pregame")

        # Explicitly keep future/outcome geometry out of V1 pregame fields.
        if classification == "PREGAME_HISTORICAL_DERIVABLE":
            lowered = feature["field_name"].lower()
            forbidden = ("postrelease", "terminal_", "_to_land", "landing_zone")
            if any(token in lowered for token in forbidden):
                errors.append(f"{label}: retrospective/outcome geometry promoted to V1 pregame field")

        conf = feature.get("confidence_abstention", {})
        if not isinstance(conf.get("abstain_if"), list) or not conf.get("abstain_if"):
            errors.append(f"{label}: abstention logic missing")
        if not isinstance(conf.get("required_quality_flags"), list) or not conf.get("required_quality_flags"):
            errors.append(f"{label}: quality/provenance flags missing")

        provenance = set(feature.get("provenance_required", []))
        missing_prov = sorted(REQUIRED_PROVENANCE - provenance)
        if missing_prov:
            errors.append(f"{label}: required provenance missing {missing_prov}")

        limitations = " ".join(feature.get("semantic_limitations", [])).lower()
        if source_key in {"BDB2021_ROUTE_GEOMETRY", "BDB2026_THROW_WINDOW"}:
            if "responsibility" not in limitations and "assignment" not in limitations:
                errors.append(f"{label}: proximity/assignment limitation missing")
        if source_key == "BDB2023_PROTECTION_GEOMETRY":
            # Provenance/support flags can be source-label fields; all must still avoid universal assignment claims.
            if "assignment" not in limitations and "interaction" not in limitations and "responsibility" not in limitations:
                errors.append(f"{label}: blocking interaction semantic limitation missing")

        if feature.get("source_access_status") != "ACCESSIBLE_REAL_FILES_AUTHENTICATED_KAGGLE":
            errors.append(f"{label}: source access status drift")
        if feature.get("raw_data_handling") != "EPHEMERAL_GITHUB_ACTIONS_ONLY_NOT_COMMITTED":
            errors.append(f"{label}: raw-data handling policy weakened")
        rights = feature.get("source_license_lineage", "").lower()
        if "does not imply" not in rights or "production" not in rights:
            errors.append(f"{label}: source/license lineage is not explicit enough")

        if feature.get("production_status") != "RESEARCH_ONLY_NOT_PROMOTED":
            errors.append(f"{label}: production status weakened")
        if feature.get("predictive_experiment_status") != "NOT_AUTHORIZED_BY_THIS_CONTRACT":
            errors.append(f"{label}: predictive experiment guard weakened")

        serialized = json.dumps(feature).lower()
        if "sportsbook_inputs" in serialized or "odds_api" in serialized:
            errors.append(f"{label}: sportsbook implementation field leaked into data-frontier contract")

    if not class_counts["PREGAME_HISTORICAL_DERIVABLE"]:
        errors.append("no pregame historical fields defined")
    if not class_counts["TARGET_GAME_POST_KICKOFF"]:
        errors.append("no target-game post-kickoff fields defined")
    if not class_counts["RETROSPECTIVE_VALIDATION_ONLY"]:
        errors.append("no retrospective validation fields defined")

    if errors:
        raise ValueError("\n".join(errors))

    return {
        "disposition": "NFL_ADVANCED_FEATURE_DICTIONARY_V1_VALIDATED",
        "field_count": len(fields),
        "temporal_class_counts": dict(sorted(class_counts.items())),
        "source_field_counts": dict(sorted(source_counts.items())),
        "source_hashes_verified": len(EXPECTED_SOURCES),
        "predictive_experiments_authorized": False,
        "production_changes_authorized": False,
    }


def main() -> None:
    result = validate_contract(load_contract())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
