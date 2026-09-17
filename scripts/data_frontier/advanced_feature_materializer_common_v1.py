"""Shared utilities for NFL Advanced Feature Materializer V1.

Full per-row derivatives remain ephemeral. Only aggregate QA/manifests are intended
for artifact upload. No predictive experiment or production integration occurs here.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

FEATURE_CONTRACT_VERSION = "NFL_ADVANCED_FEATURE_DICTIONARY_V1"
ALGORITHM_VERSION = "NFL_ADVANCED_FEATURE_MATERIALIZER_V1"
DICT_PATH = Path("docs/data_frontier/nfl_advanced_feature_dictionary_v1.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_sha256(root: Path) -> tuple[str, list[dict]]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        files.append(
            {
                "name": rel,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    material = "\n".join(f"{x['name']}:{x['sha256']}" for x in files)
    return hashlib.sha256(material.encode("utf-8")).hexdigest(), files


def load_dictionary(path: Path = DICT_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("feature_contract_version") != FEATURE_CONTRACT_VERSION:
        raise ValueError("advanced feature dictionary version drift")
    return data


def source_contract(source_key: str, dictionary: dict | None = None) -> tuple[dict, list[dict]]:
    d = dictionary or load_dictionary()
    registry = d["source_registry"]
    if source_key not in registry:
        raise KeyError(f"unknown source key {source_key}")
    fields = [x for x in d["fields"] if x["source_key"] == source_key]
    if not fields:
        raise ValueError(f"no contracted fields for {source_key}")
    return registry[source_key], fields


def safe_float_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    result = {
        "rows": int(len(series)),
        "non_null": int(valid.size),
        "nulls": int(s.isna().sum()),
        "null_rate": float(s.isna().mean()) if len(s) else None,
    }
    if valid.empty:
        result.update({"mean": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "min": None, "max": None})
        return result
    q = valid.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    result.update(
        {
            "mean": float(valid.mean()),
            "p10": float(q.loc[0.10]),
            "p25": float(q.loc[0.25]),
            "p50": float(q.loc[0.50]),
            "p75": float(q.loc[0.75]),
            "p90": float(q.loc[0.90]),
            "min": float(valid.min()),
            "max": float(valid.max()),
        }
    )
    return result


def safe_category_stats(series: pd.Series, max_values: int = 30) -> dict:
    s = series.astype("object")
    non_null = s.dropna()
    counts = non_null.astype(str).value_counts().head(max_values)
    return {
        "rows": int(len(series)),
        "non_null": int(non_null.size),
        "nulls": int(s.isna().sum()),
        "null_rate": float(s.isna().mean()) if len(s) else None,
        "value_counts": {str(k): int(v) for k, v in counts.items()},
        "value_counts_truncated": bool(non_null.astype(str).nunique() > max_values),
    }


def field_stats(series: pd.Series) -> dict:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return {"kind": "numeric", **safe_float_stats(series)}
    return {"kind": "categorical", **safe_category_stats(series)}


def write_private_table(df: pd.DataFrame, path: Path, sort_by: list[str] | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if sort_by:
        usable = [c for c in sort_by if c in out.columns]
        if usable:
            out = out.sort_values(usable, kind="mergesort", na_position="last").reset_index(drop=True)
    out.to_csv(path, index=False, float_format="%.10g", na_rep="")
    return {
        "file": path.name,
        "rows": int(len(out)),
        "columns": list(out.columns),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "uploaded": False,
    }


def assert_strict_prior(df: pd.DataFrame, target_week_col: str = "target_week", source_max_week_col: str = "history_max_source_week") -> dict:
    if df.empty:
        return {"rows_checked": 0, "violations": 0}
    t = pd.to_numeric(df[target_week_col], errors="coerce")
    s = pd.to_numeric(df[source_max_week_col], errors="coerce")
    comparable = t.notna() & s.notna()
    violations = int((s[comparable] >= t[comparable]).sum())
    if violations:
        raise ValueError(f"strict-prior temporal leakage detected: {violations} rows")
    return {"rows_checked": int(comparable.sum()), "violations": violations}


def sanitized_report(
    *,
    source_key: str,
    observed_source_hash: str,
    tables: Mapping[str, pd.DataFrame],
    private_manifests: Mapping[str, dict],
    temporal_audit: dict,
    structural_audit: dict,
    out_dir: Path,
) -> dict:
    dictionary = load_dictionary()
    source, contracted_fields = source_contract(source_key, dictionary)
    expected_hash = source["source_hash_sha256"]
    if observed_source_hash != expected_hash:
        raise ValueError(f"{source_key} source hash mismatch: {observed_source_hash} != {expected_hash}")

    expected_names = [x["field_name"] for x in contracted_fields]
    columns = set()
    for table in tables.values():
        columns.update(map(str, table.columns))
    missing = sorted(set(expected_names) - columns)
    if missing:
        raise ValueError(f"{source_key} contracted fields not materialized: {missing}")

    stats = {}
    field_contract_index = {x["field_name"]: x for x in contracted_fields}
    for field_name in expected_names:
        holders = [(name, df) for name, df in tables.items() if field_name in df.columns]
        if not holders:
            continue
        # A contracted field is expected in one semantic table. If support/debug tables
        # repeat it, concatenate only for aggregate QA and preserve table names.
        combined = pd.concat([df[[field_name]] for _, df in holders], ignore_index=True)[field_name]
        stats[field_name] = {
            "tables": [name for name, _ in holders],
            "temporal_classification": field_contract_index[field_name]["temporal_availability"]["classification"],
            "production_status": field_contract_index[field_name]["production_status"],
            "stats": field_stats(combined),
        }

    report = {
        "materializer_version": ALGORITHM_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_key": source_key,
        "source_dataset": source["source_dataset"],
        "source_kaggle_slug": source["kaggle_slug"],
        "source_season": source["source_season"],
        "source_hash_expected": expected_hash,
        "source_hash_observed": observed_source_hash,
        "source_hash_verified": True,
        "source_access_status": source["source_access_status"],
        "raw_data_handling": source["raw_data_handling"],
        "raw_competition_files_uploaded": False,
        "derived_per_row_feature_files_uploaded": False,
        "predictive_experiment": False,
        "production_changed": False,
        "sportsbook_inputs_used": False,
        "exact_coverage_responsibility_claimed": False,
        "contracted_field_count": len(expected_names),
        "contracted_fields": expected_names,
        "materialized_field_count": len(expected_names) - len(missing),
        "missing_contracted_fields": missing,
        "private_ephemeral_table_manifests": dict(private_manifests),
        "temporal_audit": temporal_audit,
        "structural_audit": structural_audit,
        "field_qa": stats,
        "disposition": "NFL_ADVANCED_FEATURE_MATERIALIZATION_V1_SOURCE_PASS",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{source_key.lower()}_advanced_feature_qa_v1.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def print_summary(report: dict) -> None:
    print(
        json.dumps(
            {
                "disposition": report["disposition"],
                "source_key": report["source_key"],
                "contracted_field_count": report["contracted_field_count"],
                "materialized_field_count": report["materialized_field_count"],
                "source_hash_verified": report["source_hash_verified"],
                "temporal_audit": report["temporal_audit"],
                "private_ephemeral_table_manifests": report["private_ephemeral_table_manifests"],
                "predictive_experiment": report["predictive_experiment"],
                "production_changed": report["production_changed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
