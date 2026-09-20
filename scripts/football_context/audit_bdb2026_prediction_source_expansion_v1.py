#!/usr/bin/env python3
"""Sanitized source audit for NFL Big Data Bowl 2026 Prediction vs Analytics corpora.

This audit is intentionally outcome-free. It inventories archive structure, historical
input/output seasons and weeks, schema, hashes, and whether the Prediction competition
materially expands the historical tracking available to the already-certified Analytics
2023 source. Raw competition data are never written to repository artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

HIST_RE = re.compile(r"^(input|output)_(\d{4})_w(\d{2})\.csv$", re.I)
CERTIFIED_ANALYTICS_CORPUS_SHA256 = "228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corpus_sha256(root: Path) -> tuple[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest = sha256(path)
        size = path.stat().st_size
        rows.append({"relative_path": rel, "bytes": size, "sha256": digest})
        h.update(rel.encode())
        h.update(b"\0")
        h.update(str(size).encode())
        h.update(b"\0")
        h.update(digest.encode())
        h.update(b"\n")
    return h.hexdigest(), rows


def csv_columns(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def line_rows(path: Path) -> int:
    with path.open("rb") as fh:
        count = sum(block.count(b"\n") for block in iter(lambda: fh.read(1 << 20), b""))
    return max(0, count - 1)


def inventory(root: Path, label: str) -> tuple[pd.DataFrame, dict[str, object]]:
    corpus_hash, manifest = corpus_sha256(root)
    by_rel = {r["relative_path"]: r for r in manifest}
    rows: list[dict[str, object]] = []
    seasons: set[int] = set()
    input_weeks: dict[int, set[int]] = {}
    output_weeks: dict[int, set[int]] = {}

    for rel, rec in sorted(by_rel.items()):
        path = root / rel
        name = path.name
        match = HIST_RE.match(name)
        kind = None
        season = None
        week = None
        historical_regular_season_file = False
        if match:
            kind = match.group(1).lower()
            season = int(match.group(2))
            week = int(match.group(3))
            historical_regular_season_file = 1 <= week <= 18
            seasons.add(season)
            target = input_weeks if kind == "input" else output_weeks
            target.setdefault(season, set()).add(week)

        cols = csv_columns(path) if path.suffix.lower() == ".csv" else []
        rows.append(
            {
                "corpus": label,
                "relative_path": rel,
                "basename": name,
                "bytes": int(rec["bytes"]),
                "sha256": str(rec["sha256"]),
                "kind": kind or "",
                "season": season,
                "week": week,
                "historical_regular_season_file": historical_regular_season_file,
                "row_count": line_rows(path) if historical_regular_season_file else None,
                "column_count": len(cols),
                "columns": "|".join(cols),
            }
        )

    summary = {
        "corpus": label,
        "corpus_sha256": corpus_hash,
        "file_count": len(manifest),
        "historical_seasons": sorted(seasons),
        "input_weeks_by_season": {str(k): sorted(v) for k, v in sorted(input_weeks.items())},
        "output_weeks_by_season": {str(k): sorted(v) for k, v in sorted(output_weeks.items())},
    }
    return pd.DataFrame(rows), summary


def compare(analytics: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, object]:
    ah = analytics.loc[analytics["historical_regular_season_file"].eq(True)].copy()
    ph = prediction.loc[prediction["historical_regular_season_file"].eq(True)].copy()

    a_inputs = ah.loc[ah["kind"].eq("input")].copy()
    p_inputs = ph.loc[ph["kind"].eq("input")].copy()

    a_by_base = a_inputs.drop_duplicates("basename").set_index("basename")
    p_by_base = p_inputs.drop_duplicates("basename").set_index("basename")
    common = sorted(set(a_by_base.index) & set(p_by_base.index))
    identical = [
        name for name in common
        if str(a_by_base.loc[name, "sha256"]) == str(p_by_base.loc[name, "sha256"])
    ]
    differing = sorted(set(common) - set(identical))
    unique_prediction = sorted(set(p_by_base.index) - set(a_by_base.index))
    prediction_seasons = sorted(
        int(x) for x in p_inputs["season"].dropna().astype(int).unique().tolist()
    )
    extra_seasons = [s for s in prediction_seasons if s != 2023]

    material_expansion = bool(extra_seasons or unique_prediction)
    if material_expansion:
        disposition = "MATERIAL_HISTORICAL_TRACKING_EXPANSION_AVAILABLE"
    else:
        disposition = "NO_MATERIAL_HISTORICAL_TRACKING_EXPANSION"

    required_input_cols = {
        "game_id", "play_id", "nfl_id", "frame_id", "player_role",
        "x", "y", "s", "a", "o", "dir",
    }
    p_input_cols: set[str] = set()
    for colstr in p_inputs["columns"].dropna().astype(str):
        p_input_cols.update(c for c in colstr.split("|") if c)

    return {
        "common_historical_input_files": len(common),
        "byte_identical_common_historical_input_files": len(identical),
        "differing_common_historical_input_files": differing,
        "prediction_unique_historical_input_files": unique_prediction,
        "prediction_historical_input_seasons": prediction_seasons,
        "prediction_additional_historical_seasons_vs_certified_2023": extra_seasons,
        "prediction_has_required_tracking_schema": required_input_cols.issubset(p_input_cols),
        "materially_expands_certified_analytics_history": material_expansion,
        "disposition": disposition,
        "sportsbook_read": False,
        "football_outcomes_used_for_predictive_scoring": False,
        "production_change_authorized": False,
        "issue_535_touched": False,
    }


def run(analytics_dir: Path, prediction_dir: Path, out_dir: Path) -> dict[str, object]:
    a, a_summary = inventory(analytics_dir, "analytics")
    p, p_summary = inventory(prediction_dir, "prediction")

    if a_summary["corpus_sha256"] != CERTIFIED_ANALYTICS_CORPUS_SHA256:
        raise RuntimeError(
            "analytics corpus hash differs from certified BDB2026 source: "
            f"{a_summary['corpus_sha256']} != {CERTIFIED_ANALYTICS_CORPUS_SHA256}"
        )

    cmp = compare(a, p)
    report = {
        "audit": "BDB2026_PREDICTION_SOURCE_EXPANSION_AUDIT_V1",
        "certified_analytics_corpus_sha256": CERTIFIED_ANALYTICS_CORPUS_SHA256,
        "analytics": a_summary,
        "prediction": p_summary,
        "comparison": cmp,
        "rb_research_note": (
            "RB predictive research remains unresolved and pinned; this source-only "
            "tracking audit does not claim to solve or advance the RB model."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([a, p], ignore_index=True).to_csv(
        out_dir / "bdb2026_prediction_source_inventory_v1.csv", index=False
    )
    (out_dir / "bdb2026_prediction_source_expansion_audit_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytics-dir", required=True, type=Path)
    ap.add_argument("--prediction-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    report = run(args.analytics_dir, args.prediction_dir, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
