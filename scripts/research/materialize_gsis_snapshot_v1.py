#!/usr/bin/env python3
"""Validate and materialize immutable NFLGSIS browser-rendered snapshots.

This script never authenticates to GSIS and never handles browser session state.
It accepts the normalized JSON emitted by an already-authorized interactive
browser capture, validates scope/provenance, and writes a deterministic gzip
payload plus a reviewable manifest. Existing snapshot directories fail closed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


CURRENT_TEAMS = {
    "ARZ", "ATL", "BLT", "BUF", "CAR", "CHI", "CIN", "CLV",
    "DAL", "DEN", "DET", "GB", "HST", "IND", "JAX", "KC",
    "LAC", "LA", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS",
}

CURRENT_MODES = {
    "Lineup Detail": {"Offense", "Defense"},
    "Down Analysis": {
        "Offense / Detailed",
        "Defense / Detailed",
        "Offense / Grouped",
        "Defense / Grouped",
    },
    "Play Propensity": {"Field Position", "Quarter", "Score Differential"},
    "Formation Usage": {"default"},
}

DISPLAY_SCHEMAS = {
    "Lineup Detail": [
        "Lineup", "Plays", "Passing Plays", "Rushing Plays", "Avg Gain",
        "Avg Gain, Pass", "Avg Gain, Rush", "First Downs", "Touchdowns",
        "Fumbles Lost", "Interceptions",
    ],
    "Down Analysis / Detailed": [
        "Distance", "Plays", "Yards", "Avg", "1st Downs", "Conversion%",
    ],
    "Down Analysis / Grouped": [
        "Yards to Go", "Rushing Plays", "Avg Yards per Rush",
        "Rushing First Downs", "% Rushing Plays Converted", "Passing Plays",
        "Avg Yards per Pass", "Passing First Downs",
        "% Passing Plays Converted",
    ],
    "Play Propensity / Field Position": [
        "Yards to Go", "% Rushing Plays, All", "% Passing Plays, All",
        "% Rushing Plays, Red Zone", "% Passing Plays, Red Zone",
        "% Rushing Plays, Goal to Go", "% Passing Plays, Goal to Go",
        "% Rushing Plays, Own Side of Field",
        "% Passing Plays, Own Side of Field",
        "% Rushing Plays, Opp. Side of Field",
        "% Passing Plays, Opp. Side of Field",
    ],
    "Play Propensity / Quarter": [
        "Yards to Go", "% Rushing Plays, Q1", "% Passing Plays, Q1",
        "% Rushing Plays, Q2", "% Passing Plays, Q2",
        "% Rushing Plays, Q3", "% Passing Plays, Q3",
        "% Rushing Plays, Q4", "% Passing Plays, Q4",
        "% Rushing Plays, Overtime", "% Passing Plays, Overtime",
    ],
    "Play Propensity / Score Differential": [
        "Yards to Go", "% Rushing Plays, Winning by Two Scores",
        "% Passing Plays, Winning by Two Scores",
        "% Rushing Plays, Winning by One Score",
        "% Passing Plays, Winning by One Score",
        "% Rushing Plays, Game Tied", "% Passing Plays, Game Tied",
        "% Rushing Plays, Losing by One Score",
        "% Passing Plays, Losing by One Score",
        "% Rushing Plays, Losing by Two Scores",
        "% Passing Plays, Losing by Two Scores",
    ],
    "Formation Usage": [
        "Yards to Go", "# TEs", "# WRs", "Play Count", "Avg Gain",
        "Avg Gain, NFL Rank", "Rushing Plays", "Passing Plays", "Pass %",
        "Avg Gain per Rush", "Avg Gain per Rush, NFL Rank",
        "Avg Gain per Pass", "Avg Gain per Pass, NFL Rank",
    ],
}

SENSITIVE_KEY_FRAGMENTS = {
    "authorization", "cookie", "credential", "password", "secret",
    "session", "storage", "token",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def filter_map(record: dict) -> dict[str, str]:
    return {item["id"]: item["value"] for item in record.get("filters", [])}


def check_sensitive_keys(value: object, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS):
                raise ValueError(f"sensitive key rejected at {path}.{key}")
            check_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_sensitive_keys(child, f"{path}[{index}]")


def check_urls(value: object, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"url", "base_url"} and isinstance(child, str):
                parts = urlsplit(child)
                if parts.query or parts.username or parts.password:
                    raise ValueError(f"unsafe URL rejected at {path}.{key}")
            check_urls(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_urls(child, f"{path}[{index}]")


def validate_current(archive: dict) -> None:
    if archive.get("season") != 2026 or archive.get("phase") != "REG":
        raise ValueError("current archive must be 2026 REG")
    records = archive.get("records", [])
    expected = {
        (report, mode, team)
        for report, modes in CURRENT_MODES.items()
        for mode in modes
        for team in CURRENT_TEAMS
    }
    actual = set()
    for record in records:
        filters = filter_map(record)
        key = (record.get("report"), record.get("mode"), filters.get("select2"))
        if key in actual:
            raise ValueError(f"duplicate current record: {key}")
        actual.add(key)
        if filters.get("select0") != "2026" or filters.get("select1") != "Reg":
            raise ValueError(f"season/phase drift in {key}")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"current coverage mismatch; missing={missing}, extra={extra}")


def validate_pilot(archive: dict) -> None:
    if archive.get("season") != 2025 or archive.get("phase") != "REG":
        raise ValueError("pilot archive must be 2025 REG")
    records = archive.get("records", [])
    teams = {record.get("selected", {}).get("team") for record in records}
    if teams != {"ARZ", "CHI", "WAS"} or len(records) != 3:
        raise ValueError(f"historical pilot scope drifted: {sorted(teams)}")
    schemas = {
        tuple(cell["text"] for cell in record["tables"][0]["rows"][0]["cells"])
        for record in records
    }
    if schemas != {tuple(DISPLAY_SCHEMAS["Lineup Detail"])}:
        raise ValueError("historical Lineup Detail schema drifted")


def build_manifest(archive: dict, raw_bytes: bytes, compressed: bytes) -> dict:
    records = archive["records"]
    summaries = []
    for record in records:
        filters = filter_map(record)
        selected = record.get("selected", {})
        tables = record.get("tables", [])
        summaries.append({
            "report": record.get("report"),
            "mode": record.get("mode"),
            "team": filters.get("select2") or selected.get("team"),
            "team_label": next(
                (f.get("label") for f in record.get("filters", []) if f.get("id") == "select2"),
                selected.get("team_label"),
            ),
            "capture_timestamp_utc": record.get("capture_timestamp_utc"),
            "last_updated": record.get("last_updated"),
            "table_count": len(tables),
            "raw_row_count": sum(len(table.get("rows", [])) for table in tables),
            "availability": record.get("availability", "rendered" if tables else "empty"),
        })
    return {
        "archive_format_version": archive["archive_format_version"],
        "snapshot_id": archive["snapshot_id"],
        "created_at_utc": archive["created_at_utc"],
        "capture_window_utc": archive.get("capture_window_utc"),
        "season": archive["season"],
        "phase": archive["phase"],
        "source": archive["source"],
        "point_in_time_semantics": archive.get("point_in_time_semantics"),
        "research_boundary": archive.get("research_boundary"),
        "purpose": archive.get("purpose"),
        "leakage_warning": archive.get("leakage_warning"),
        "display_schemas": DISPLAY_SCHEMAS,
        "record_count": len(records),
        "records_by_report_mode": dict(sorted(Counter(
            f"{record.get('report')} | {record.get('mode')}" for record in records
        ).items())),
        "empty_or_alert_records": [
            item for item in summaries if item["availability"] != "rendered"
        ],
        "record_summaries": summaries,
        "raw_file": "raw_snapshot.json.gz",
        "raw_uncompressed_bytes": len(raw_bytes),
        "raw_uncompressed_sha256": sha256(raw_bytes),
        "raw_gzip_bytes": len(compressed),
        "raw_gzip_sha256": sha256(compressed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("current", "historical-pilot"), required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"immutable snapshot target already exists: {args.output}")
    archive = json.loads(args.input.read_text(encoding="utf-8"))
    check_sensitive_keys(archive)
    check_urls(archive)
    if args.kind == "current":
        validate_current(archive)
    else:
        validate_pilot(archive)

    raw_bytes = (json.dumps(archive, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    compressed = gzip.compress(raw_bytes, compresslevel=9, mtime=0)
    manifest = build_manifest(archive, raw_bytes, compressed)

    args.output.mkdir(parents=True)
    (args.output / "raw_snapshot.json.gz").write_bytes(compressed)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "snapshot_id": archive["snapshot_id"],
        "records": len(archive["records"]),
        "raw_gzip_sha256": manifest["raw_gzip_sha256"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
