#!/usr/bin/env python3
"""Fail closed on malformed verified manual player-name aliases."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.utils.canonical_names import build_manual_map, canonicalize_player_name_safe, norm_key

PATH = Path("data/manual_name_overrides.csv")
AUDIT = Path("data/manual_name_overrides_audit.json")


def validate() -> dict:
    if not PATH.exists() or PATH.stat().st_size <= 0:
        result = {"disposition": "NO_MANUAL_NAME_OVERRIDES", "rows": 0}
        AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    df = pd.read_csv(PATH)
    required = {"player_source_name", "full_name", "reason", "verified_source", "verified_date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"manual name overrides missing columns: {sorted(missing)}")
    if not isinstance(df.index, pd.RangeIndex):
        raise RuntimeError(
            "manual name overrides parsed with a non-default index; this usually means an unquoted extra comma shifted a row"
        )
    if df.empty:
        raise RuntimeError("manual name overrides exists but has zero rows")
    for col in required:
        blank = df[col].astype("string").fillna("").str.strip().eq("")
        if blank.any():
            raise RuntimeError(f"manual name overrides contains blank {col}: rows={int(blank.sum())}")

    keys = df["player_source_name"].astype(str).map(norm_key)
    if keys.eq("").any() or keys.duplicated().any():
        raise RuntimeError("manual name overrides contains blank or duplicate normalized source names")
    targets = df["full_name"].astype(str).map(norm_key)
    if targets.eq("").any():
        raise RuntimeError("manual name overrides contains blank normalized canonical names")

    # Clear the one-entry cache so this validator proves the file currently on
    # disk, not a mapping cached before artifact restoration.
    build_manual_map.cache_clear()
    loaded = build_manual_map()
    failures = []
    for row in df.itertuples(index=False):
        source = str(row.player_source_name).strip()
        expected = str(row.full_name).strip()
        actual, _ = canonicalize_player_name_safe(source)
        if actual != expected:
            failures.append({"source": source, "expected": expected, "actual": actual})
    if failures:
        raise RuntimeError(f"manual name overrides do not round-trip through canonicalization: {failures[:20]}")

    result = {
        "disposition": "VERIFIED_MANUAL_NAME_OVERRIDES_READY",
        "rows": int(len(df)),
        "normalized_source_keys": int(keys.nunique()),
        "round_trip_failures": 0,
    }
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[manual_name_overrides] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
