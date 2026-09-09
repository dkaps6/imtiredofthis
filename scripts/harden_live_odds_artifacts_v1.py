#!/usr/bin/env python3
"""Semantic hardening for live OddsAPI artifacts before modeling.

This is deliberately downstream of the provider fetch and upstream of PlayerForm.
It repairs only deterministic representation defects that preserve the underlying
offer values, and it fails closed on anything that is not demonstrably mechanical.

2026-09-07 production incident addressed here:
- player offers were multiplied by joining to every game-odds outcome row for the
  same event_id (18 rows/event on the paid Week-1 snapshot);
- the expanded rows propagated into data/props_raw.csv and enriched artifacts;
- grouped outputs/props_raw.csv contained each true book offer 18 times inside
  offers_json;
- blank event/market "no offer posted" sentinels leaked into the player-name map.

The provider source files outputs/props_<market>.csv are treated as the closest
source-of-truth. If they themselves contain exact duplicates, the run fails rather
than silently deduplicating them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

DATA = Path("data")
OUTPUTS = Path("outputs")

RAW_OFFER_ARTIFACTS = [
    DATA / "props_raw.csv",
    OUTPUTS / "props_enriched.csv",
    DATA / "props_enriched.csv",
]
NAME_MAP_ARTIFACTS = [
    OUTPUTS / "player_name_map_from_props.csv",
    DATA / "player_name_map_from_props.csv",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def validate_game_identity(path: Path = OUTPUTS / "odds_game.csv") -> int:
    """Require exactly one canonical home/away identity per event_id."""
    df = _read(path)
    if df.empty:
        return 0
    required = {"event_id", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"odds_game missing game identity columns: {sorted(missing)}")
    x = df[["event_id", "home_team", "away_team"]].copy()
    x["event_id"] = x["event_id"].astype("string").fillna("").str.strip()
    x["home_team"] = x["home_team"].map(canon_team)
    x["away_team"] = x["away_team"].map(canon_team)
    if x["event_id"].eq("").any() or x[["home_team", "away_team"]].isna().any().any():
        raise RuntimeError("odds_game contains blank/unresolved event or team identity")
    pairs = x.drop_duplicates()
    counts = pairs.groupby("event_id", dropna=False).size()
    bad = counts.loc[counts.ne(1)]
    if not bad.empty:
        sample = pairs.loc[pairs["event_id"].isin(bad.index)].head(20).to_dict("records")
        raise RuntimeError(
            "Conflicting home/away identity for OddsAPI event_id; "
            f"bad_events={bad.to_dict()} sample={sample}"
        )
    return int(pairs["event_id"].nunique())


def _dedupe_exact_rows(path: Path) -> dict[str, int]:
    df = _read(path)
    before = int(len(df))
    if df.empty:
        return {"before": before, "after": before, "removed": 0}
    clean = df.drop_duplicates().copy()
    removed = before - int(len(clean))
    clean.to_csv(path, index=False)
    if clean.duplicated().any():
        raise RuntimeError(f"Exact duplicates remain after mechanical repair: {path}")
    return {"before": before, "after": int(len(clean)), "removed": int(removed)}


def _normalize_json_value(value: object) -> object:
    """Recursively convert pandas/JSON NaN-like values to JSON null."""
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _json_key(value: object) -> str:
    return json.dumps(
        _normalize_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _dedupe_grouped_offers(path: Path = OUTPUTS / "props_raw.csv") -> dict[str, int]:
    df = _read(path)
    if df.empty or "offers_json" not in df.columns:
        return {"rows": int(len(df)), "entries_before": 0, "entries_after": 0, "removed": 0}
    before_total = 0
    after_total = 0
    encoded: list[str] = []
    for raw in df["offers_json"].fillna("[]"):
        try:
            offers = json.loads(raw) if str(raw).strip() else []
        except Exception as exc:
            raise RuntimeError(f"Malformed offers_json in {path}: {exc}") from exc
        if not isinstance(offers, list):
            raise RuntimeError(f"offers_json must be a list in {path}")
        before_total += len(offers)
        seen: set[str] = set()
        unique: list[object] = []
        for offer in offers:
            normalized = _normalize_json_value(offer)
            key = _json_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        after_total += len(unique)
        encoded.append(json.dumps(unique, allow_nan=False))
    df["offers_json"] = encoded
    df.to_csv(path, index=False)
    return {
        "rows": int(len(df)),
        "entries_before": int(before_total),
        "entries_after": int(after_total),
        "removed": int(before_total - after_total),
    }


def _sanitize_name_maps() -> dict[str, int]:
    """Remove blank no-market sentinels; fail on unresolved real player names."""
    blank_removed = 0
    unresolved_real = 0
    max_rows = 0
    for path in NAME_MAP_ARTIFACTS:
        df = _read(path)
        if df.empty:
            continue
        raw_col = next((c for c in ("raw_name", "book_player_name", "player_raw") if c in df.columns), None)
        canon_col = next((c for c in ("canonical_player_name", "canonical_name") if c in df.columns), None)
        if raw_col is None:
            raise RuntimeError(f"Player-name map has no raw-name column: {path}")
        raw = df[raw_col].astype("string").fillna("").str.strip()
        canon = (
            df[canon_col].astype("string").fillna("").str.strip()
            if canon_col is not None
            else pd.Series("", index=df.index, dtype="string")
        )
        sentinel = raw.eq("") & canon.eq("")
        blank_removed += int(sentinel.sum())
        clean = df.loc[~sentinel].copy()
        raw_clean = clean[raw_col].astype("string").fillna("").str.strip()
        canon_clean = (
            clean[canon_col].astype("string").fillna("").str.strip()
            if canon_col is not None
            else pd.Series("", index=clean.index, dtype="string")
        )
        unresolved_flag = (
            pd.to_numeric(clean["unresolved"], errors="coerce").fillna(0).eq(1)
            if "unresolved" in clean.columns
            else pd.Series(False, index=clean.index)
        )
        bad = raw_clean.ne("") & (canon_clean.eq("") | unresolved_flag)
        unresolved_real += int(bad.sum())
        clean.to_csv(path, index=False)
        max_rows = max(max_rows, int(len(clean)))
    if unresolved_real:
        raise RuntimeError(f"OddsAPI name map contains {unresolved_real} unresolved real player rows")
    return {
        "blank_name_sentinels_removed": int(blank_removed),
        "unresolved_real_player_names": int(unresolved_real),
        "player_name_map_rows": int(max_rows),
    }


def _provider_source_duplicate_rows() -> int:
    """Provider per-market raw offer files must be intrinsically duplicate-free."""
    duplicate_rows = 0
    for path in sorted(OUTPUTS.glob("props_*.csv")):
        if path.name in {"props_raw.csv", "props_enriched.csv", "props_raw_wide.csv"}:
            continue
        df = _read(path)
        duplicate_rows += int(df.duplicated().sum()) if not df.empty else 0
    return int(duplicate_rows)


def harden_live_odds_artifacts() -> dict:
    event_count = validate_game_identity()
    provider_dups = _provider_source_duplicate_rows()
    if provider_dups:
        raise RuntimeError(
            "Provider per-market source files contain exact duplicate offers; refusing a join-only repair. "
            f"duplicate_rows={provider_dups}"
        )

    raw_audit = {str(path): _dedupe_exact_rows(path) for path in RAW_OFFER_ARTIFACTS}
    grouped = _dedupe_grouped_offers()
    names = _sanitize_name_maps()
    raw_removed = int(sum(v["removed"] for v in raw_audit.values()))

    for path in RAW_OFFER_ARTIFACTS:
        df = _read(path)
        if not df.empty and df.duplicated().any():
            raise RuntimeError(f"Post-hardening duplicate rows remain: {path}")

    result = {
        "disposition": "LIVE_ODDS_ARTIFACTS_SEMANTICALLY_HARDENED",
        "game_identity_event_count": int(event_count),
        "provider_source_duplicate_rows": int(provider_dups),
        "raw_artifact_exact_duplicates_removed": int(raw_removed),
        "raw_artifact_audit": raw_audit,
        "grouped_offer_entries_before": int(grouped["entries_before"]),
        "grouped_offer_entries_after": int(grouped["entries_after"]),
        "grouped_offer_entries_removed": int(grouped["removed"]),
        **names,
    }
    audit_path = DATA / "live_odds_artifact_hardening.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    harden_live_odds_artifacts()
