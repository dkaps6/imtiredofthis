#!/usr/bin/env python3
"""Run the Sharp Football collector with maintained pace-schema adapters.

Sharp's pace table has changed headers over time. Keep provider-specific
changes at this boundary and continue exposing the canonical ``team`` +
``neutral_pace`` contract expected by TeamForm.

As of 2026-09-03 the live table exposes:
``Offense`` + ``Play Clock Used`` + ``Neutral`` + ``Neutral Pass Rate``.
Sharp defines ``Neutral`` as neutral-situation play clock used (lower is faster),
so that exact column maps to canonical ``neutral_pace``. ``Play Clock Used`` is
the all-situation value and must not be substituted for neutral pace.

The legacy generic alias pass also strips underscores while comparing names. If
it is run twice, an already-canonical ``neutral_pace`` becomes ``neutralpace``
and can be mistaken for its own alias, causing the canonical column to be
coalesced with and then dropped from itself. The v2 pace alias adapter is
idempotent and never drops an existing canonical target.

Week-2 operations note (2026-09-17): the old Sharp team-form merger predates the
canonical Coverage-v2 layer and hard-fails when Sharp's legacy coverage tables
are unavailable. Coverage-v2 is built later from its own provider/fallback path
and is the canonical coverage authority. This runner therefore permits that
obsolete coverage requirement to fail softly without inventing man/zone rates.
The real pace source remains required. Other Sharp tendency/line tables are
optional enrichments under AGENTS.md provider policy: if unavailable during this
compatibility path, their unavailability is written explicitly to
``data/sharp_v2_recovery_status.json`` rather than silently masked.
"""
from __future__ import annotations

import json
import os
from io import StringIO
from typing import Any, Dict, Optional

import pandas as pd

import scripts.providers.sharpfootball_pull as sharp

_ORIGINAL_RENAME_EXPECTED_COLS = sharp._rename_expected_cols
_ORIGINAL_MERGE_TEAM_FORM = sharp.merge_team_form
_RECOVERY_STATUS = os.path.join(sharp.DATA_DIR, "sharp_v2_recovery_status.json")
_SOURCE_KINDS = ("def_tend", "off_tend", "pace", "coverage_pos", "coverage_scheme", "dl", "ol")
_REQUIRED_NONCOVERAGE = ("pace",)
_OPTIONAL_NONCOVERAGE = ("def_tend", "off_tend", "dl", "ol")
_COVERAGE_KINDS = ("coverage_pos", "coverage_scheme")


def _flatten_col(col: Any) -> str:
    if isinstance(col, tuple):
        return " ".join(str(part).strip() for part in col if str(part).strip())
    return str(col).strip()


def normalize_pace_table_v2(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("[sharp_v2] pace table is empty")

    pace = df.copy()
    pace.columns = [_flatten_col(c) for c in pace.columns]
    upper = {str(c).strip().upper(): c for c in pace.columns}

    team_col = None
    for candidate in ("TEAM", "OFFENSE", "CLUB"):
        if candidate in upper:
            team_col = upper[candidate]
            break
    if team_col is None:
        for col in pace.columns:
            name = str(col).upper()
            if "TEAM" in name or "OFFENSE" in name or "CLUB" in name:
                team_col = col
                break
    if team_col is None:
        raise RuntimeError(
            f"[sharp_v2] could not identify pace team/offense column; columns={list(pace.columns)}"
        )

    neutral_col = None
    exact = (
        "NEUTRAL",
        "NEUTRAL SCRIPT (SEC/PLAY)",
        "NEUTRAL SCRIPT SEC/PLAY",
        "NEUTRAL SCRIPT (SECONDS/PLAY)",
        "NEUTRAL SECS/PLAY",
        "NEUTRAL PACE",
        "SITUATION NEUTRAL PACE",
    )
    for candidate in exact:
        if candidate in upper:
            neutral_col = upper[candidate]
            break
    if neutral_col is None:
        for col in pace.columns:
            name = str(col).upper()
            if (
                "NEUTRAL" in name
                and ("SEC" in name or "SECOND" in name or "PACE" in name)
                and "DB RATE" not in name
                and "PASS RATE" not in name
            ):
                neutral_col = col
                break
    if neutral_col is None:
        raise RuntimeError(
            f"[sharp_v2] could not identify neutral pace column; columns={list(pace.columns)}"
        )

    rename = {team_col: "team", neutral_col: "neutral_pace"}

    for col in pace.columns:
        name = str(col).upper()
        if col == neutral_col:
            continue
        if "NEUTRAL" in name and ("LAST 5" in name or "L5" in name) and (
            "SEC" in name or "SECOND" in name or "PACE" in name
        ):
            rename[col] = "neutral_pace_last5"
            break

    pace = pace.rename(columns=rename)
    pace["neutral_pace"] = pd.to_numeric(
        pace["neutral_pace"]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )
    if "neutral_pace_last5" in pace.columns:
        pace["neutral_pace_last5"] = pd.to_numeric(
            pace["neutral_pace_last5"]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce",
        )

    if pace["neutral_pace"].notna().sum() == 0:
        raise RuntimeError("[sharp_v2] neutral pace column normalized to all missing")
    usable = pace["neutral_pace"].dropna()
    if not usable.between(10.0, 40.0, inclusive="both").all():
        bad = usable.loc[~usable.between(10.0, 40.0, inclusive="both")].head(10).tolist()
        raise RuntimeError(f"[sharp_v2] neutral pace values outside play-clock seconds range: {bad}")
    return pace


def rename_expected_cols_v2(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    """Idempotent pace aliases; delegate all non-pace tables unchanged."""
    if kind != "pace":
        return _ORIGINAL_RENAME_EXPECTED_COLS(kind, df)

    alias_map = sharp.COLUMN_ALIAS_PATTERNS.get("pace", {})
    out = df.copy()
    targets = sorted(alias_map, key=lambda value: len(sharp._slug(value)), reverse=True)
    for target in targets:
        aliases = alias_map.get(target, set())
        normalized_aliases = {sharp._slug(target)} | {sharp._slug(a) for a in aliases}
        for col in list(out.columns):
            if col in ("team", "team_raw") or col == target:
                continue
            if sharp._slug(col) not in normalized_aliases:
                continue
            if target in out.columns:
                out[target] = out[target].where(out[target].notna(), out[col])
                out.drop(columns=[col], inplace=True)
            else:
                out = out.rename(columns={col: target})
            break
    return out


def fallback_pace_table_v2(
    html: Optional[str] = None,
    season: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    if not html and season is not None:
        html = sharp._fetch_html(sharp.URLS["pace"], int(season), "pace_fallback_v2")
    if not html:
        return None

    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return None

    for table in tables:
        try:
            candidate = normalize_pace_table_v2(table)
            candidate = sharp._normalize_team_col(candidate)
            candidate = rename_expected_cols_v2("pace", candidate)
            candidate = sharp._to_numeric(candidate)
            candidate = candidate.loc[
                candidate["team"].astype(str).isin(sharp.TEAM_CODES),
                ["team", "neutral_pace"],
            ].drop_duplicates("team")
            if not candidate.empty:
                return candidate
        except Exception:
            continue
    return None


def _source_status(pieces: Dict[str, pd.DataFrame], prepared: Dict[str, pd.DataFrame]) -> dict:
    rows = {}
    for kind in _SOURCE_KINDS:
        raw = pieces.get(kind)
        ready = prepared.get(kind)
        rows[kind] = {
            "raw_rows": int(len(raw)) if isinstance(raw, pd.DataFrame) else 0,
            "prepared_rows": int(len(ready)) if isinstance(ready, pd.DataFrame) else 0,
            "available": bool(isinstance(ready, pd.DataFrame) and not ready.empty),
            "policy": (
                "required_core"
                if kind in _REQUIRED_NONCOVERAGE
                else "canonical_coverage_v2_supersedes_legacy"
                if kind in _COVERAGE_KINDS
                else "optional_enrichment"
            ),
        }
    return rows


def _write_recovery_status(*, season: int, source_status: dict, disposition: str) -> None:
    missing_optional = [
        kind for kind in _OPTIONAL_NONCOVERAGE if not bool(source_status[kind]["available"])
    ]
    missing_coverage = [
        kind for kind in _COVERAGE_KINDS if not bool(source_status[kind]["available"])
    ]
    payload = {
        "season": int(season),
        "disposition": disposition,
        "coverage_authority": "Coverage-v2",
        "legacy_coverage_values_fabricated": 0,
        "sportsbook_inputs_used": 0,
        "required_noncoverage_sources": list(_REQUIRED_NONCOVERAGE),
        "missing_optional_enrichments": missing_optional,
        "missing_legacy_coverage_sources": missing_coverage,
        "sources": source_status,
    }
    os.makedirs(os.path.dirname(_RECOVERY_STATUS), exist_ok=True)
    with open(_RECOVERY_STATUS, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[sharp_v2] recovery status -> {_RECOVERY_STATUS}: {payload}")


def _merge_without_legacy_coverage_requirement(
    season: int,
    pieces: Dict[str, pd.DataFrame],
) -> int:
    """Publish real Sharp fields after a legacy coverage-only contract failure.

    The real pace source is core and must exist. Other noncoverage Sharp pieces
    are optional enrichments and are explicitly disclosed if unavailable. Legacy
    man/zone values are never fabricated; canonical Coverage-v2 owns coverage.
    """
    prepared: Dict[str, pd.DataFrame] = {}
    for kind, df in pieces.items():
        prepped = sharp._prepare_piece_for_merge(kind, df)
        if prepped is not None:
            prepared[kind] = prepped

    status = _source_status(pieces, prepared)
    missing_required = [
        kind for kind in _REQUIRED_NONCOVERAGE if not bool(status[kind]["available"])
    ]
    if missing_required:
        _write_recovery_status(
            season=season,
            source_status=status,
            disposition="REJECTED_MISSING_REQUIRED_NONCOVERAGE_SOURCE",
        )
        raise RuntimeError(
            f"[sharp_v2] refusing legacy-coverage recovery; required noncoverage source(s) missing: {missing_required}"
        )

    base = pd.DataFrame({"team": sorted(sharp.TEAM_CODES)})
    for kind in _SOURCE_KINDS:
        df = prepared.get(kind)
        if df is None or df.empty:
            continue
        base = base.merge(df, on="team", how="left")

    raw_cols = [c for c in base.columns if c.startswith("team_raw_")]
    if raw_cols:
        base["team_raw"] = base[raw_cols].bfill(axis=1).iloc[:, 0]
        base.drop(columns=raw_cols, inplace=True)
    base["team_abbr"] = base["team"]

    pace_df = prepared.get("pace")
    if pace_df is not None and "neutral_pace" in pace_df.columns:
        base = base.merge(
            pace_df[["team", "neutral_pace"]],
            on="team",
            how="left",
            suffixes=("", "_pace"),
        )
        if "neutral_pace_pace" in base.columns:
            if "neutral_pace" in base.columns:
                base["neutral_pace"] = base["neutral_pace"].combine_first(base["neutral_pace_pace"])
            else:
                base["neutral_pace"] = base["neutral_pace_pace"]
            base.drop(columns=["neutral_pace_pace"], inplace=True)

    cov_df = prepared.get("coverage_scheme")
    if cov_df is not None:
        cov_cols = [c for c in ("coverage_man_rate", "coverage_zone_rate") if c in cov_df.columns]
        if cov_cols:
            base = base.merge(
                cov_df[["team"] + cov_cols],
                on="team",
                how="left",
                suffixes=("", "_cov"),
            )
            for col in cov_cols:
                aux = f"{col}_cov"
                if aux in base.columns:
                    if col in base.columns:
                        base[col] = base[col].combine_first(base[aux])
                    else:
                        base[col] = base[aux]
                    base.drop(columns=[aux], inplace=True)

    if "neutral_pace" not in base.columns or pd.to_numeric(base["neutral_pace"], errors="coerce").isna().all():
        fallback = fallback_pace_table_v2(season=season)
        if fallback is not None and not fallback.empty:
            base = base.merge(fallback, on="team", how="left", suffixes=("", "_fallback"))
            if "neutral_pace_fallback" in base.columns:
                if "neutral_pace" in base.columns:
                    base["neutral_pace"] = pd.to_numeric(base["neutral_pace"], errors="coerce").combine_first(
                        pd.to_numeric(base["neutral_pace_fallback"], errors="coerce")
                    )
                else:
                    base["neutral_pace"] = pd.to_numeric(base["neutral_pace_fallback"], errors="coerce")
                base.drop(columns=["neutral_pace_fallback"], inplace=True)

    if "neutral_pace" not in base.columns:
        raise RuntimeError("[sharp_v2] legacy-coverage recovery could not recover neutral_pace")
    base["neutral_pace"] = pd.to_numeric(base["neutral_pace"], errors="coerce")
    if base["neutral_pace"].isna().all():
        raise RuntimeError("[sharp_v2] legacy-coverage recovery neutral_pace is all missing")
    base["neutral_pace"] = base["neutral_pace"].fillna(base["neutral_pace"].median())

    for col in ("coverage_man_rate", "coverage_zone_rate"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    out_path = os.path.join(sharp.DATA_DIR, "sharp_team_form.csv")
    base.to_csv(out_path, index=False)
    present = [c for c in ("coverage_man_rate", "coverage_zone_rate") if c in base.columns and base[c].notna().any()]
    _write_recovery_status(
        season=season,
        source_status=status,
        disposition="LEGACY_COVERAGE_REQUIREMENT_BYPASSED_WITH_SOURCE_DISCLOSURE",
    )
    print(
        "[sharp_v2] LEGACY_COVERAGE_SOURCE_UNAVAILABLE_CONTINUING_TO_COVERAGE_V2 "
        f"rows={len(base)} real_coverage_cols={present} out={out_path}"
    )
    return len(base)


def merge_team_form_v2(season: int, pieces: Dict[str, pd.DataFrame]) -> int:
    try:
        return _ORIGINAL_MERGE_TEAM_FORM(season, pieces)
    except RuntimeError as exc:
        msg = str(exc)
        coverage_failure = (
            "coverage_man_rate" in msg
            or "coverage_zone_rate" in msg
            or "missing or empty required col coverage_" in msg
        )
        if not coverage_failure:
            raise
        return _merge_without_legacy_coverage_requirement(season, pieces)


def main() -> None:
    sharp._normalize_pace_table = normalize_pace_table_v2
    sharp._rename_expected_cols = rename_expected_cols_v2
    sharp._fallback_pace_table = fallback_pace_table_v2
    sharp.merge_team_form = merge_team_form_v2
    sharp.main()


if __name__ == "__main__":
    main()
