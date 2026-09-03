#!/usr/bin/env python3
"""Run the Sharp Football collector with maintained pace-schema adapters.

Sharp's pace table has changed headers over time.  Keep provider-specific
changes at this boundary and continue exposing the canonical ``team`` +
``neutral_pace`` contract expected by TeamForm.

As of 2026-09-03 the live table exposes:
``Offense`` + ``Play Clock Used`` + ``Neutral`` + ``Neutral Pass Rate``.
Sharp defines ``Neutral`` as neutral-situation play clock used (lower is faster),
so that exact column maps to canonical ``neutral_pace``.  ``Play Clock Used`` is
the all-situation value and must not be substituted for neutral pace.

The legacy generic alias pass also strips underscores while comparing names. If
it is run twice, an already-canonical ``neutral_pace`` becomes ``neutralpace``
and can be mistaken for its own alias, causing the canonical column to be
coalesced with and then dropped from itself. The v2 pace alias adapter is
idempotent and never drops an existing canonical target.
"""
from __future__ import annotations

from io import StringIO
from typing import Any, Optional

import pandas as pd

import scripts.providers.sharpfootball_pull as sharp

_ORIGINAL_RENAME_EXPECTED_COLS = sharp._rename_expected_cols


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
        # Current live Sharp schema.  This is neutral play-clock-used pace,
        # distinct from the adjacent all-situation Play Clock Used column.
        "NEUTRAL",
        # Historical/live variants retained for source continuity.
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
    # Semantic guard: Sharp neutral play-clock-used values are seconds on the
    # 40-second play clock.  This catches accidental mapping to percentages,
    # ranks, or pass-rate fields without fabricating/substituting data.
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

    # More-specific targets first so a future neutral_pace_last5 cannot be
    # collapsed into neutral_pace merely because both share a prefix.
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


def main() -> None:
    sharp._normalize_pace_table = normalize_pace_table_v2
    sharp._rename_expected_cols = rename_expected_cols_v2
    sharp._fallback_pace_table = fallback_pace_table_v2
    sharp.main()


if __name__ == "__main__":
    main()
