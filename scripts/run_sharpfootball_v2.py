#!/usr/bin/env python3
"""Run the Sharp Football collector with maintained 2026 schema adapters.

Sharp's pace table changed in 2026 from older ``neutral pace`` style headers to
``Offense`` + ``Neutral Script (Sec/Play)``.  Keep that provider-specific change
at the boundary and continue exposing the canonical ``team`` + ``neutral_pace``
contract expected by TeamForm.
"""
from __future__ import annotations

from io import StringIO
from typing import Any, Optional

import pandas as pd

import scripts.providers.sharpfootball_pull as sharp


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
            ):
                neutral_col = col
                break
    if neutral_col is None:
        raise RuntimeError(
            f"[sharp_v2] could not identify neutral pace column; columns={list(pace.columns)}"
        )

    rename = {team_col: "team", neutral_col: "neutral_pace"}

    # Preserve useful recent-pace variants if Sharp supplies them under a new
    # label, but never confuse a pass-rate field for seconds/play.
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
    return pace


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
    sharp._fallback_pace_table = fallback_pace_table_v2
    sharp.main()


if __name__ == "__main__":
    main()
