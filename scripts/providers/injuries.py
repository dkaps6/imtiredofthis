#!/usr/bin/env python3
"""Fetch injury reports for the given season and materialise data/injuries.csv."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path("data")
OUTPATH = DATA_DIR / "injuries.csv"


def _first_present(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    for col in columns:
        if col in frame.columns:
            return col
    return None


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    player_col = _first_present(
        [
            "player",  # nflreadpy
            "player_display_name",
            "player_name",
            "full_name",
            "gsis_name",
        ],
        df,
    )
    team_col = _first_present(
        [
            "team",
            "recent_team",
            "team_abbr",
            "club",
            "team_code",
            "abbr",
        ],
        df,
    )
    status_col = _first_present(
        [
            "status",
            "game_status",
            "injury_status",
            "practice_status",
            "practice_participation",
        ],
        df,
    )

    if player_col is None:
        return pd.DataFrame(columns=["player", "team", "status"])

    out = pd.DataFrame({"player": df[player_col].astype(str).str.strip()})

    if team_col is not None:
        out["team"] = df[team_col].astype(str).str.upper().str.strip()
    else:
        out["team"] = ""

    if status_col is not None:
        out["status"] = df[status_col].astype(str).str.title().str.strip()
    else:
        out["status"] = "Unknown"

    if "report_date" in df.columns:
        out["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    elif "date" in df.columns:
        out["report_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        out["report_date"] = pd.NaT

    return out[["player", "team", "status", "report_date"]]


def _write_stub() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    stub = pd.DataFrame(columns=["player", "team", "status", "report_date"])
    stub.to_csv(OUTPATH, index=False)
    print(f"[injuries] wrote empty stub → {OUTPATH}")


def main(season: int) -> int:
    DATA_DIR.mkdir(exist_ok=True)

    sources: list[str] = []
    df = pd.DataFrame()
    used_source = ""

    try:
        import nfl_data_py as nfl

        temp = nfl.import_injuries([season])
        if temp is not None and not temp.empty:
            df = temp
            used_source = "nfl_data_py.import_injuries"
    except Exception as exc:  # pragma: no cover - network/dep specific
        sources.append(f"nfl_data_py: {exc}")

    if df.empty:
        try:
            import nflreadpy as nflv

            temp = nflv.load_injuries(seasons=[season])
            if temp is not None and not temp.empty:
                df = temp
                used_source = "nflreadpy.load_injuries"
        except Exception as exc:  # pragma: no cover - network/dep specific
            sources.append(f"nflreadpy: {exc}")

    if df.empty:
        if sources:
            print(f"[injuries] no injury data fetched ({'; '.join(sources)})")
        _write_stub()
        return 0

    normalised = _standardise(df)
    if normalised.empty:
        if sources:
            print(f"[injuries] normalisation produced no rows ({'; '.join(sources)})")
        _write_stub()
        return 0

    normalised = normalised.drop_duplicates()
    normalised.to_csv(OUTPATH, index=False)
    print(f"[injuries] wrote {len(normalised)} rows via {used_source} → {OUTPATH}")
    return 0


if __name__ == "__main__":
    season = int(os.getenv("SEASON", os.getenv("season", "2025")))
    sys.exit(main(season))
