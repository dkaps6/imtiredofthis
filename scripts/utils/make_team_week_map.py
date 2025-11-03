#!/usr/bin/env python3
"""Build a team-week opponent map from odds or schedule data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.utils.name_clean import normalize_team


ODDS_PATH = Path("data/odds_game.csv")
SCHEDULE_PATH = Path("data/schedule.csv")
OUT_PATH = Path("data/team_week_map.csv")


def _first_thursday_on_or_after_sept1(season: int) -> pd.Timestamp:
    """Approximate NFL Week 1 anchor: first Thursday on/after Sept 1 (UTC)."""

    # Create Sept 1st as a timezone-aware timestamp in a single, unambiguous way.
    # Using the string constructor avoids multiple tz pathways that can clash.
    d = pd.Timestamp(f"{season}-09-01", tz="UTC")
    # Thursday = 3 (Mon=0)
    offset = (3 - d.weekday()) % 7
    return d + pd.Timedelta(days=offset)


def _infer_week_from_kickoff(season: int, kickoff: pd.Series) -> pd.Series:
    """Infer NFL 'week' from kickoff timestamps when a week column isn't present.

    Approximation: Week 1 window starts the Tuesday prior to the Week 1 Thursday,
    and each week advances every 7 days. Result is clipped to [1, 22].
    """

    if kickoff is None:
        return pd.Series(dtype="Int64")

    ts = pd.to_datetime(kickoff, errors="coerce", utc=True)
    if ts.isna().all():
        return pd.Series(pd.NA, index=ts.index, dtype="Int64")

    anchor_thu = _first_thursday_on_or_after_sept1(season)
    week1_start = (anchor_thu - pd.Timedelta(days=2)).normalize()  # Tuesday 00:00 UTC
    weeks = ((ts - week1_start) / pd.Timedelta(days=7)).floordiv(1).astype("Int64") + 1
    weeks = weeks.where(weeks >= 1, 1).where(weeks <= 22, 22)
    return weeks


def _read_first(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            except Exception as err:  # pragma: no cover - defensive logging
                print(f"[team_week_map] WARN: failed to read {path}: {err}")
                continue
            if not df.empty:
                return df
    return pd.DataFrame()


def _norm_team(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    normalized = series.fillna("").astype("string").map(normalize_team)
    normalized = normalized.where(normalized.ne(""), pd.NA)
    return normalized.astype("string")


def build_map(season: int) -> pd.DataFrame:
    """Return a long-form team/week schedule with opponents."""

    odds = _read_first([ODDS_PATH])
    schedule = _read_first([SCHEDULE_PATH])

    df = odds if not odds.empty else schedule
    if df is None or df.empty:
        raise FileNotFoundError(
            "Need data/odds_game.csv or data/schedule.csv to build team_week_map"
        )

    working = df.copy()
    working.columns = [c.lower() for c in working.columns]

    if "season" in working.columns:
        working["season"] = pd.to_numeric(working["season"], errors="coerce").astype("Int64")
    else:
        working["season"] = pd.Series(season, index=working.index, dtype="Int64")

    if "kickoff_utc" not in working.columns and "commence_time" in working.columns:
        working.rename(columns={"commence_time": "kickoff_utc"}, inplace=True)

    if "kickoff_utc" not in working.columns:
        working["kickoff_utc"] = pd.NaT
    else:
        working["kickoff_utc"] = pd.to_datetime(working["kickoff_utc"], errors="coerce", utc=True)

    if "week" in working.columns:
        working["week"] = pd.to_numeric(working["week"], errors="coerce")
    else:
        inferred = _infer_week_from_kickoff(season, working["kickoff_utc"])
        working["week"] = inferred

    if working["week"].isna().any():
        missing = int(working["week"].isna().sum())
        raise ValueError(
            f"Failed to determine week for {missing} rows in schedule source"
        )

    working["week"] = working["week"].astype(int)

    rename_map = {
        "home": "home_team",
        "away": "away_team",
        "home_team_id": "home_team",
        "away_team_id": "away_team",
    }
    for src, dst in rename_map.items():
        if src in working.columns and dst not in working.columns:
            working.rename(columns={src: dst}, inplace=True)

    if "home_team" not in working.columns or "away_team" not in working.columns:
        raise ValueError("schedule source missing home_team/away_team columns")

    working["home_team"] = _norm_team(working["home_team"])
    working["away_team"] = _norm_team(working["away_team"])

    if "event_id" in working.columns:
        game_id = working["event_id"].astype("string").str.strip()
    elif "game_id" in working.columns:
        game_id = working["game_id"].astype("string").str.strip()
    else:
        game_id = working.index.astype(str)
    working["game_id"] = game_id.where(game_id.ne(""), working.index.astype(str))

    if "venue" not in working.columns:
        working["venue"] = pd.NA

    home = working.assign(team=working["home_team"], opponent=working["away_team"], is_home=True)
    away = working.assign(team=working["away_team"], opponent=working["home_team"], is_home=False)

    cols = [
        "season",
        "week",
        "game_id",
        "kickoff_utc",
        "venue",
        "team",
        "opponent",
        "is_home",
    ]
    tw = pd.concat([home[cols], away[cols]], ignore_index=True)
    tw["team"] = _norm_team(tw["team"])
    tw["opponent"] = _norm_team(tw["opponent"])

    tw["is_bye"] = tw["opponent"].isna() | tw["opponent"].eq("BYE")
    tw.loc[tw["is_bye"], "opponent"] = "BYE"
    tw["is_bye"] = tw["is_bye"].fillna(False).astype(bool)

    tw["game_id"] = tw["game_id"].astype("string").str.strip()
    tw.loc[tw["game_id"].eq(""), "game_id"] = pd.NA
    tw["kickoff_utc"] = pd.to_datetime(tw["kickoff_utc"], errors="coerce", utc=True)

    out_cols = ["season", "week", "team", "opponent", "game_id", "kickoff_utc", "is_bye"]
    out = tw.loc[:, out_cols].copy()

    # --- NEW: safeguard against duplicates (one row per team/week) ---
    dupes = out[out.duplicated(subset=["season", "week", "team"], keep=False)]
    if not dupes.empty:
        # keep the earliest kickoff (or arbitrary first) per team/week
        out = (
            out.sort_values(["season", "week", "kickoff_utc"])
            .drop_duplicates(subset=["season", "week", "team"], keep="first")
        )
        print(
            f"[make_team_week_map] WARNING: dropped {len(dupes)} duplicate rows (kept first per team/week)"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tw = build_map(args.season)
    tw.to_csv(args.out, index=False)
    n_rows = len(tw)
    print(f"[team_week_map] wrote {n_rows} rows -> {args.out}")
    if n_rows == 0:
        raise RuntimeError(
            "team_week_map produced 0 rows. Check data/odds_game.csv (or data/schedule.csv) columns: "
            "season, week, home/home_team, away/away_team."
        )


if __name__ == "__main__":
    main()
