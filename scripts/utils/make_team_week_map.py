#!/usr/bin/env python3
"""Build a team-week opponent map from odds or schedule data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.utils.team_maps import TEAM_NAME_TO_ABBR


ODDS_PATH = Path("data/odds_game.csv")
SCHEDULE_PATH = Path("data/schedule.csv")
OUT_PATH = Path("data/team_week_map.csv")


def _first_thursday_on_or_after_sept1(season: int) -> pd.Timestamp:
    """Approximate NFL Week 1 anchor: first Thursday on/after Sept 1 (UTC)."""

    d = pd.Timestamp(season, 9, 1, tz="UTC")
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
    upper = series.fillna("").astype("string").str.upper().str.strip()
    mapped = upper.map(TEAM_NAME_TO_ABBR.get)
    fallback = upper.where(upper.isin(TEAM_NAME_TO_ABBR.values()), "")
    resolved = mapped.fillna(fallback)
    resolved = resolved.replace("", pd.NA)
    return resolved.astype("string")


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
        working["week"] = pd.to_numeric(working["week"], errors="coerce").astype("Int64")
    else:
        working["week"] = _infer_week_from_kickoff(season, working["kickoff_utc"]).astype("Int64")

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

    duplicate_mask = tw.duplicated(["season", "week", "team"], keep=False)
    if duplicate_mask.any():
        dupes = tw.loc[duplicate_mask, ["season", "week", "team"]].drop_duplicates()
        raise RuntimeError(
            "Duplicate (season, week, team) detected in team_week_map; check schedule sources\n"
            + dupes.to_string(index=False)
        )

    return tw


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
