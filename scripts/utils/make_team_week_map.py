#!/usr/bin/env python3
"""Build a team-week opponent map from odds or schedule data."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.utils.name_clean import normalize_team


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def _write_game_lines_from_team_week_map(
    tw: pd.DataFrame, out_path: str = "data/game_lines.csv"
) -> None:
    """
    Convert team_week_map rows to a per-game slate with home/away + kickoff times.
    Expected inputs in `tw` (at least): season, week, home, away
    Optional inputs used if present: local_tz, kickoff_local, kickoff_utc, kickoff (any tz)
    If no kickoff present, synthesize 13:00 local time for the home team (UTC conversion if local_tz given).
    """

    if tw is None or tw.empty:
        return

    df = tw.copy()

    # normalize column names we might get from upstream
    rename_map = {
        "home_team": "home",
        "away_team": "away",
        "home_abbr": "home",
        "away_abbr": "away",
        "kickoff_raw": "kickoff_local",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Pick the columns we care about
    keep = [
        c
        for c in [
            "season",
            "week",
            "home",
            "away",
            "local_tz",
            "kickoff_local",
            "kickoff_utc",
            "kickoff",
        ]
        if c in df.columns
    ]
    slate = (
        df[keep]
        .drop_duplicates(subset=["season", "week", "home", "away"])
        .reset_index(drop=True)
    )

    # If we have only a generic 'kickoff', treat it as local time if local_tz exists; otherwise leave it as is.
    if "kickoff" in slate.columns and "kickoff_local" not in slate.columns:
        slate["kickoff_local"] = slate["kickoff"]

    # Synthesize missing kickoff timestamps at 13:00 local
    if "kickoff_local" not in slate.columns:
        slate["kickoff_local"] = pd.NaT
    if "local_tz" not in slate.columns:
        slate["local_tz"] = None

    mask_missing = slate["kickoff_local"].isna()
    if mask_missing.any():
        default_times: list[pd.Timestamp | datetime] = []
        for _, row in slate.loc[mask_missing].iterrows():
            tz_name = row.get("local_tz")
            tz = None
            if isinstance(tz_name, str) and tz_name:
                try:
                    tz = ZoneInfo(tz_name)
                except Exception:
                    tz = None
            if tz is None:
                tz = ZoneInfo("UTC")
            inferred_date = None
            if "season" in row and "week" in row:
                try:
                    season = int(row["season"])
                    week = int(row["week"])
                    inferred_date = _first_thursday_on_or_after_sept1(season) + pd.Timedelta(
                        days=(week - 1) * 7
                    )
                except Exception:
                    inferred_date = None
            if inferred_date is None:
                inferred_date = pd.Timestamp.utcnow()
            kickoff_dt = pd.Timestamp(
                year=inferred_date.year,
                month=inferred_date.month,
                day=inferred_date.day,
                hour=13,
                minute=0,
                tz=tz,
            )
            default_times.append(kickoff_dt)
        slate.loc[mask_missing, "kickoff_local"] = default_times

    # Build kickoff_utc if possible
    if "kickoff_utc" not in slate.columns:
        slate["kickoff_utc"] = pd.NaT
    try:
        for idx, row in slate.iterrows():
            kl = row.get("kickoff_local", pd.NaT)
            tz_name = row.get("local_tz")
            if pd.isna(kl):
                continue
            if isinstance(kl, str):
                try:
                    kl_dt = pd.to_datetime(kl)
                except Exception:
                    kl_dt = pd.NaT
            else:
                kl_dt = pd.to_datetime(kl)
            if pd.isna(kl_dt):
                continue
            if getattr(kl_dt, "tzinfo", None) is None:
                if isinstance(tz_name, str) and tz_name:
                    try:
                        kl_dt = kl_dt.tz_localize(ZoneInfo(tz_name))
                    except (TypeError, ValueError):
                        kl_dt = kl_dt.tz_localize("UTC")
                else:
                    kl_dt = kl_dt.tz_localize("UTC")
            if tz_name and isinstance(tz_name, str):
                try:
                    slate.at[idx, "local_tz"] = tz_name
                except Exception:
                    pass
            try:
                slate.at[idx, "kickoff_utc"] = kl_dt.astimezone(ZoneInfo("UTC"))
            except Exception:
                pass
    except Exception:
        pass

    out_cols = ["home", "away", "local_tz", "kickoff_local", "kickoff_utc"]
    for c in out_cols:
        if c not in slate.columns:
            slate[c] = pd.NA

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slate[out_cols].to_csv(out_path, index=False)


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
    _write_game_lines_from_team_week_map(tw, out_path="data/game_lines.csv")
    try:
        gl_rows = len(pd.read_csv("data/game_lines.csv"))
    except Exception:
        gl_rows = 0
    print(f"[team_week_map] wrote data/game_lines.csv rows={gl_rows}")
    n_rows = len(tw)
    print(f"[team_week_map] wrote {n_rows} rows -> {args.out}")
    if n_rows == 0:
        raise RuntimeError(
            "team_week_map produced 0 rows. Check data/odds_game.csv (or data/schedule.csv) columns: "
            "season, week, home/home_team, away/away_team."
        )


if __name__ == "__main__":
    main()
