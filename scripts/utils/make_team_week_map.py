#!/usr/bin/env python3
"""Build a team-week opponent map from odds or schedule data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from scripts.utils.name_clean import normalize_team

DATA_DIR = Path("data")
TEAM_WEEK_PATH = DATA_DIR / "team_week_map.csv"
GAME_LINES_PATH = DATA_DIR / "game_lines.csv"
ODDS_PATH = DATA_DIR / "odds_game.csv"
SCHEDULE_PATH = DATA_DIR / "schedule.csv"


def _ensure_int(value: object) -> object:
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return pd.NA
    except Exception:
        pass
    try:
        return int(float(value))
    except Exception:
        return pd.NA


def _canon_pair(a: str, b: str) -> Tuple[str, str]:
    a_norm = (a or "").strip().upper()
    b_norm = (b or "").strip().upper()
    return (a_norm, b_norm) if a_norm <= b_norm else (b_norm, a_norm)


def _first_thursday_on_or_after_sept1(season: int) -> pd.Timestamp:
    """Return the first Thursday on/after September 1 for the given season."""

    anchor = pd.Timestamp(year=season, month=9, day=1, tz="UTC")
    offset = (3 - anchor.weekday()) % 7  # Thursday == 3
    return anchor + pd.Timedelta(days=offset)


def _infer_week_from_kickoff(season: int, kickoff_utc: pd.Series) -> pd.Series:
    anchor = _first_thursday_on_or_after_sept1(season)
    kick = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")
    delta_days = (kick - anchor) / pd.Timedelta(days=1)
    weeks = (delta_days // 7 + 1).astype("float")
    weeks = weeks.where(~pd.isna(weeks), pd.NA)
    weeks = weeks.where(weeks >= 1, 1)
    return weeks.astype("Int64")


def _norm_team(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    out = (
        series.fillna("")
        .astype("string")
        .str.strip()
        .str.upper()
    )
    out = out.map(lambda val: normalize_team(val) if val else val)
    out = out.replace("", pd.NA)
    return out.astype("string")


def _load_or_build_schedule_source(season: int) -> pd.DataFrame:
    candidates = [ODDS_PATH, SCHEDULE_PATH, DATA_DIR / f"schedule_{season}.csv"]
    for path in candidates:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        except Exception as err:  # pragma: no cover - defensive logging
            print(f"[make_team_week_map] WARN: failed to read {path}: {err}")
            continue
        if df is None or df.empty:
            continue

        working = df.copy()
        working.columns = [str(c).lower() for c in working.columns]

        rename_map = {
            "home_team": "home",
            "home_abbr": "home",
            "home_code": "home",
            "home_team_id": "home",
            "away_team": "away",
            "away_abbr": "away",
            "away_code": "away",
            "away_team_id": "away",
        }
        for src, dst in rename_map.items():
            if src in working.columns and dst not in working.columns:
                working.rename(columns={src: dst}, inplace=True)

        if {"home", "away"}.isdisjoint(working.columns):
            continue
        if "home" not in working.columns or "away" not in working.columns:
            continue

        if "season" in working.columns:
            working["season"] = pd.to_numeric(working["season"], errors="coerce").astype("Int64")
        else:
            working["season"] = pd.Series(season, index=working.index, dtype="Int64")

        if "kickoff_utc" not in working.columns and "commence_time" in working.columns:
            working.rename(columns={"commence_time": "kickoff_utc"}, inplace=True)
        if "kickoff" in working.columns and "kickoff_utc" not in working.columns:
            working.rename(columns={"kickoff": "kickoff_utc"}, inplace=True)
        working["kickoff_utc"] = pd.to_datetime(working.get("kickoff_utc"), utc=True, errors="coerce")

        if "week" not in working.columns or working["week"].isna().all():
            working["week"] = _infer_week_from_kickoff(season, working["kickoff_utc"])
        else:
            working["week"] = working["week"].map(_ensure_int).astype("Int64")

        home = working.assign(team=working["home"], opponent=working["away"], home_away="H")
        away = working.assign(team=working["away"], opponent=working["home"], home_away="A")
        combined = pd.concat([home, away], ignore_index=True)

        combined["team"] = _norm_team(combined.get("team"))
        combined["opponent"] = _norm_team(combined.get("opponent"))
        combined["home_away"] = combined.get("home_away", pd.Series(dtype="string")).astype("string").str.upper()
        combined["season"] = pd.to_numeric(combined.get("season"), errors="coerce").astype("Int64")
        combined["week"] = pd.to_numeric(combined.get("week"), errors="coerce").astype("Int64")
        combined["kickoff_utc"] = pd.to_datetime(combined.get("kickoff_utc"), utc=True, errors="coerce")
        combined["bye"] = combined["opponent"].isna() | combined["opponent"].eq("BYE")
        combined.loc[combined["bye"], "opponent"] = "BYE"

        keep = [
            "season",
            "week",
            "team",
            "opponent",
            "home_away",
            "kickoff_utc",
            "bye",
        ]
        extra_cols = [c for c in ("event_id", "game_id") if c in combined.columns]
        return combined.loc[:, keep + extra_cols].dropna(subset=["team"])

    raise FileNotFoundError(
        "Need a schedule source (data/odds_game.csv or data/schedule.csv) to build team_week_map"
    )


def build_map(season: int) -> pd.DataFrame:
    """Assemble the team_week_map for a given season."""

    src = _load_or_build_schedule_source(season)
    df = src.copy()

    for col in ("team", "opponent"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.upper().str.strip()
    if "home_away" in df.columns:
        df["home_away"] = df["home_away"].astype("string").str.upper().str[:1]
    df["season"] = int(season)

    if "week" not in df.columns or df["week"].isna().all():
        df["week"] = _infer_week_from_kickoff(season, df.get("kickoff_utc"))
    else:
        df["week"] = df["week"].map(_ensure_int).astype("Int64")

    if "bye" not in df.columns:
        df["bye"] = False

    keep_cols = [
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "kickoff_utc",
        "bye",
    ]
    extras = [c for c in ("event_id", "game_id") if c in df.columns]
    df = df.loc[:, [c for c in keep_cols + extras if c in df.columns]].copy()
    df["kickoff_utc"] = pd.to_datetime(df.get("kickoff_utc"), utc=True, errors="coerce")

    df = df.sort_values(["season", "week", "team", "kickoff_utc"], na_position="last")
    before = len(df)
    df = df.drop_duplicates(subset=["season", "week", "team"], keep="first")
    dropped = before - len(df)
    if dropped:
        print(
            f"[make_team_week_map] WARNING: dropped {dropped} duplicate rows (kept first per team/week)"
        )
    return df.reset_index(drop=True)


def _write_game_lines_from_team_week_map(tw: pd.DataFrame, out_path: Path = GAME_LINES_PATH) -> pd.DataFrame:
    """Collapse team-week rows into unique games (home vs away)."""

    if tw is None or tw.empty:
        return pd.DataFrame(columns=["season", "week", "home", "away", "kickoff_utc"])

    bye_mask = tw.get("bye")
    if bye_mask is None:
        bye_mask = pd.Series(False, index=tw.index, dtype="boolean")
    else:
        bye_mask = pd.Series(bye_mask).fillna(False).astype(bool)

    fr = tw.loc[~bye_mask].copy()
    fr["home"] = fr.apply(
        lambda row: row.get("team") if str(row.get("home_away", "")).upper() == "H" else row.get("opponent"),
        axis=1,
    )
    fr["away"] = fr.apply(
        lambda row: row.get("opponent") if str(row.get("home_away", "")).upper() == "H" else row.get("team"),
        axis=1,
    )
    fr["pair"] = fr.apply(lambda row: _canon_pair(row.get("home"), row.get("away")), axis=1)

    grp = (
        fr.sort_values(["season", "week", "kickoff_utc"], na_position="last")
        .groupby(["season", "week", "pair"], as_index=False)
        .first()
    )

    out = pd.DataFrame(
        {
            "season": pd.to_numeric(grp.get("season"), errors="coerce").astype("Int64"),
            "week": pd.to_numeric(grp.get("week"), errors="coerce").astype("Int64"),
            "home": grp.get("home"),
            "away": grp.get("away"),
            "kickoff_utc": pd.to_datetime(grp.get("kickoff_utc"), utc=True, errors="coerce"),
        }
    )
    out = out.dropna(subset=["home", "away"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[make_team_week_map] wrote {len(out)} rows → {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--out", type=Path, default=TEAM_WEEK_PATH)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tw = build_map(args.season)
    tw.to_csv(args.out, index=False)
    print(f"[make_team_week_map] wrote {len(tw)} rows → {args.out}")
    _write_game_lines_from_team_week_map(tw, out_path=GAME_LINES_PATH)


if __name__ == "__main__":
    main()
