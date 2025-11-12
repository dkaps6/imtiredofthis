#!/usr/bin/env python3
"""Build a team-week opponent map from odds or schedule data."""
from __future__ import annotations

import argparse
import io
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# import both names; we’ll use a single local name
from scripts._opponent_map import canon_team as _canon_any, _canon_team_series as _canon_series
from scripts.utils.name_clean import normalize_team


# unify on one local callable name that works for both scalar and Series
def canon_team(x):
    return _canon_any(x) if not isinstance(x, pd.Series) else _canon_series(x)

SCHED_DIR = Path("data/schedules")
SCHED_DIR.mkdir(parents=True, exist_ok=True)

NFLVERSE_URL = "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/sched_{season}.csv.gz"
# PFR kept as last-resort; often 403s on CI
PFR_URL = "https://www.pro-football-reference.com/years/{season}/games.htm"
# ESPN API (weeks 1..23 in regular/post types)
ESPN_WEEK_EVENTS = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/2/weeks/{week}/events"

LEGACY_SCHED = Path("data/schedule.csv")
_EXPECTED_COLS = {"season", "week", "gameday", "game_id", "home_team", "away_team"}

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
}


def _validate_schedule(df: pd.DataFrame) -> pd.DataFrame:
    if not _EXPECTED_COLS.issubset(df.columns):
        raise ValueError(
            f"schedule missing cols; have={sorted(df.columns)} need={sorted(_EXPECTED_COLS)}"
        )
    out = df.loc[:, sorted(_EXPECTED_COLS)].copy()
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce", utc=True)
    return out


def _fetch_schedule_nflverse(season: int) -> pd.DataFrame:
    url = NFLVERSE_URL.format(season=season)
    r = requests.get(url, timeout=30)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"nflverse schedule fetch failed: {url} status={r.status_code}")
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
        df = pd.read_csv(gz)
    # ensure canonical columns
    if "gameday" not in df.columns and "gamedate" in df.columns:
        df["gameday"] = df["gamedate"]
    df = df.rename(columns={"home_team": "home_team", "away_team": "away_team"})
    df["home_team"] = canon_team(df["home_team"])
    df["away_team"] = canon_team(df["away_team"])
    return _validate_schedule(
        df[["season", "week", "gameday", "game_id", "home_team", "away_team"]]
    )


def _fetch_schedule_espn(season: int) -> pd.DataFrame:
    # Regular season weeks typically 1..18; include 19..23 to cover bye/post-season variations safely
    rows = []
    for wk in range(1, 24):
        url = ESPN_WEEK_EVENTS.format(season=season, week=wk)
        r = requests.get(url, timeout=20, headers=UA)
        if r.status_code != 200:
            # Stop if the week endpoint disappears for future weeks, but keep any rows we got
            continue
        try:
            data = json.loads(r.text)
        except json.JSONDecodeError:
            continue
        if "items" not in data or not data["items"]:
            continue
        for item in data["items"]:
            # item is a URL to the event; fetch details
            ev_resp = requests.get(item, timeout=20, headers=UA)
            try:
                ev = json.loads(ev_resp.text)
            except json.JSONDecodeError:
                continue
            # event 'competitions'[0] contains competitors & date
            if not ev.get("competitions"):
                continue
            comp = ev["competitions"][0]
            date_iso = comp.get("date")
            gameday = pd.NaT
            if date_iso:
                try:
                    dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00")).astimezone(
                        timezone.utc
                    )
                    gameday = pd.to_datetime(dt, utc=True)
                except Exception:
                    try:
                        gameday = pd.to_datetime(date_iso, utc=True)
                    except Exception:
                        gameday = pd.NaT

            home_abbr, away_abbr = None, None
            for c in comp.get("competitors", []):
                team = c.get("team", {})
                abbr = team.get("abbreviation")
                if c.get("homeAway") == "home":
                    home_abbr = abbr
                elif c.get("homeAway") == "away":
                    away_abbr = abbr
            if not home_abbr or not away_abbr:
                continue

            home_abbr = canon_team(pd.Series([home_abbr])).iloc[0]
            away_abbr = canon_team(pd.Series([away_abbr])).iloc[0]
            game_id = f"{season}_{wk}_{home_abbr}_{away_abbr}"
            rows.append(
                {
                    "season": season,
                    "week": wk,
                    "gameday": gameday,
                    "game_id": game_id,
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                }
            )
    if not rows:
        raise RuntimeError("ESPN schedule fetch produced 0 rows")
    df = pd.DataFrame(rows).drop_duplicates(subset=["game_id"])
    return _validate_schedule(df)


# PFR single page lists all weeks for a season

# Map PFR display names to your abbreviations (extend as needed)
PFR_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def _name_to_abbr(s: pd.Series) -> pd.Series:
    # handle full names (PFR) -> our abbrs, then canonicalize to our codes
    out = s.map(PFR_TO_ABBR).fillna(s)
    return canon_team(out)


def _fetch_schedule_pfr(season: int) -> pd.DataFrame:
    url = PFR_URL.format(season=season)
    resp = requests.get(url, headers=UA, timeout=30)
    if resp.status_code != 200 or not resp.text:
        raise RuntimeError(f"PFR fetch failed: {url} status={resp.status_code}")

    # parse all tables and pick the first matching one
    tables = pd.read_html(resp.text)
    t = None
    for tbl in tables:
        cols = set(tbl.columns.astype(str))
        if {"Week", "Date"}.issubset(cols) and (
            {"Visitor", "Home"}.issubset(cols)
            or {"Winner/tie", "Loser/tie", "Unnamed: 5"}.issubset(cols)
            or {"Winner/tie", "Loser/tie", "at"}.issubset(cols)
        ):
            t = tbl
            break
    if t is None:
        # helpful debug drop (optional): write HTML to help investigate
        with open(f"data/schedules/pfr_{season}.html", "w", encoding="utf-8") as f:
            f.write(resp.text)
        raise RuntimeError(f"PFR schedule table not found: {url}")

    # keep only numeric NFL weeks (skip Preseason/Playoffs rows)
    mask_week = t["Week"].astype(str).str.match(r"^\d+$", na=False)
    t = t.loc[mask_week].copy()
    t["season"] = int(season)
    t["week"] = t["Week"].astype(int)
    t["gameday"] = pd.to_datetime(t["Date"], errors="coerce")
    if {"Visitor", "Home"}.issubset(set(t.columns)):
        # future-schedule style
        t["away_team"] = _name_to_abbr(t["Visitor"])
        t["home_team"] = _name_to_abbr(t["Home"])
    else:
        # results style: Winner/Loser with 'at' flag ('@' means Winner was away)
        at_col = "at" if "at" in t.columns else "Unnamed: 5"
        is_winner_away = t[at_col].astype(str).str.strip().eq("@")
        winner = _name_to_abbr(t["Winner/tie"])
        loser = _name_to_abbr(t["Loser/tie"])
        t["home_team"] = winner.where(~is_winner_away, loser)
        t["away_team"] = winner.where(is_winner_away, loser)

    t["game_id"] = (
        t["season"].astype(str)
        + "_"
        + t["week"].astype(str)
        + "_"
        + t["home_team"]
        + "_"
        + t["away_team"]
    )
    t = t[["season", "week", "gameday", "game_id", "home_team", "away_team"]]
    return _validate_schedule(t)

DATA_DIR = Path("data")
TEAM_WEEK_PATH = DATA_DIR / "team_week_map.csv"
GAME_LINES_PATH = DATA_DIR / "game_lines.csv"


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


def _canon_pair(a: str, b: str) -> tuple[str, str]:
    a_norm = (a or "").strip().upper()
    b_norm = (b or "").strip().upper()
    return (a_norm, b_norm) if a_norm <= b_norm else (b_norm, a_norm)


def _first_thursday_on_or_after_sept1(season: int) -> pd.Timestamp:
    """Return the first Thursday on/after September 1 for the given season."""

    # Construct from string to avoid timezone warnings on some pandas builds.
    anchor = pd.Timestamp(f"{season}-09-01 00:00:00", tz="UTC")
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


def _prepare_schedule_rows(df: pd.DataFrame, season: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

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

    if "home" not in working.columns or "away" not in working.columns:
        return pd.DataFrame()

    if "season" in working.columns:
        working["season"] = pd.to_numeric(working["season"], errors="coerce").astype("Int64")
    else:
        working["season"] = pd.Series(season, index=working.index, dtype="Int64")

    if "kickoff_utc" not in working.columns and "commence_time" in working.columns:
        working.rename(columns={"commence_time": "kickoff_utc"}, inplace=True)
    if "kickoff_utc" not in working.columns and "gameday" in working.columns:
        working.rename(columns={"gameday": "kickoff_utc"}, inplace=True)
    if "kickoff" in working.columns and "kickoff_utc" not in working.columns:
        working.rename(columns={"kickoff": "kickoff_utc"}, inplace=True)
    working["kickoff_utc"] = pd.to_datetime(working.get("kickoff_utc"), utc=True, errors="coerce")

    if "week" not in working.columns or working["week"].isna().all():
        working["week"] = _infer_week_from_kickoff(season, working["kickoff_utc"])
    else:
        working["week"] = working["week"].map(_ensure_int).astype("Int64")

    home = working.assign(
        team=working["home"],
        opponent=working["away"],
        home_away="H",
        home_abbr=working["home"],
        away_abbr=working["away"],
    )
    away = working.assign(
        team=working["away"],
        opponent=working["home"],
        home_away="A",
        home_abbr=working["home"],
        away_abbr=working["away"],
    )
    combined = pd.concat([home, away], ignore_index=True)

    combined["team"] = _norm_team(combined.get("team"))
    combined["opponent"] = _norm_team(combined.get("opponent"))
    combined["home_abbr"] = _norm_team(combined.get("home_abbr"))
    combined["away_abbr"] = _norm_team(combined.get("away_abbr"))
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
        "home_abbr",
        "away_abbr",
        "home_away",
        "kickoff_utc",
        "bye",
    ]
    extra_cols = [c for c in ("event_id", "game_id") if c in combined.columns]
    keep = [c for c in keep if c in combined.columns]
    return combined.loc[:, keep + extra_cols].dropna(subset=["team"])


def _load_or_build_schedule_source(
    season: int, schedule_path: Optional[str] = None
) -> pd.DataFrame:
    cache_csv = SCHED_DIR / f"schedule_{season}.csv"

    # 0) explicit file if provided
    if schedule_path:
        try:
            df = pd.read_csv(schedule_path)
            df["home_team"] = canon_team(df["home_team"])
            df["away_team"] = canon_team(df["away_team"])
            df = _validate_schedule(df)
            print(f"[team_week_map] Using explicit schedule: {schedule_path} rows={len(df)}")
            return df
        except Exception as e:
            print(f"[team_week_map] Explicit schedule invalid; ignoring: {e}")
            # if it's our cache path, nuke it so we can refetch
            try:
                sp = Path(schedule_path)
                if sp.resolve().is_file() and sp.parent.resolve() == SCHED_DIR.resolve():
                    sp.unlink(missing_ok=True)
                    print(f"[team_week_map] Deleted invalid explicit cache: {schedule_path}")
            except Exception:
                pass

    # 1) cached per-season
    if cache_csv.exists() and cache_csv.stat().st_size > 0:
        try:
            df = _validate_schedule(pd.read_csv(cache_csv))
            print(f"[team_week_map] Using cached schedule: {cache_csv} rows={len(df)}")
            return df
        except Exception as e:
            print(f"[team_week_map] Bad cache; refetching: {e}")
            try:
                cache_csv.unlink(missing_ok=True)
            except Exception:
                pass

    # 2) legacy single-file data/schedule.csv
    if LEGACY_SCHED.exists() and LEGACY_SCHED.stat().st_size > 0:
        try:
            legacy = pd.read_csv(LEGACY_SCHED)
            legacy["home_team"] = canon_team(legacy["home_team"])
            legacy["away_team"] = canon_team(legacy["away_team"])
            legacy = _validate_schedule(legacy)
            legacy_this = legacy[legacy["season"] == int(season)].copy()
            if not legacy_this.empty:
                legacy_this.to_csv(cache_csv, index=False)
                print(f"[team_week_map] Migrated legacy → {cache_csv} rows={len(legacy_this)}")
                return legacy_this
            else:
                print(f"[team_week_map] Legacy schedule has no rows for season {season}; will fetch.")
        except Exception as e:
            print(f"[team_week_map] Legacy schedule invalid; ignoring: {e}")

    # 3) nflverse (preferred)
    try:
        df = _fetch_schedule_nflverse(season)
        df.to_csv(cache_csv, index=False)
        print(f"[team_week_map] Fetched nflverse → {cache_csv} rows={len(df)}")
        return df
    except Exception as e:
        print(f"[team_week_map] nflverse failed: {e}")

    # 4) ESPN (new, CI-friendly)
    try:
        df = _fetch_schedule_espn(season)
        df.to_csv(cache_csv, index=False)
        print(f"[team_week_map] Fetched ESPN → {cache_csv} rows={len(df)}")
        return df
    except Exception as e:
        print(f"[team_week_map] ESPN failed: {e}")

    # 5) PFR fallback (last resort; may 403 on CI)
    try:
        df = _fetch_schedule_pfr(season)
        df.to_csv(cache_csv, index=False)
        print(f"[team_week_map] Fetched PFR → {cache_csv} rows={len(df)}")
        return df
    except Exception as e:
        print(f"[team_week_map] PFR failed: {e}")

    # 6) last resort (give downstream something to join)
    raise RuntimeError("Could not build schedule from any source")


def build_map(season: int, schedule_path: Optional[str] = None) -> pd.DataFrame:
    """Assemble the team_week_map for a given season."""

    df_sched = _load_or_build_schedule_source(season, schedule_path)
    print(
        f"[team_week_map] seasons={df_sched['season'].unique().tolist()} "
        f"weeks={df_sched['week'].nunique()} games={len(df_sched)}"
    )

    df = _prepare_schedule_rows(df_sched, season)

    if df.empty:
        raise FileNotFoundError("Materialized schedule did not contain usable rows")

    print(f"[make_team_week_map] schedule rows: {len(df)} for season={season}")
    df = df.copy()

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
        "home_abbr",
        "away_abbr",
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


def _write_game_lines_from_team_week_map(
    tw: pd.DataFrame, out_path: Path = GAME_LINES_PATH
) -> pd.DataFrame:
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

    if not out.empty:
        season_str = out["season"].astype("Int64").astype("string")
        week_str = out["week"].astype("Int64").astype("string").str.zfill(2)
        home_str = out["home"].astype("string").str.upper().str.strip()
        away_str = out["away"].astype("string").str.upper().str.strip()
        out["game_id"] = season_str + "_" + week_str + "_" + home_str + "_" + away_str
        out.loc[
            season_str.isna()
            | week_str.isna()
            | home_str.isna()
            | away_str.isna(),
            "game_id",
        ] = pd.NA

    stadium_meta = pd.DataFrame(
        [
            ("ARI", "State Farm Stadium", "US/Arizona", "dome", "grass"),
            ("ATL", "Mercedes-Benz Stadium", "US/Eastern", "dome", "turf"),
            ("BAL", "M&T Bank Stadium", "US/Eastern", "outdoor", "grass"),
            ("BUF", "Highmark Stadium", "US/Eastern", "outdoor", "turf"),
            ("CAR", "Bank of America Stadium", "US/Eastern", "outdoor", "grass"),
            ("CHI", "Soldier Field", "US/Central", "outdoor", "grass"),
            ("CIN", "Paycor Stadium", "US/Eastern", "outdoor", "turf"),
            ("CLE", "Cleveland Browns Stadium", "US/Eastern", "outdoor", "grass"),
            ("DAL", "AT&T Stadium", "US/Central", "dome", "turf"),
            ("DEN", "Empower Field at Mile High", "US/Mountain", "outdoor", "grass"),
            ("DET", "Ford Field", "US/Eastern", "dome", "turf"),
            ("GB", "Lambeau Field", "US/Central", "outdoor", "grass"),
            ("HOU", "NRG Stadium", "US/Central", "dome", "turf"),
            ("IND", "Lucas Oil Stadium", "US/Eastern", "dome", "turf"),
            ("JAX", "EverBank Stadium", "US/Eastern", "outdoor", "turf"),
            ("KC", "GEHA Field at Arrowhead", "US/Central", "outdoor", "grass"),
            ("LAC", "SoFi Stadium", "US/Pacific", "dome", "turf"),
            ("LAR", "SoFi Stadium", "US/Pacific", "dome", "turf"),
            ("LV", "Allegiant Stadium", "US/Pacific", "dome", "turf"),
            ("MIA", "Hard Rock Stadium", "US/Eastern", "outdoor", "grass"),
            ("MIN", "U.S. Bank Stadium", "US/Central", "dome", "turf"),
            ("NE", "Gillette Stadium", "US/Eastern", "outdoor", "turf"),
            ("NO", "Caesars Superdome", "US/Central", "dome", "turf"),
            ("NYG", "MetLife Stadium", "US/Eastern", "outdoor", "turf"),
            ("NYJ", "MetLife Stadium", "US/Eastern", "outdoor", "turf"),
            ("PHI", "Lincoln Financial Field", "US/Eastern", "outdoor", "grass"),
            ("PIT", "Acrisure Stadium", "US/Eastern", "outdoor", "turf"),
            ("SEA", "Lumen Field", "US/Pacific", "outdoor", "turf"),
            ("SF", "Levi's Stadium", "US/Pacific", "outdoor", "grass"),
            ("TB", "Raymond James Stadium", "US/Eastern", "outdoor", "grass"),
            ("TEN", "Nissan Stadium", "US/Central", "outdoor", "grass"),
            ("WAS", "Commanders Field", "US/Eastern", "outdoor", "grass"),
        ],
        columns=["home", "stadium", "tz", "roof", "surface"],
    )

    out = out.merge(stadium_meta, on="home", how="left")

    column_order = [
        "season",
        "week",
        "home",
        "away",
        "kickoff_utc",
        "stadium",
        "tz",
        "roof",
        "surface",
        "game_id",
    ]
    out = out[[c for c in column_order if c in out.columns]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[make_team_week_map] wrote {len(out)} rows → {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--out", type=Path, default=TEAM_WEEK_PATH)
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help="Optional local schedule CSV to use instead of downloading.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tw = build_map(args.season, schedule_path=args.schedule)
    tw.to_csv(args.out, index=False)
    print(f"[make_team_week_map] wrote {len(tw)} rows → {args.out}")
    _write_game_lines_from_team_week_map(tw, out_path=GAME_LINES_PATH)


if __name__ == "__main__":
    main()
