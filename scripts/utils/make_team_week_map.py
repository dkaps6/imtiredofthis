#!/usr/bin/env python3
"""Build a team-week opponent map from odds or schedule data."""
from __future__ import annotations

import argparse
import io
import gzip
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Canonicalize team names
from scripts._opponent_map import canon_team
from scripts.utils.name_clean import normalize_team

TEAM_WEEK_MAP_CSV = os.environ.get(
    "TEAM_WEEK_MAP_CSV",
    "data/team_week_map.csv",
)

logger = logging.getLogger(__name__)

SCHED_DIR = Path("data/schedules")
SCHED_DIR.mkdir(parents=True, exist_ok=True)

NFLVERSE_URL = "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/sched_{season}.csv.gz"
# PFR kept as last-resort; often 403s on CI
PFR_URL = "https://www.pro-football-reference.com/years/{season}/games.htm"
ESPN_WEEK_URL = "https://www.espn.com/nfl/schedule/_/week/{week}/year/{season}/seasontype/2"

_EXPECTED_COLS = {"season", "week", "gameday", "game_id", "home_team", "away_team"}

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
}


def canon_team_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(canon_team)


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
    df["home_team"] = canon_team_series(df["home_team"])
    df["away_team"] = canon_team_series(df["away_team"])
    return _validate_schedule(
        df[["season", "week", "gameday", "game_id", "home_team", "away_team"]]
    )


def _fetch_schedule_espn_html(season: int, max_week: int = 19) -> pd.DataFrame:
    """
    Scrape ESPN weekly schedule pages and build a full-season schedule.
    Columns: season, week, gameday, game_id, home_team, away_team
    """
    rows = []
    sess = requests.Session()

    for wk in range(1, max_week + 1):
        url = ESPN_WEEK_URL.format(week=wk, season=season)
        r = sess.get(url, timeout=30)
        if r.status_code != 200:
            continue

        soup = BeautifulSoup(r.text, "lxml")

        # ESPN renders day-grouped tables; parse all matchup rows
        for table in soup.select("table"):
            for tr in table.select("tbody tr"):
                tds = tr.find_all("td")
                if len(tds) < 1:
                    continue

                matchup_txt = " ".join(tds[0].get_text(" ", strip=True).split())
                low = matchup_txt.lower()
                if " at " in low:
                    away_raw, home_raw = re.split(r"\sat\s", matchup_txt, flags=re.I, maxsplit=1)
                    away_raw, home_raw = away_raw.strip(), home_raw.strip()
                elif " vs " in low:
                    # treat left as HOME for 'vs' formatting
                    home_raw, away_raw = re.split(r"\s+vs\s+", matchup_txt, flags=re.I, maxsplit=1)
                    home_raw, away_raw = home_raw.strip(), away_raw.strip()
                else:
                    continue

                # date header (optional)
                gameday = None
                date_header = tr.find_previous("h2")
                if date_header:
                    gameday = date_header.get_text(strip=True)

                # try to capture event id for determinism (optional)
                game_id = None
                a = tr.find("a", href=True)
                if a and "gameId" in a["href"]:
                    m = re.search(r"gameId=(\d+)", a["href"])
                    if m:
                        game_id = m.group(1)

                rows.append(
                    {
                        "season": int(season),
                        "week": int(wk),
                        "gameday": gameday,
                        "game_id": game_id,
                        "home_team": home_raw,
                        "away_team": away_raw,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("ESPN HTML fallback produced empty schedule")

    # Canonicalize to house abbreviations
    df["home_team"] = canon_team_series(df["home_team"])
    df["away_team"] = canon_team_series(df["away_team"])

    # Drop any rows that failed to canonicalize (very rare with our map)
    df = df.dropna(subset=["home_team", "away_team"])
    df = df[df["home_team"] != ""]
    df = df[df["away_team"] != ""]

    return df


def _fetch_schedule_espn_json(
    season: int, seasontype: int = 2, weeks: list[int] | None = None
) -> pd.DataFrame:
    """
    Robust ESPN JSON fallback (no HTML/JS). Pulls per-week scoreboard JSON and
    returns columns: season, week, gameday (UTC date), game_id, home_team, away_team.

    - Uses site.api.espn.com (no cookie required).
    - Canonicalizes team abbreviations via canon_team().
    - Adds small jitter + UA header to be polite.
    """
    if weeks is None:
        # Regular season weeks; adjust if you later want preseason/postseason
        weeks = list(range(1, 19))

    rows: list[dict] = []
    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": ua,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )

    for wk in weeks:
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            f"?week={wk}&year={season}&seasontype={seasontype}"
        )
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                print(
                    f"[team_week_map][ESPN-JSON] week {wk}: HTTP {r.status_code} for {url}"
                )
                time.sleep(0.8 + random.random() * 0.6)
                continue

            payload = r.json()
            events = payload.get("events", []) or []
            print(f"[team_week_map][ESPN-JSON] week {wk}: {len(events)} events")

            for ev in events:
                gid = ev.get("id")
                date_iso = ev.get("date")
                comps = (ev.get("competitions") or [])
                if not comps:
                    continue
                comp = comps[0]
                competitors = comp.get("competitors") or []
                home_abbr = away_abbr = None
                kickoff_str = ""
                kickoff_source = comp.get("date") or date_iso
                if kickoff_source:
                    kickoff_ts = pd.to_datetime(kickoff_source, errors="coerce", utc=True)
                    if pd.isna(kickoff_ts):
                        kickoff_ts = pd.to_datetime(kickoff_source, errors="coerce")
                    if pd.notna(kickoff_ts):
                        if kickoff_ts.tzinfo is None:
                            kickoff_ts = kickoff_ts.tz_localize("UTC")
                        try:
                            kickoff_et = kickoff_ts.tz_convert("America/New_York")
                        except Exception:
                            kickoff_et = kickoff_ts.tz_convert("UTC").tz_convert(
                                "America/New_York"
                            )
                        kickoff_str = kickoff_et.strftime("%Y-%m-%d %H:%M:%S ET")

                for c in competitors:
                    team = (c.get("team") or {})
                    abbr = (
                        team.get("abbreviation")
                        or team.get("shortDisplayName")
                        or team.get("location")
                        or team.get("displayName")
                    )
                    side = c.get("homeAway")
                    if side == "home":
                        home_abbr = abbr
                    elif side == "away":
                        away_abbr = abbr

                if not (home_abbr and away_abbr and gid):
                    # log diagnostic row for debugging
                    print(
                        f"[team_week_map][ESPN-JSON] week {wk}: skipped missing fields "
                        f"(gid={gid}, home={home_abbr}, away={away_abbr})"
                    )
                    continue

                rows.append(
                    {
                        "season": int(season),
                        "week": int(wk),
                        "gameday": date_iso,
                        "game_id": str(gid),
                        "home_team": home_abbr,
                        "away_team": away_abbr,
                        "kickoff_et": kickoff_str,
                    }
                )

            time.sleep(0.8 + random.random() * 0.6)
        except Exception as e:
            print(f"[team_week_map][ESPN-JSON] week {wk}: exception {e}")

    if not rows:
        print("[team_week_map][ESPN-JSON] produced 0 rows")
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "gameday",
                "game_id",
                "home_team",
                "away_team",
                "kickoff_et",
            ]
        )

    df = pd.DataFrame(rows)
    # Canonicalize team codes to match our internal keys
    df["home_team"] = canon_team_series(df["home_team"])
    df["away_team"] = canon_team_series(df["away_team"])
    if "kickoff_et" not in df.columns:
        df["kickoff_et"] = ""
    else:
        df["kickoff_et"] = df["kickoff_et"].fillna("").astype(str)
    # Basic sanity
    df = df.dropna(subset=["home_team", "away_team", "game_id"]).reset_index(drop=True)
    print(f"[team_week_map][ESPN-JSON] final rows: {len(df)}")
    return df


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
    return canon_team_series(out)


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
TEAM_WEEK_PATH = Path(TEAM_WEEK_MAP_CSV)
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


def _load_or_build_schedule_source(season: int, schedule_path: str | None) -> pd.DataFrame:
    # 0) local cache first (explicit or implicit)
    local = schedule_path or f"data/schedules/schedule_{season}.csv"
    try:
        if os.path.exists(local):
            df_local = pd.read_csv(local)
            need = {"season", "week", "home_team", "away_team"}
            if need.issubset(df_local.columns) and not df_local.empty:
                df_local["home_team"] = canon_team_series(df_local["home_team"])
                df_local["away_team"] = canon_team_series(df_local["away_team"])
                df_local = _validate_schedule(df_local)
                print(f"[team_week_map] Using local schedule cache: {local}")
                return df_local
    except Exception as e:
        print(f"[team_week_map] local cache read failed: {e}")

    # 1) nflverse (existing helper)
    try:
        df_nv = _fetch_schedule_nflverse(season)  # your existing function
        if not df_nv.empty:
            print("[team_week_map] nflverse schedule fetched")
            return df_nv
    except Exception as e:
        print(f"[team_week_map] nflverse failed: {e}")

    # 2) PFR (existing helper)
    try:
        df_pfr = _fetch_schedule_pfr(season)  # your existing function
        if not df_pfr.empty:
            print("[team_week_map] PFR schedule fetched")
            return df_pfr
    except Exception as e:
        print(f"[team_week_map] PFR failed: {e}")

    # NEW: ESPN JSON fallback (robust, no HTML parsing)
    try:
        df_espn = _fetch_schedule_espn_json(season=int(season), seasontype=2)
        if not df_espn.empty:
            # Persist debug copy for auditing
            out_json = Path("data/schedules")
            out_json.mkdir(parents=True, exist_ok=True)
            with open(out_json / f"schedule_{season}_espn.json", "w") as fp:
                json.dump(df_espn.to_dict(orient="records"), fp, indent=2)
            print(f"[team_week_map] ESPN JSON fallback produced {len(df_espn)} rows")
            return df_espn.rename(
                columns={
                    "home_team": "home_team",
                    "away_team": "away_team",
                    "game_id": "game_id",
                    "gameday": "gameday",
                }
            )
        else:
            print("[team_week_map] ESPN JSON fallback produced empty schedule")
    except Exception as e:
        print(f"[team_week_map] ESPN JSON fallback failed: {e}")

    # 3) ESPN HTML fallback (new)
    try:
        df_espn = _fetch_schedule_espn_html(season)
        df_espn = _validate_schedule(df_espn)
        os.makedirs("data/schedules", exist_ok=True)
        cache_path = f"data/schedules/schedule_{season}.csv"
        df_espn.to_csv(cache_path, index=False)
        print(f"[team_week_map] ESPN HTML fallback succeeded → cached {cache_path}")
        return df_espn
    except Exception as e:
        print(f"[team_week_map] ESPN HTML failed: {e}")

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
        "kickoff_et",
        "bye",
    ]
    extras = [c for c in ("event_id", "game_id") if c in df.columns]
    df = df.loc[:, [c for c in keep_cols + extras if c in df.columns]].copy()
    df = df.rename(columns={"home_abbr": "home_team_abbr", "away_abbr": "away_team_abbr"})
    df["kickoff_utc"] = pd.to_datetime(df.get("kickoff_utc"), utc=True, errors="coerce")

    df = df.sort_values(["season", "week", "team", "kickoff_utc"], na_position="last")
    before = len(df)
    df = df.drop_duplicates(subset=["season", "week", "team"], keep="first")
    dropped = before - len(df)
    if dropped:
        print(
            f"[make_team_week_map] WARNING: dropped {dropped} duplicate rows (kept first per team/week)"
        )

    df = df.reset_index(drop=True)

    out_path = TEAM_WEEK_MAP_CSV
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        "[team_week_map] wrote %d rows to %s",
        len(df),
        os.path.abspath(out_path),
    )
    print(f"[team_week_map] wrote {out_path} with {len(df)} rows; sample:")
    print(df.head(5).to_string(index=False))

    return df


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
    team_cols = []
    if "home" in tw.columns:
        team_cols.append(tw["home"])
    elif "home_team" in tw.columns:
        team_cols.append(tw["home_team"])
    if "away" in tw.columns:
        team_cols.append(tw["away"])
    elif "away_team" in tw.columns:
        team_cols.append(tw["away_team"])
    if team_cols:
        team_concat = pd.concat(team_cols, ignore_index=True).dropna()
        team_count = pd.unique(team_concat).size
    else:
        team_count = 0
    print(
        f"[team_week_map] rows={len(tw)} "
        f"weeks={tw['week'].nunique()} "
        f"teams={team_count}"
    )
    _write_game_lines_from_team_week_map(tw, out_path=GAME_LINES_PATH)


if __name__ == "__main__":
    main()
