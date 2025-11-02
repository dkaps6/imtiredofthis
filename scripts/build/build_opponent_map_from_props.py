#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

import numpy as np
import pandas as pd
from nfl_data_py import import_schedules

VALID = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
    "PHI","PIT","SEA","SF","TB","TEN","WAS"
}

TEAM_NAME_TO_ABBR = {
    "ARI":"ARI","ARZ":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR",
    "CHI":"CHI","CIN":"CIN","CLE":"CLE","DAL":"DAL","DEN":"DEN","DET":"DET",
    "GB":"GB","GNB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","JAC":"JAX",
    "KC":"KC","KCC":"KC","LAC":"LAC","LAR":"LAR","LA":"LAR","LV":"LV","OAK":"LV",
    "LAS":"LV","MIA":"MIA","MIN":"MIN","NE":"NE","NWE":"NE","NO":"NO","NOR":"NO",
    "NYG":"NYG","NYJ":"NYJ","PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF",
    "SFO":"SF","TB":"TB","TAM":"TB","TEN":"TEN","WAS":"WAS","WSH":"WAS","WFT":"WAS",
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL",
    "BUFFALO BILLS":"BUF","CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI",
    "CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE","DALLAS COWBOYS":"DAL",
    "DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX",
    "KANSAS CITY CHIEFS":"KC","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR",
    "LAS VEGAS RAIDERS":"LV","MIAMI DOLPHINS":"MIA","MINNESOTA VIKINGS":"MIN",
    "NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
    "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT",
    "SEATTLE SEAHAWKS":"SEA","SAN FRANCISCO 49ERS":"SF","TAMPA BAY BUCCANEERS":"TB",
    "TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS","WASHINGTON FOOTBALL TEAM":"WAS",
    "ARIZONA":"ARI","CARDINALS":"ARI","ATLANTA":"ATL","FALCONS":"ATL","BALTIMORE":"BAL",
    "RAVENS":"BAL","BUFFALO":"BUF","BILLS":"BUF","CAROLINA":"CAR","PANTHERS":"CAR",
    "CHICAGO":"CHI","BEARS":"CHI","CINCINNATI":"CIN","BENGALS":"CIN","CLEVELAND":"CLE",
    "BROWNS":"CLE","DALLAS":"DAL","COWBOYS":"DAL","DENVER":"DEN","BRONCOS":"DEN",
    "DETROIT":"DET","LIONS":"DET","GREEN BAY":"GB","PACKERS":"GB","HOUSTON":"HOU",
    "TEXANS":"HOU","INDIANAPOLIS":"IND","COLTS":"IND","JACKSONVILLE":"JAX",
    "JAGUARS":"JAX","KANSAS CITY":"KC","CHIEFS":"KC","CHARGERS":"LAC","RAMS":"LAR",
    "LOS ANGELES":"LAR","LAS VEGAS":"LV","RAIDERS":"LV","MIAMI":"MIA","DOLPHINS":"MIA",
    "MINNESOTA":"MIN","VIKINGS":"MIN","NEW ENGLAND":"NE","PATRIOTS":"NE",
    "NEW ORLEANS":"NO","SAINTS":"NO","GIANTS":"NYG","JETS":"NYJ","PHILADELPHIA":"PHI",
    "EAGLES":"PHI","PITTSBURGH":"PIT","STEELERS":"PIT","SEATTLE":"SEA",
    "SEAHAWKS":"SEA","SAN FRANCISCO":"SF","49ERS":"SF","TAMPA BAY":"TB",
    "BUCCANEERS":"TB","TENNESSEE":"TEN","TITANS":"TEN","WASHINGTON":"WAS",
    "COMMANDERS":"WAS"
}
TEAM_NAME_TO_ABBR.update({k.lower(): v for k, v in TEAM_NAME_TO_ABBR.items()})

import re as _re_nh
import unicodedata as _ud_nh

_SUFFIX_RE_NH = _re_nh.compile(r"\b(JR|SR|II|III|IV|V)\b\.?", _re_nh.IGNORECASE)
_LEADING_NUM_RE_NH = _re_nh.compile(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", _re_nh.UNICODE)


def _deaccent_nh(s: str) -> str:
    try:
        return _ud_nh.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    except Exception:
        return s


def _clean_person_name_nh(s: str) -> str:
    s = (s or "").replace("\xa0", " ").strip()
    s = _LEADING_NUM_RE_NH.sub("", s)
    s = s.replace(".", " ")
    s = _SUFFIX_RE_NH.sub("", s)
    s = _re_nh.sub(r"\s+", " ", s)
    s = _deaccent_nh(s)
    return s.strip()


def _player_key_from_name_nh(s: str) -> str:
    s = _clean_person_name_nh(s)
    return _re_nh.sub(r"[^a-z0-9]", "", s.lower())


def _canon_team(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s in TEAM_NAME_TO_ABBR:
        abbr = TEAM_NAME_TO_ABBR[s]
        return abbr if abbr in VALID else ""
    s2 = _re_nh.sub(r"[^A-Z0-9 ]+", "", s).strip()
    if s2 in TEAM_NAME_TO_ABBR:
        abbr = TEAM_NAME_TO_ABBR[s2]
        return abbr if abbr in VALID else ""
    return ""


def build_name_canonical_map_from_inputs(df_list):
    """
    df_list: list of DataFrames that have at least ['team','player'].
    Returns dict[team_abbr][name_variant_lower] = canonical_full_name
    """

    rows = []
    for df in df_list:
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            team = str(r.get("team", "")).strip().upper()
            full = str(r.get("player", "")).strip()
            if not team or not full:
                continue
            parts = full.split()
            if len(parts) < 2:
                continue
            first = parts[0]
            last = parts[-1]
            key_full = f"{first} {last}".lower()
            key_init_dot = f"{first[0]}. {last}".lower()
            key_init_nodot = f"{first[0]} {last}".lower()
            if team not in VALID:
                continue
            rows.append((team, full, key_full, key_init_dot, key_init_nodot))
    out = {}
    for team, full, k1, k2, k3 in rows:
        out.setdefault(team, {})
        out[team][k1] = full
        out[team][k2] = full
        out[team][k3] = full
    return out


def apply_name_canonicalization_local(df, canon_map):
    df = df.copy()
    for idx, row in df.iterrows():
        t = str(row.get("team", "")).strip().upper()
        p = str(row.get("player", "")).strip()
        if not t or not p:
            continue
        key1 = p.lower()
        key2 = key1.replace(".", "")
        fixed = None
        if t in canon_map:
            if key1 in canon_map[t]:
                fixed = canon_map[t][key1]
            elif key2 in canon_map[t]:
                fixed = canon_map[t][key2]
        if fixed:
            df.at[idx, "player"] = fixed
    return df


def get_nfl_schedule(season: int) -> pd.DataFrame:
    df = import_schedules([season])
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["REG"])]
    rename_map = {
        "home_team": "team_home",
        "away_team": "team_away",
        "game_date": "gameday",
        "venue": "stadium",
        "site_city": "location",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    required = ["week", "team_home", "team_away"]
    missing = [c for c in required if c not in df.columns]
    if missing or df.empty:
        raise RuntimeError(
            f"Schedule missing columns {missing} or empty for season {season}"
        )
    keep = [
        "week",
        "team_home",
        "team_away",
    ] + [c for c in ["game_id", "gameday", "stadium", "location"] if c in df.columns]
    return df[keep].reset_index(drop=True)


def expand_to_team_opp(schedule: pd.DataFrame) -> pd.DataFrame:
    """Expand schedule rows to team/opponent rows using nflverse data."""

    if schedule is None or schedule.empty:
        raise RuntimeError("nflverse schedule returned no games to expand")

    home_col = next(
        (col for col in ["team_home", "home_team", "home"] if col in schedule.columns),
        None,
    )
    away_col = next(
        (col for col in ["team_away", "away_team", "away"] if col in schedule.columns),
        None,
    )
    if not home_col or not away_col:
        raise RuntimeError("nflverse schedule is missing home/away team columns")

    if "week" not in schedule.columns:
        raise RuntimeError("nflverse schedule is missing the 'week' column")

    timestamp_col = next(
        (
            col
            for col in [
                "gameday",
                "gamedate",
                "game_day",
                "game_date",
                "kickoff",
                "game_time",
                "gametime",
            ]
            if col in schedule.columns
        ),
        None,
    )

    games = schedule[[away_col, home_col, "week", "season"]].copy()
    games.rename(columns={away_col: "away_abbr", home_col: "home_abbr"}, inplace=True)

    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    games = games.dropna(subset=["week"]).astype({"week": int})

    if timestamp_col:
        games["game_timestamp"] = schedule.loc[games.index, timestamp_col]
    else:
        games["game_timestamp"] = np.nan

    columns = ["team", "opponent", "week", "season", "game_timestamp"]
    away = games.rename(
        columns={"away_abbr": "team", "home_abbr": "opponent"}
    )[["team", "opponent", "week", "season", "game_timestamp"]]
    home = games.rename(
        columns={"home_abbr": "team", "away_abbr": "opponent"}
    )[["team", "opponent", "week", "season", "game_timestamp"]]

    team_map = pd.concat([away, home], ignore_index=True).drop_duplicates(subset=columns)

    if team_map.empty:
        raise RuntimeError("expanded nflverse schedule produced no team/opponent rows")

    return team_map


def infer_default_season() -> int:
    """Return a reasonable default NFL season based on today's date."""
    today = pd.Timestamp.utcnow()
    return today.year if today.month >= 3 else today.year - 1


def main() -> None:
    # Allow optional season argument (first numeric argument wins)
    season = None
    for arg in sys.argv[1:]:
        try:
            season = int(arg)
            break
        except (TypeError, ValueError):
            continue
    if season is None:
        season = infer_default_season()

    schedule = get_nfl_schedule(season)
    schedule = schedule.copy()
    schedule["season"] = season
    team_map = expand_to_team_opp(schedule)

    roles_path = Path("data") / "roles_ourlads.csv"
    players_df = pd.DataFrame()
    if roles_path.exists():
        try:
            tmp = pd.read_csv(roles_path)
        except Exception as exc:
            print(
                "[build_opponent_map_from_props] failed to load roles_ourlads.csv: "
                f"{exc}"
            )
            tmp = pd.DataFrame()
        if not tmp.empty and {"team", "player"}.issubset(tmp.columns):
            tmp = tmp[["team", "player"]].dropna(subset=["team", "player"]).copy()
            tmp["team"] = tmp["team"].astype(str).str.upper().str.strip().map(_canon_team)
            tmp = tmp[tmp["team"].isin(VALID)]
            tmp["player"] = tmp["player"].astype(str).str.strip()
            players_df = tmp.drop_duplicates()

    if players_df.empty:
        players_df = players_df.reindex(columns=["team", "player"])

    team_map = team_map.copy()
    team_map["team"] = team_map["team"].astype(str).str.upper().str.strip().map(_canon_team)
    team_map["opponent"] = team_map["opponent"].astype(str).str.upper().str.strip().map(_canon_team)
    team_map = team_map[
        team_map["team"].isin(VALID) & team_map["opponent"].isin(VALID)
    ]

    merged = team_map.merge(players_df, on="team", how="left")

    canon_map = build_name_canonical_map_from_inputs([merged])
    merged = apply_name_canonicalization_local(merged, canon_map)

    merged["player_key"] = merged["player"].astype(str).map(_player_key_from_name_nh)
    merged["team_key"] = merged["team"].astype(str).str.upper().str.strip()

    out_df = merged.loc[
        merged["player"].notna()
        & (merged["player"].astype(str).str.strip() != "")
        & merged["team"].notna()
        & (merged["team"].astype(str).str.strip() != ""),
        [
            "player",
            "team",
            "opponent",
            "week",
            "season",
            "game_timestamp",
            "player_key",
            "team_key",
        ],
    ].copy()

    if "week" not in out_df.columns or out_df["week"].isna().all():
        raise RuntimeError(
            "[build_opponent_map_from_props] no usable week mapping for any player"
        )

    out_path = Path(os.path.join("data", "opponent_map_from_props.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(
        f"[build_opponent_map_from_props] wrote {len(out_df)} rows → {out_path}"
    )

    debug_dir = Path(os.path.join("data", "_debug"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_df.head(50).to_csv(debug_dir / "opponent_sample.csv", index=False)
    print(
        "[build_opponent_map_from_props] wrote debug sample → "
        f"{debug_dir / 'opponent_sample.csv'}"
    )


if __name__ == "__main__":
    main()
