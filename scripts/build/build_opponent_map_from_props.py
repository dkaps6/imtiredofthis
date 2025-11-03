#!/usr/bin/env python3
import sys
from pathlib import Path

import difflib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _normalize_team(val: str | None) -> str | None:
    """
    Normalize team strings (abbr/casing/spacing) so cross-source joins succeed.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper().replace(" ", "")
    alias = {
        "WSH": "WAS",
        "JAX": "JAC",
        "LAR": "LA",
    }
    return alias.get(s, s)


def _infer_opponent_row(row: pd.Series) -> str | None:
    """
    Given a row that contains team_norm, home_team_norm, away_team_norm,
    return the opponent abbr if possible, else None.
    """
    tn = row.get("team_norm")
    h = row.get("home_team_norm")
    a = row.get("away_team_norm")
    if tn and h and a:
        return a if tn == h else (h if tn == a else None)
    return None


def _fallback_match_team(row: pd.Series, odds_teams: pd.DataFrame) -> dict | None:
    """
    Pick a best-guess game for props rows missing event_id by matching team_norm
    against either home_team_norm or away_team_norm. Uses exact then fuzzy match.
    Returns a dict of matched odds fields (or None).
    """
    tn = row.get("team_norm")
    if not tn:
        return None

    exact = odds_teams[
        (odds_teams["home_team_norm"] == tn) | (odds_teams["away_team_norm"] == tn)
    ]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    all_teams = list(
        set(odds_teams["home_team_norm"]) | set(odds_teams["away_team_norm"])
    )
    close = difflib.get_close_matches(tn, all_teams, n=1)
    if close:
        fuzzy = odds_teams[
            (odds_teams["home_team_norm"] == close[0])
            | (odds_teams["away_team_norm"] == close[0])
        ]
        if not fuzzy.empty:
            return fuzzy.iloc[0].to_dict()
    return None


def build_opponent_map(
    props: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
    props_path: str = "data/props_raw.csv",
    odds_path: str = "data/odds_game.csv",
    out_path: str = "data/opponent_map_from_props.csv",
) -> pd.DataFrame:
    """
    Build a dense mapping of (player, team, opponent, event_id, season, week).
    - Primary join on event_id
    - Fallback join on normalized team when event_id missing
    - Infers opponent from home/away when possible
    """

    def _safe_read(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    if props is None or props.empty:
        props = _safe_read(props_path)
    if odds is None or odds.empty:
        odds = _safe_read(odds_path)

    props = props.copy() if props is not None else pd.DataFrame()
    odds = odds.copy() if odds is not None else pd.DataFrame()

    if props.empty:
        props = pd.DataFrame(columns=["player", "team", "event_id"])
    if odds.empty:
        odds = pd.DataFrame(columns=["event_id", "home_team", "away_team", "week", "season"])

    team_source = None
    for cand in ["team", "player_team_abbr", "team_abbr", "player_team"]:
        if cand in props.columns:
            team_source = props[cand]
            break
    if team_source is None:
        team_source = pd.Series([None] * len(props), index=props.index)
    props["team_norm"] = team_source.apply(_normalize_team)

    def _clean_event_id(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return np.nan
        sval = str(val).strip()
        return sval if sval else np.nan

    if "event_id" in props.columns:
        props["event_id"] = props["event_id"].apply(_clean_event_id)
    else:
        props["event_id"] = np.nan

    odds_home = odds.get(
        "home_team", pd.Series([None] * len(odds), index=odds.index)
    )
    odds_away = odds.get(
        "away_team", pd.Series([None] * len(odds), index=odds.index)
    )
    odds["home_team_norm"] = odds_home.apply(_normalize_team)
    odds["away_team_norm"] = odds_away.apply(_normalize_team)

    if "event_id" in odds.columns:
        odds["event_id"] = odds["event_id"].apply(_clean_event_id)
    else:
        odds["event_id"] = np.nan

    ts_col = next(
        (
            col
            for col in ["kickoff_ts", "commence_time", "game_timestamp"]
            if col in odds.columns
        ),
        None,
    )
    if ts_col:
        odds["game_timestamp"] = pd.to_datetime(odds[ts_col], errors="coerce")

    if "week" in odds.columns:
        odds["week"] = pd.to_numeric(odds["week"], errors="coerce")
    if "season" in odds.columns:
        odds["season"] = pd.to_numeric(odds["season"], errors="coerce")

    merged = props.merge(odds, how="left", on="event_id", suffixes=("", "_odds"))

    missing_evt = merged[merged["event_id"].isna()].copy()
    if not missing_evt.empty and not odds.empty:
        base_cols = [
            "home_team_norm",
            "away_team_norm",
            "week",
            "season",
            "game_timestamp",
            "event_id",
        ]
        odds_cols = [col for col in base_cols if col in odds.columns]
        odds_min = odds[odds_cols].drop_duplicates()

        matched = missing_evt.apply(
            lambda r: _fallback_match_team(r, odds_min), axis=1
        ).apply(pd.Series)
        for col in [
            "home_team_norm",
            "away_team_norm",
            "week",
            "season",
            "game_timestamp",
            "event_id",
        ]:
            if col in missing_evt.columns:
                missing_evt[col] = missing_evt[col].fillna(matched.get(col))
            else:
                missing_evt[col] = matched.get(col)

        merged = pd.concat(
            [merged[~merged["event_id"].isna()], missing_evt], ignore_index=True
        )

    merged["opponent"] = merged.apply(_infer_opponent_row, axis=1)

    player_col = (
        "player"
        if "player" in merged.columns
        else ("player_name" if "player_name" in merged.columns else None)
    )
    if player_col is None:
        merged["player"] = ""
        player_col = "player"

    keep = [
        player_col,
        "team_norm",
        "opponent",
        "event_id",
        "season",
        "week",
        "game_timestamp",
    ]

    for col in keep:
        if col not in merged.columns:
            merged[col] = np.nan

    out = merged[keep].rename(columns={player_col: "player"})
    out = out.drop_duplicates()
    out = out.rename(columns={"team_norm": "team"})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[opponent_map] wrote {len(out):,} rows -> {out_path}")
    missing_opp = out["opponent"].isna().sum()
    if missing_opp:
        print(
            f"[opponent_map] WARNING: {missing_opp} rows still missing opponent (could not infer)"
        )
    return out


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

    props_path_candidates = [
        Path("data") / "props_raw.csv",
        Path("outputs") / "props_raw.csv",
    ]
    props_df = pd.DataFrame()
    props_path_str = None
    for cand in props_path_candidates:
        if cand.exists():
            props_path_str = str(cand)
            try:
                props_df = pd.read_csv(cand)
            except Exception as exc:
                print(
                    "[build_opponent_map_from_props] failed to load props file "
                    f"{cand}: {exc}"
                )
                props_df = pd.DataFrame()
            break
    if props_path_str is None:
        props_path_str = str(props_path_candidates[0])

    odds_candidates = [
        Path("data") / "odds_game.csv",
        Path("outputs") / "odds_game.csv",
    ]
    odds_df = pd.DataFrame()
    odds_path_str = None
    for cand in odds_candidates:
        if cand.exists():
            odds_path_str = str(cand)
            try:
                odds_df = pd.read_csv(cand)
            except Exception as exc:
                print(
                    "[build_opponent_map_from_props] failed to load odds file "
                    f"{cand}: {exc}"
                )
                odds_df = pd.DataFrame()
            break
    if odds_path_str is None:
        odds_path_str = str(odds_candidates[0])

    out_path = Path("data") / "opponent_map_from_props.csv"
    build_opponent_map(
        props=props_df if not props_df.empty else None,
        odds=odds_df if not odds_df.empty else None,
        props_path=props_path_str,
        odds_path=odds_path_str,
        out_path=str(out_path),
    )


if __name__ == "__main__":
    main()
