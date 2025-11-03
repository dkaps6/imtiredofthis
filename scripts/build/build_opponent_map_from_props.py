import argparse
from pathlib import Path

import pandas as pd

from scripts.utils.name_clean import canonical_player

DATA = Path("data")

# --- RESTORED, LIGHTWEIGHT FALLBACK ---
TEAM_NAME_TO_ABBR = {
    "ARIZONA CARDINALS": "ARI", "ATLANTA FALCONS": "ATL", "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF", "CAROLINA PANTHERS": "CAR", "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN", "CLEVELAND BROWNS": "CLE", "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN", "DETROIT LIONS": "DET", "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU", "INDIANAPOLIS COLTS": "IND", "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC", "LAS VEGAS RAIDERS": "LV", "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR", "MIAMI DOLPHINS": "MIA", "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE", "NEW ORLEANS SAINTS": "NO", "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ", "PHILADELPHIA EAGLES": "PHI", "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF", "SEATTLE SEAHAWKS": "SEA", "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN", "WASHINGTON COMMANDERS": "WAS",
}
TEAM_NAME_TO_ABBR.update({k.lower(): v for k, v in TEAM_NAME_TO_ABBR.items()})


def _norm_team_name(val) -> str:
    if not isinstance(val, str):
        return ""
    raw = val.strip()
    if not raw:
        return ""
    for key in (raw, raw.upper(), raw.lower()):
        if key in TEAM_NAME_TO_ABBR:
            return TEAM_NAME_TO_ABBR[key]
    upper = raw.upper()
    if upper in TEAM_NAME_TO_ABBR.values():
        return upper
    return ""


def _canonicalize_players(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    for cand in ("player_clean_key", "player_key", "player", "player_name"):
        if cand in df.columns:
            df["player_name_raw"] = df[cand]
            break
    else:
        df["player_name_raw"] = ""
    df["player_clean_key"] = df["player_name_raw"].astype(str).map(canonical_player)
    return df


def _ensure_event_id(df: pd.DataFrame) -> pd.DataFrame:
    if "event_id" not in df.columns:
        df["event_id"] = ""
    df["event_id"] = df["event_id"].astype(str)
    return df


def _infer_team_from_props(df: pd.DataFrame) -> pd.Series:
    candidate_cols = [
        c for c in df.columns if c.lower() in {"team", "team_abbr", "player_team", "player_team_abbr", "team_name"}
    ]
    if not candidate_cols:
        return pd.Series(["" for _ in range(len(df))], index=df.index, dtype=object)
    base = df[candidate_cols[0]].fillna("").astype(str)
    mapped = base.map(_norm_team_name)
    fallback = base.str.upper().where(base.str.upper().isin(TEAM_NAME_TO_ABBR.values()), "")
    return mapped.where(mapped.ne(""), fallback)


def build_opponent_map(
    props_path: str = "data/props_raw.csv",
    odds_path: str = "data/odds_game.csv",
    out_path: str = "data/opponent_map_from_props.csv",
    season: int | None = None,
    week: str | None = None,
) -> pd.DataFrame:
    props = pd.read_csv(props_path)
    odds = pd.read_csv(odds_path)

    if props.empty or odds.empty:
        DATA.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["player", "team", "opponent", "event_id"])
        empty.to_csv(out_path, index=False)
        print("[opponent_map] wrote data/opponent_map_from_props.csv rows=0")
        return empty

    props = _canonicalize_players(props)
    props = _ensure_event_id(props)
    odds = _ensure_event_id(odds.copy())

    join_cols = [c for c in ["event_id", "home_team", "away_team", "season", "week", "commence_time"] if c in odds.columns]
    joined = props.merge(odds[join_cols], on="event_id", how="left", suffixes=("", "_odds"))

    for col in ("home_team", "away_team"):
        if col in joined.columns:
            joined[f"{col}_abbr"] = joined[col].map(_norm_team_name)
            fallback = joined[col].astype(str).str.upper()
            joined[f"{col}_abbr"] = joined[f"{col}_abbr"].where(
                joined[f"{col}_abbr"].ne(""), fallback.where(fallback.isin(TEAM_NAME_TO_ABBR.values()), "")
            )
        else:
            joined[f"{col}_abbr"] = ""

    joined["team_abbr"] = _infer_team_from_props(joined)

    def infer_team(row):
        team = row.get("team_abbr", "")
        if team:
            return team
        for side in ("home_team_abbr", "away_team_abbr"):
            side_val = row.get(side, "")
            if side_val:
                return side_val
        return ""

    joined["team_abbr"] = joined.apply(infer_team, axis=1)

    def infer_opp(row):
        team = row.get("team_abbr", "")
        home = row.get("home_team_abbr", "")
        away = row.get("away_team_abbr", "")
        if team and home and away:
            if team == home:
                return away
            if team == away:
                return home
        return ""

    joined["opponent_abbr"] = joined.apply(infer_opp, axis=1)

    timestamp_col = next(
        (c for c in ["market_timestamp", "timestamp", "last_update", "created_at", "updated_at"] if c in joined.columns),
        None,
    )
    if timestamp_col:
        joined["_ts"] = pd.to_datetime(joined[timestamp_col], errors="coerce")

    keep_cols = [
        c
        for c in [
            "season",
            "week",
            "event_id",
            "commence_time",
            "player_name_raw",
            "player_clean_key",
            "team_abbr",
            "opponent_abbr",
            "_ts",
        ]
        if c in joined.columns
    ]
    out = joined[keep_cols].copy()

    for col in ("team_abbr", "opponent_abbr"):
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str).str.upper()

    if "season" not in out.columns:
        out["season"] = pd.NA
    if "week" not in out.columns:
        out["week"] = pd.NA

    dedup_subset = [c for c in ["season", "week", "player_clean_key"] if c in out.columns]
    if dedup_subset:
        sort_cols = dedup_subset + (["_ts"] if "_ts" in out.columns else [])
        out = out.sort_values(sort_cols)
        out = out.drop_duplicates(subset=dedup_subset, keep="last")
    if "_ts" in out.columns:
        out = out.drop(columns=["_ts"])

    resolved_season = season
    if resolved_season is None and "season" in out.columns:
        season_series = pd.to_numeric(out["season"], errors="coerce")
        if season_series.notna().any():
            try:
                resolved_season = int(season_series.dropna().iloc[-1])
            except Exception:
                resolved_season = None
    if resolved_season is not None:
        out["season"] = int(resolved_season)
    else:
        out["season"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    resolved_week = week
    if resolved_week is None and "week" in out.columns:
        wk_series = out["week"]
        if not wk_series.dropna().empty:
            resolved_week = wk_series.dropna().astype(str).iloc[-1]
    if resolved_week is None:
        out["week"] = ""
    else:
        out["week"] = str(resolved_week)
    out["week"] = out["week"].astype("string")

    if "commence_time" in out.columns:
        ts = pd.to_datetime(out["commence_time"], errors="coerce", utc=True)
    else:
        ts = pd.Series(pd.NaT, index=out.index)
    out["game_timestamp"] = ts.astype("string")

    required = {
        "player": "player",
        "player_name": "player",
        "player_name_raw": "player",
        "name": "player",
        "team": "team",
        "team_abbr": "team",
        "player_team_abbr": "team",
        "opponent": "opponent",
        "opponent_abbr": "opponent",
        "opponent_team_abbr": "opponent",
        "event_id": "event_id",
        "props_event_id": "event_id",
    }
    renamed = {c: required[c] for c in out.columns if c in required}
    out = out.rename(columns=renamed)

    for col in ["player", "team", "opponent"]:
        if col not in out.columns:
            out[col] = pd.NA

    out["player"] = out["player"].fillna("").astype(str).str.strip()
    out["team"] = out["team"].fillna("").astype(str).str.upper().str.strip()
    out["opponent"] = out["opponent"].fillna("").astype(str).str.upper().str.strip()

    keep = [c for c in ["player", "team", "opponent", "event_id"] if c in out.columns]
    out = out[keep]
    out = out[out["player"].astype(str).str.strip() != ""]
    out = out.drop_duplicates()

    DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[opponent_map] wrote data/opponent_map_from_props.csv rows={len(out)}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--props-path", default="data/props_raw.csv")
    parser.add_argument("--odds-path", default="data/odds_game.csv")
    parser.add_argument("--out-path", default="data/opponent_map_from_props.csv")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=str, default=None)
    args = parser.parse_args()

    build_opponent_map(
        props_path=args.props_path,
        odds_path=args.odds_path,
        out_path=args.out_path,
        season=args.season,
        week=args.week,
    )
