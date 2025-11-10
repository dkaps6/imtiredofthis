import glob
import os
from typing import Optional, Tuple

import pandas as pd

TEAM_FIXES = {
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "LA": "LAR",
    "ARZ": "ARI",
    "WSH": "WAS",
}

WEEK_FIX_KEYS = ["week", "game_week"]


def _canon_team(x: str) -> str:
    if not isinstance(x, str):
        return x
    x = x.strip().upper()
    return TEAM_FIXES.get(x, x)


def _infer_week_col(df: pd.DataFrame) -> str:
    for col in WEEK_FIX_KEYS:
        if col in df.columns:
            return col
    return "week"


def _coerce_datetime(col: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(col, errors="coerce", utc=True)
    except Exception:
        return pd.Series(pd.NaT, index=col.index if hasattr(col, "index") else None)


def _write(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)


def _try_local_candidates(season: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    candidates = [
        f"data/schedule_{season}.csv",
        "data/schedule.csv",
        *sorted(glob.glob("data/*schedule*.csv"), reverse=True),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "season" in df.columns:
            try:
                mask = df["season"].astype(int) == int(season)
                df = df.loc[mask].copy()
            except Exception:
                df = df.copy()
        if not df.empty:
            return df, path
    return None, None


def _download_nflverse(season: int) -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/schedules.csv.gz"
    df = pd.read_csv(url, compression="gzip", low_memory=False)
    df = df[df["season"].astype(int) == int(season)].copy()

    df.rename(columns={"home_team": "home_team", "away_team": "away_team", "game_id": "event_id"}, inplace=True)

    if "start_time" in df.columns:
        kickoff = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    elif {"game_date", "game_time"}.issubset(df.columns):
        kickoff = pd.to_datetime(df["game_date"] + " " + df["game_time"], errors="coerce", utc=True)
    else:
        kickoff = pd.Series(pd.NaT, index=df.index)

    df["kickoff_utc"] = kickoff
    week_col = _infer_week_col(df)

    out = df.assign(
        week=df[week_col].astype(int),
        season=int(season),
        home_team=df["home_team"].map(_canon_team),
        away_team=df["away_team"].map(_canon_team),
    )[["season", "week", "home_team", "away_team", "kickoff_utc", "event_id"]].drop_duplicates()

    return out


def build_or_get_schedule(season: int, out_path: str = "data/schedule.csv") -> str:
    df, src = _try_local_candidates(season)
    if df is None:
        df = _download_nflverse(season)
        src = "nflverse_remote"

    required = {"season", "week", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Schedule missing columns {missing}")

    df = df.copy()
    df["home_team"] = df["home_team"].map(_canon_team)
    df["away_team"] = df["away_team"].map(_canon_team)
    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = _coerce_datetime(df["kickoff_utc"])

    _write(df, out_path)
    print(f"[build_schedule] Wrote schedule ({len(df)} rows) to {out_path} (source={src})")
    return out_path


__all__ = ["build_or_get_schedule"]
