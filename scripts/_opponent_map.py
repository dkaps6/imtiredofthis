# scripts/_opponent_map.py
import logging
import os
from typing import Iterable, Optional, Union

import pandas as pd

logger = logging.getLogger("opponent_map")

# Normalize many site/book/team variants into model's canonical 2–3 letter codes.
TEAM_ALIASES = {
    # user-specified weird site codes
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    # common alternates
    "ARZ": "ARI",
    "JAC": "JAX",
    "JAX": "JAX",
    "LA": "LAR",
    "STL": "LAR",
    "OAK": "LV",
    "WSH": "WAS",
    "WFT": "WAS",
    # Expanded aliases from legacy pipelines
    "BAL RAVENS": "BAL",
    "CLEVELAND BROWNS": "CLE",
    "HOUSTON TEXANS": "HOU",
    "COMMANDERS": "WAS",
    "NEP": "NE",
    "N.E.": "NE",
    "GNB": "GB",
    "G.B.": "GB",
    "SFO": "SF",
    "S.F.": "SF",
    "KCC": "KC",
    "K.C.": "KC",
    "SD": "LAC",
    "S.D.": "LAC",
    "LA CHARGERS": "LAC",
    "LA RAMS": "LAR",
    "LAR": "LAR",
    "N.O.": "NO",
    "NOR": "NO",
    "NOS": "NO",
    "T.B.": "TB",
    "TAM": "TB",
    "N.Y. JETS": "NYJ",
    "NY JETS": "NYJ",
    "N.Y. GIANTS": "NYG",
    "NY GIANTS": "NYG",
}


def _canon_team_str(s: str) -> str | None:
    if not s:
        return None
    t = str(s).strip().upper()
    if t == "" or t in {"NA", "NONE"}:
        return None
    t = TEAM_ALIASES.get(t, t)
    if t in {"NAN", "<NA>"}:
        return None
    return TEAM_ALIASES.get(t, t)


def normalize_team(
    x: Union[pd.Series, list, tuple, str, None]
) -> Union[pd.Series, str, None]:
    """
    Vector-safe normalizer. If x is a Series/array-like, return a Series with
    alias mapping + trim + upper. If x is a scalar, do the same and return str|None.
    """

    if isinstance(x, pd.Series):
        series = x.astype("string").fillna("").map(_canon_team_str)
        return series.astype("string")
    if isinstance(x, (list, tuple)):
        s = pd.Series(list(x), dtype="string").fillna("").map(_canon_team_str)
        return s.astype("string")
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, pd.api.extensions.ExtensionArray):
        s = pd.Series(x).astype("string").fillna("").map(_canon_team_str)
        return s.astype("string")
    if pd.isna(x):  # type: ignore[arg-type]
        return None
    return _canon_team_str(x)


def map_normalize_team(series: pd.Series) -> pd.Series:
    """Backwards-compat wrapper: expects a Series, returns Series."""

    normalized = normalize_team(series)
    if isinstance(normalized, pd.Series):
        return normalized.astype("string")
    return pd.Series(normalized, index=series.index, dtype="string")


def map_normalize_opponent_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize any opponent/team columns we use in props/opponent mapping.
    Expected optional columns: 'team', 'team_abbr', 'opponent', 'opponent_abbr', 'home_team', 'away_team'
    """

    out = df.copy()
    for col in (
        "team",
        "team_abbr",
        "opponent",
        "opponent_abbr",
        "home_team",
        "away_team",
    ):
        if col in out.columns:
            out[col] = map_normalize_team(out[col].astype("string"))
    return out

CANON_SET = set([
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
    "IND","JAX","KC","LV","LAC","LA","LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
])


def normalize_team_series(vals: Union[pd.Series, Iterable]) -> pd.Series:
    if isinstance(vals, pd.Series):
        return map_normalize_team(vals)
    return map_normalize_team(pd.Series(list(vals), dtype="string"))

def build_opponent_map(week: Optional[int] = 10, team_map_path: str = "data/team_week_map.csv") -> pd.DataFrame:
    """
    Build symmetric opponent pairs from team_week_map.csv, week-filtered.
    Writes data/opponent_map.csv for downstream joins and returns the df.
    """
    try:
        tm = pd.read_csv(team_map_path)
    except FileNotFoundError:
        logger.warning("[OpponentMap] %s not found", team_map_path)
        return pd.DataFrame(columns=["event_id","week","team","opponent"])
    except Exception as exc:
        logger.error("[OpponentMap] Failed to read %s: %s", team_map_path, exc)
        return pd.DataFrame(columns=["event_id","week","team","opponent"])

    required = {"event_id","week","team","opponent"}
    if not required.issubset(tm.columns):
        missing = ", ".join(sorted(required - set(tm.columns)))
        logger.error("[OpponentMap] team_week_map missing columns: %s", missing)
        return pd.DataFrame(columns=list(required))

    working = tm.copy()
    working["team"] = map_normalize_team(working["team"].astype("string"))
    working["opponent"] = working["opponent"].astype("string")
    bye_mask = working["opponent"].str.upper().str.strip().eq("BYE")
    normalized_opponent = map_normalize_team(working.loc[~bye_mask, "opponent"].astype("string"))
    working.loc[~bye_mask, "opponent"] = normalized_opponent
    working.loc[bye_mask, "opponent"] = "BYE"

    if week is not None:
        working = working[working["week"] == week]

    out_rows = []
    for _, r in working.iterrows():
        evt = str(r.get("event_id","")).strip()
        wk = r.get("week")
        t = r.get("team","")
        opp = r.get("opponent","")
        if not t:
            continue
        if opp == "BYE":
            out_rows.append({"event_id": evt, "week": wk, "team": t, "opponent": "BYE"})
        elif opp:
            out_rows.append({"event_id": evt, "week": wk, "team": t, "opponent": opp})
            out_rows.append({"event_id": evt, "week": wk, "team": opp, "opponent": t})

    df = pd.DataFrame(out_rows).drop_duplicates(subset=["event_id","team","opponent"])
    os.makedirs("data/_debug", exist_ok=True)
    df.to_csv("data/opponent_map.csv", index=False)
    print(f"[OpponentMap] wrote {len(df)} rows for week={week} → data/opponent_map.csv")
    return df

def attach_opponent(
    df: pd.DataFrame,
    team_col: str = "team",
    coverage_path: str = "data/team_week_map.csv",
    opponent_col: str = "opponent",
    inplace: bool = True,
    week: Optional[int] = None,
) -> pd.DataFrame:
    """
    Attach opponent by (season, week, team) using team_week_map.csv;
    fall back to any pre-built data/opponent_map_from_props.csv when available.
    """
    if df is None or df.empty:
        return df
    target = df if inplace else df.copy()
    if team_col not in target.columns:
        return target

    target[team_col] = normalize_team_series(target[team_col])
    if opponent_col not in target.columns:
        target[opponent_col] = pd.NA

    sched = build_opponent_map(week=week, team_map_path=coverage_path)
    if not sched.empty:
        join_cols = []
        for col in ("season","week","team"):
            if col in target.columns and col in sched.columns:
                join_cols.append(col)
        # event_id optional improvement
        extra = [c for c in ("event_id",) if c in target.columns and c in sched.columns]
        sel = list(set(join_cols + extra + ["opponent"]))
        over = sched[sel].drop_duplicates()
        over = over.rename(columns={"opponent": "__schedule_opp"})
        target = target.merge(over, on=join_cols + extra, how="left")
        if "__schedule_opp" in target.columns:
            target[opponent_col] = target[opponent_col].fillna(target["__schedule_opp"])
            target.drop(columns=["__schedule_opp"], inplace=True)

    # If props-built map exists, allow it to fill remaining holes.
    props_path = "data/opponent_map_from_props.csv"
    if os.path.exists(props_path):
        try:
            om = pd.read_csv(props_path)
        except Exception:
            om = pd.DataFrame()
        if not om.empty:
            for c in ("team","opponent"):
                if c in om.columns:
                    om[c] = normalize_team_series(om[c])
            if "team" in target.columns and "team" in om.columns:
                over2 = om[["team","opponent"]].dropna().drop_duplicates().rename(columns={"opponent":"__props_opp"})
                target = target.merge(over2, on=["team"], how="left")
                if "__props_opp" in target.columns:
                    target[opponent_col] = target[opponent_col].fillna(target["__props_opp"])
                    target.drop(columns=["__props_opp"], inplace=True)

    # Log any unresolved mappings for debugging
    miss = target[target[opponent_col].isna()].copy()
    if not miss.empty:
        os.makedirs("data/_debug", exist_ok=True)
        miss.to_csv("data/_debug/opponent_map_unresolved.csv", index=False)
        print(f"[OpponentMap] WARNING unresolved opponent rows: {len(miss)} → data/_debug/opponent_map_unresolved.csv")

    return target

# --- Legacy API shims for backward compatibility with existing scripts ---
# Some scripts still import: map_normalize_team, team_map
# Keep these thin wrappers so we don’t have to touch the callers.

def map_normalize_team(x):
    """Legacy alias → normalize a single team token to canonical code."""
    return normalize_team(x)


def map_normalize_team_series(vals):
    """Legacy alias → normalize a pandas Series/iterable of team tokens."""
    return normalize_team_series(vals)


def dump_norm_debug(df: pd.DataFrame, path: str) -> None:
    try:
        sel = df.copy()
        for col in ("team", "opponent", "team_abbr", "opponent_abbr"):
            if col in sel.columns:
                sel[col] = map_normalize_team(sel[col].astype("string"))
        sel.head(200).to_csv(path, index=False)
    except Exception:
        pass


def team_map(week: int = 10, team_map_path: str = "data/team_week_map.csv"):
    """Legacy alias → returns the schedule/opponent map for a given week."""
    return build_opponent_map(week=week, team_map_path=team_map_path)

# Be explicit about public surface (helps static tools/linters).
__all__ = [
    "TEAM_ALIASES", "CANON_SET",
    "normalize_team", "normalize_team_series",
    "map_normalize_team", "map_normalize_team_series",
    "map_normalize_opponent_map", "dump_norm_debug",
    "build_opponent_map", "attach_opponent",
    # legacy:
    "team_map",
]
