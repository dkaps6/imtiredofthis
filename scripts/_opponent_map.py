# scripts/_opponent_map.py
import logging
from typing import Iterable, Optional, Union
import os
import pandas as pd

logger = logging.getLogger("opponent_map")

# Normalize many site/book/team variants into model's canonical 2–3 letter codes.
TEAM_ALIASES = {
    # Site/book oddities & historical
    "BLT": "BAL", "BAL RAVENS": "BAL",
    "CLV": "CLE", "CLEVELAND BROWNS": "CLE",
    "HST": "HOU", "HOUSTON TEXANS": "HOU",
    "JAC": "JAX", "WSH": "WAS", "WFT": "WAS", "COMMANDERS": "WAS",
    "NEP": "NE", "N.E.": "NE",
    "GNB": "GB", "G.B.": "GB",
    "SFO": "SF", "S.F.": "SF",
    "ARZ": "ARI",
    "KCC": "KC", "K.C.": "KC",
    "SD": "LAC", "S.D.": "LAC", "LA CHARGERS": "LAC",
    "STL": "LA", "LA RAMS": "LA", "LAR": "LA",
    "N.O.": "NO", "NOR": "NO", "NOS": "NO",
    "T.B.": "TB", "TAM": "TB",
    "N.Y. JETS": "NYJ", "NY JETS": "NYJ",
    "N.Y. GIANTS": "NYG", "NY GIANTS": "NYG",
    "OAK": "LV",  # legacy
    # Identity passthroughs
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","KC":"KC",
    "LV":"LV","LAC":"LAC","LA":"LA","MIA":"MIA","MIN":"MIN","NE":"NE","NO":"NO","NYG":"NYG","NYJ":"NYJ",
    "PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF","TB":"TB","TEN":"TEN","WAS":"WAS",
}

CANON_SET = set([
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
    "IND","JAX","KC","LV","LAC","LA","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
])

def _normalize_one(x: Union[str, object]) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if not s or s in {"NAN","NA","NONE","NULL","<NA>"}:
        return ""
    s = s.replace(".", "").replace("  ", " ")
    s = TEAM_ALIASES.get(s, s)
    if s in CANON_SET:
        return s
    return TEAM_ALIASES.get(s, s)

def normalize_team(val: Union[str, pd.Series, object]) -> object:
    if isinstance(val, pd.Series):
        return val.apply(_normalize_one)
    return _normalize_one(val)

def normalize_team_series(vals: Union[pd.Series, Iterable]) -> pd.Series:
    s = pd.Series(vals, copy=False)
    return s.apply(_normalize_one)

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
    working["team"] = normalize_team_series(working["team"])
    working["opponent"] = working["opponent"].astype(str).str.upper().str.strip()
    working.loc[working["opponent"].ne("BYE"), "opponent"] = normalize_team_series(working["opponent"])

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
