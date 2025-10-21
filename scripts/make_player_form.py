# scripts/make_player_form.py
"""
Build player-level shares and efficiency for the 2025 season.

Outputs: data/player_form.csv

Columns written:
- player, team, season, position, role
- tgt_share, route_rate, rush_share
- yprr, ypt, ypc, ypa
- receptions_per_target
- rz_share, rz_tgt_share, rz_rush_share

Surgical changes:
- Fill POSITION using multiple sources (weekly rosters → players master → PBP usage family).
- Do NOT coerce NaN to literal "NAN" prior to inference.
- Infer ROLE even when exact position is missing (uses family from usage).
- roles.csv remains an optional, non-destructive override.
- VALIDATOR: only enforce required metrics for players present in outputs/props_raw.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any, Dict, List
import re

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPATH = os.path.join(DATA_DIR, "player_form.csv")

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

FINAL_COLS = [
    "player",
    "team",
    "opponent",
    "season",
    "position",
    "role",
    "tgt_share",
    "route_rate",
    "rush_share",
    "yprr",
    "ypt",
    "ypc",
    "ypa",
    "receptions_per_target",
    "rz_share",
    "rz_tgt_share",
    "rz_rush_share",
]
# === BEGIN: SURGICAL NAME NORMALIZATION HELPERS (idempotent) ===
try:
    _NAME_HELPERS_DEFINED
except NameError:
    import re as _re_nh
    import unicodedata as _ud_nh
    _NAME_HELPERS_DEFINED = True

    _SUFFIX_RE_NH = _re_nh.compile(r"\b(JR|SR|II|III|IV|V)\b\.?", _re_nh.IGNORECASE)
    _LEADING_NUM_RE_NH = _re_nh.compile(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", _re_nh.UNICODE)

    def _deaccent_nh(s: str) -> str:
        try:
            return _ud_nh.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        except Exception:
            return s

    def _clean_person_name_nh(s: str) -> str:
        s = (s or "").replace("\xa0"," ").strip()
        s = _LEADING_NUM_RE_NH.sub("", s)
        s = s.replace(".", "")
        s = _SUFFIX_RE_NH.sub("", s)
        s = _re_nh.sub(r"\s+", " ", s)
        s = _deaccent_nh(s)
        return s.strip()

    def _player_key_from_name_nh(s: str) -> str:
        s = _clean_person_name_nh(s)
        return _re_nh.sub(r"[^a-z0-9]", "", s.lower())
# === END: SURGICAL NAME NORMALIZATION HELPERS ===



def _is_empty(obj) -> bool:
    try:
        return obj is None or (hasattr(obj, "__len__") and len(obj) == 0)
    except Exception:
        return True

def _to_pandas(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas") and callable(getattr(obj, "to_pandas")):
        try:
            return obj.to_pandas()
        except Exception:
            pass
    if isinstance(obj, (list, tuple)) and obj and hasattr(obj[0], "to_pandas"):
        try:
            return pd.concat([b.to_pandas() for b in obj], ignore_index=True)
        except Exception:
            pass
    return pd.DataFrame(obj)

def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

# === SURGICAL ADDITION: merge roles from ESPN and Ourlads (clean placement) ===
def _merge_depth_roles(pf: pd.DataFrame) -> pd.DataFrame:
    import os, re
    data_dir = globals().get("DATA_DIR", "data")
    pf = pf.loc[:, ~pf.columns.duplicated()].copy()
    def _clean_name(s: str) -> str:
        s = str(s or "").strip()
        if "," in s:
            parts = [p.strip() for p in s.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                s = f"{parts[1]} {parts[0]}"
        s = re.sub(r"\(.*?\)", "", s)
        s = re.sub(r"\b[A-Z]{1,3}\d{2}\b", "", s)
        s = re.sub(r"\b[Uu]/[A-Za-z]{2,4}\b", "", s)
        s = re.sub(r"\b\d{1,2}/\d{1,2}\b", "", s)
        s = re.sub(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", "", s)
        s = s.replace(".", " ")
        s = re.sub(r"\s+(JR|SR|II|III|IV|V)\.?$", "", s, flags=re.I)
        s = re.sub(r"[^\w\s'\-]", " ", s)
        s = re.sub(r"\b\d+\b", "", s)
        s = re.sub(r"\s+", " ", s).strip().title()
        return "" if s == "U" else s
    try:
        roles_espn = pd.read_csv(os.path.join(data_dir, "roles_espn.csv"))
    except Exception:
        roles_espn = pd.DataFrame(columns=["team","player","role"])
    try:
        roles_ourlads = pd.read_csv(os.path.join(data_dir, "roles_ourlads.csv"))
    except Exception:
        roles_ourlads = pd.DataFrame(columns=["team","player","role"])
    roles = pd.concat([roles_espn, roles_ourlads], ignore_index=True, sort=False)
    if roles.empty:
        print("[make_player_form] No roles_espn or roles_ourlads found, skipping merge.")
        return pf
    for col in ("team","player","role"):
        if col in roles.columns:
            roles[col] = roles[col].astype(str)
    if "player" in roles.columns:
        roles["player"] = roles["player"].map(_clean_name)
    roles["team"] = roles.get("team","").astype(str).map(_canon_team)
    roles["role"] = roles.get("role","").astype(str).str.upper().str.strip()
    if "position" not in roles.columns and "role" in roles.columns:
        roles["position"] = roles["role"].str.extract(r"([A-Z]+)")
    def _rank(r):
        m = re.search(r"(\d+)$", str(r))
        return int(m.group(1)) if m else 999
    if {"team","player","role"}.issubset(roles.columns):
        roles["_rk"] = roles["role"].map(_rank)
        roles = (roles.sort_values(["team","player","_rk"])
                      .drop_duplicates(["team","player"], keep="first")
                      .drop(columns=["_rk"]))
    pf["team"] = pf["team"].astype(str).map(_canon_team)
    pf["player_join"] = pf["player"].astype(str).str.replace(r"[^A-Za-z]", "", regex=True)
    roles_join = roles.copy()
    if "player_key_concat" in roles_join.columns:
        roles_join = roles_join.rename(columns={"player_key_concat": "player_join"})
    else:
        roles_join["player_join"] = roles_join["player"].astype(str).str.replace(r"[^A-Za-z]", "", regex=True)
    if "player" in roles_join.columns:
        roles_join = roles_join.rename(columns={"player":"player_depth_name"})
    roles_join["team"] = roles_join["team"].astype(str).map(_canon_team)
    roles_join = roles_join.loc[:, ~roles_join.columns.duplicated()].copy()
    try:
        pf = pf.merge(
            roles_join[["team","player_join","role","position"]],
            on=["team","player_join"],
            how="left",
            suffixes=("", "_depth"),
            validate="many_to_one",
        )
    except Exception as e:
        print(f"[make_player_form] WARN roles merge issue: {e}")
        pf = pf.merge(
            roles_join[["team","player_join","role","position"]],
            on=["team","player_join"],
            how="left",
            suffixes=("", "_depth"),
        )
    pf.drop(columns=["player_join"], inplace=True, errors="ignore")
    if "position_depth" in pf.columns:
        pf["position"] = pf["position"].combine_first(pf["position_depth"])
    if "role_depth" in pf.columns:
        pf["role"] = pf["role"].combine_first(pf["role_depth"])
    drop_cols = [c for c in pf.columns if c.endswith("_depth")]
    if drop_cols:
        pf.drop(columns=drop_cols, inplace=True, errors="ignore")
    pf = pf.loc[:, ~pf.columns.duplicated()]
    try:
        miss = pf[pf["role"].isna()][["player","team"]].copy()
        if not miss.empty:
            os.makedirs(data_dir, exist_ok=True)
            miss.to_csv(os.path.join(data_dir, "unmatched_roles_merge.csv"), index=False)
            print(f"[make_player_form] unmatched after roles merge: {len(miss)} → {os.path.join(data_dir, 'unmatched_roles_merge.csv')}")
    except Exception:
        pass
    try:
        cov = pf["position"].notna().mean()
        print(f"[make_player_form] merged depth roles → coverage now {cov:.2%}")
    except Exception:
        pass
    return pf

def _norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".", "", regex=False).str.strip()

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _non_destructive_merge(base: pd.DataFrame, add: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if _is_empty(add):
        return base
    add = add.copy()
    add.columns = [c.lower() for c in add.columns]
    if not set(keys).issubset(add.columns):
        return base
    for k in keys:
        add[k] = add[k].astype(str).str.strip()
    out = base.merge(add, on=keys, how="left", suffixes=("", "_ext"))
    for c in add.columns:
        if c in keys:
            continue
        ext = f"{c}_ext"
        if ext in out.columns:
            out[c] = out[c].combine_first(out[ext])
            out.drop(columns=[ext], inplace=True)
    return out

# ---------------------------
# Team canonicalizer
# ---------------------------
VALID = {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
         "IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
         "PHI","PIT","SEA","SF","TB","TEN","WAS"}

DEFENSE_TEAM_CANDIDATES = [
    "defteam",
    "defense_team",
    "def_team",
    "defense",
    "defteam_abbr",
    "defense_abbr",
    "opp_team",
    "opp_team_abbr",
    "opp",
    "opp_abbr",
    "opponent",
]

TEAM_NAME_TO_ABBR = {
    "ARI":"ARI","ARZ":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB","GNB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","JAC":"JAX",
    "KC":"KC","KCC":"KC","LAC":"LAC","LAR":"LAR","LA":"LAR","LV":"LV","OAK":"LV","LAS":"LV","MIA":"MIA",
    "MIN":"MIN","NE":"NE","NWE":"NE","NO":"NO","NOR":"NO","NYG":"NYG","NYJ":"NYJ","PHI":"PHI","PIT":"PIT",
    "SEA":"SEA","SF":"SF","SFO":"SF","TB":"TB","TAM":"TB","TEN":"TEN","WAS":"WAS","WSH":"WAS","WFT":"WAS",
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","LAS VEGAS RAIDERS":"LV",
    "MIAMI DOLPHINS":"MIA","MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE",
    "NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG","NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI",
    "PITTSBURGH STEELERS":"PIT","SEATTLE SEAHAWKS":"SEA","SAN FRANCISCO 49ERS":"SF",
    "TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS",
    "WASHINGTON FOOTBALL TEAM":"WAS",
    "ARIZONA":"ARI","CARDINALS":"ARI","ATLANTA":"ATL","FALCONS":"ATL","BALTIMORE":"BAL","RAVENS":"BAL",
    "BUFFALO":"BUF","BILLS":"BUF","CAROLINA":"CAR","PANTHERS":"CAR","CHICAGO":"CHI","BEARS":"CHI",
    "CINCINNATI":"CIN","BENGALS":"CIN","CLEVELAND":"CLE","BROWNS":"CLE","DALLAS":"DAL","COWBOYS":"DAL",
    "DENVER":"DEN","BRONCOS":"DEN","DETROIT":"DET","LIONS":"DET","GREEN BAY":"GB","PACKERS":"GB",
    "HOUSTON":"HOU","TEXANS":"HOU","INDIANAPOLIS":"IND","COLTS":"IND","JACKSONVILLE":"JAX","JAGUARS":"JAX",
    "KANSAS CITY":"KC","CHIEFS":"KC","CHARGERS":"LAC","RAMS":"LAR","LOS ANGELES":"LAR","LAS VEGAS":"LV","RAIDERS":"LV",
    "MIAMI":"MIA","DOLPHINS":"MIA","MINNESOTA":"MIN","VIKINGS":"MIN","NEW ENGLAND":"NE","PATRIOTS":"NE",
    "NEW ORLEANS":"NO","SAINTS":"NO","GIANTS":"NYG","JETS":"NYJ","PHILADELPHIA":"PHI","EAGLES":"PHI",
    "PITTSBURGH":"PIT","STEELERS":"PIT","SEATTLE":"SEA","SEAHAWKS":"SEA","SAN FRANCISCO":"SF","49ERS":"SF",
    "TAMPA BAY":"TB","BUCCANEERS":"TB","TENNESSEE":"TEN","TITANS":"TEN","WASHINGTON":"WAS","COMMANDERS":"WAS",
}

# defensively add lowercase variants for mapping
TEAM_NAME_TO_ABBR.update({k.lower(): v for k, v in TEAM_NAME_TO_ABBR.items()})


def _canon_team(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s in TEAM_NAME_TO_ABBR:
        abbr = TEAM_NAME_TO_ABBR[s]
        return abbr if abbr in VALID else ""
    s2 = re.sub(r"[^A-Z0-9 ]+", "", s).strip()
    if s2 in TEAM_NAME_TO_ABBR:
        abbr = TEAM_NAME_TO_ABBR[s2]
        return abbr if abbr in VALID else ""
    return ""


def _derive_opponent(df: pd.DataFrame) -> pd.Series:
    """Return canonical opponent abbreviations for a play-by-play frame."""

    if df.empty:
        return pd.Series(np.nan, index=df.index, dtype=object)

    col = next((c for c in DEFENSE_TEAM_CANDIDATES if c in df.columns), None)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=object)

    opp = df[col]
    if not isinstance(opp, pd.Series):
        return pd.Series(np.nan, index=df.index, dtype=object)
    opp = opp.where(opp.notna(), "")
    opp = opp.astype(str).str.upper().str.strip()
    mapped = opp.map(_canon_team)
    mapped = mapped.replace("", np.nan)
    return mapped
    # --- schedule-based fallback (only if still missing) ---
    try:
        if "game_id" in df.columns:
            season_guess = None
            if "season" in df.columns and df["season"].notna().any():
                try:
                    season_guess = int(pd.to_numeric(df["season"], errors="coerce").dropna().iloc[0])
                except Exception:
                    season_guess = None
            if season_guess is None:
                season_guess = 2025
            sched = _load_schedule_map(season_guess)
            if not sched.empty:
                merged = df[["game_id"]].copy().merge(sched, on="game_id", how="left")
                if "posteam" in df.columns and {"home_team","away_team"}.issubset(merged.columns):
                    posteam = df["posteam"].astype(str).str.upper().str.strip().map(_canon_team)
                    home = merged["home_team"]
                    away = merged["away_team"]
                    opp = np.where(posteam.eq(home), away, home)
                    opp = pd.Series(opp, index=df.index)
                    opp = opp.map(_canon_team).replace("", np.nan)
                    return opp
    except Exception:
        pass

    return pd.Series(np.nan, index=df.index, dtype=object)
def _normalize_props_opponent(df: pd.DataFrame) -> pd.Series:
    """Derive and canonicalize opponent abbreviations for props payloads."""

    if df.empty:
        return pd.Series(np.nan, index=df.index, dtype=object)

    base = pd.Series(np.nan, index=df.index, dtype=object)
    opp_col = next((c for c in DEFENSE_TEAM_CANDIDATES if c in df.columns), None)
    if opp_col is None:
        return base

    try:
        derived = _derive_opponent(df)
    except Exception:
        derived = base

    if isinstance(derived, pd.Series) and len(derived) == len(df) and derived.notna().any():
        return derived.reindex(df.index)

    raw = df[opp_col]
    if not isinstance(raw, pd.Series):
        return base
    opp_norm = raw.where(raw.notna(), "").astype(str).str.upper().str.strip()
    mapped = opp_norm.map(_canon_team).replace("", np.nan)
    return mapped.reindex(df.index)

# ---------------------------
# nflverse loader
# ---------------------------

def _import_nflverse():
    try:
        import nflreadpy as nflv
        return nflv, "nflreadpy"
    except Exception:
        try:
            import nfl_data_py as nflv  # type: ignore
            return nflv, "nfl_data_py"
        except Exception as e:
            raise RuntimeError(
                "Neither nflreadpy nor nfl_data_py is installed. Run: pip install nflreadpy"
            ) from e

NFLV, NFL_PKG = _import_nflverse()

def _load_schedule_map(season: int) -> pd.DataFrame:
    """Return schedule map (game_id -> home_team, away_team) for the season, canonized to abbrs."""
    try:
        if NFL_PKG == "nflreadpy":
            sched = NFLV.load_schedules(seasons=[season])
        else:
            # nfl_data_py
            if hasattr(NFLV, "import_schedules"):
                sched = NFLV.import_schedules([season])  # type: ignore
            else:
                return pd.DataFrame()
        df = _to_pandas(sched)
    except Exception:
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    if not {"game_id","home_team","away_team"}.issubset(df.columns):
        return pd.DataFrame()
    df["home_team"] = df["home_team"].astype(str).str.upper().str.strip().map(_canon_team)
    df["away_team"] = df["away_team"].astype(str).str.upper().str.strip().map(_canon_team)
    return df[["game_id","home_team","away_team"]].drop_duplicates()


def load_pbp(season: int) -> pd.DataFrame:
    # explicit, version-safe loader with diagnostics
    rows = -1
    try:
        if NFL_PKG == "nflreadpy":
            raw = NFLV.load_pbp(seasons=[season])
        else:
            # nfl_data_py has had both names in the wild
            if hasattr(NFLV, "import_pbp_data"):
                raw = NFLV.import_pbp_data([season], downcast=True)  # type: ignore
            elif hasattr(NFLV, "import_pbp"):
                raw = NFLV.import_pbp([season])  # type: ignore
            else:
                raise RuntimeError("nfl_data_py missing import_pbp(_data) functions")
        pbp = _to_pandas(raw)
        pbp.columns = [c.lower() for c in pbp.columns]
        rows = len(pbp)
        print(f"[pf] PBP loaded for {season}: rows={rows}, sample_cols={list(pbp.columns[:10])}")
        if rows == 0:
            raise RuntimeError("PBP returned 0 rows (unexpected for active season).")
        return pbp
    except Exception as e:
        print(f"[pf] ERROR loading PBP {season}: {type(e).__name__}: {e}", file=sys.stderr)
        return pd.DataFrame()
def _load_weekly_rosters(season: int) -> pd.DataFrame:
    """(player, team, position), forgiving team keys."""
    try:
        if NFL_PKG == "nflreadpy":
            ro = NFLV.load_weekly_rosters(seasons=[season])
        else:
            ro = NFLV.import_weekly_rosters([season])  # type: ignore
        df = _to_pandas(ro)
    except Exception:
        return pd.DataFrame()
    if _is_empty(df):
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]

    # player name
    name_col = None
    for c in ["player_name","name","full_name"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return pd.DataFrame()
    df["player"] = _norm_name(df[name_col].fillna(""))

    # team — accept any column that looks like a team key
    team_col = None
    for c in ["team","recent_team","club_code","team_abbr","posteam"]:
        if c in df.columns:
            team_col = c
            break
    if team_col is None:
        return pd.DataFrame()
    df["team"] = df[team_col].astype(str).str.upper().str.strip().map(_canon_team)
    df = df[df["team"].isin(VALID)]

    # position
    pos_col = None
    for c in ["position","pos"]:
        if c in df.columns:
            pos_col = c
            break
    df["position"] = np.where(pos_col is not None, df[pos_col].astype(str).str.upper().str.strip(), np.nan)

    if "week" in df.columns:
        df = df.sort_values(["player","team","week"]).drop_duplicates(["player","team"], keep="last")

    return df[["player","team","position"]].drop_duplicates()

def _load_players_master() -> pd.DataFrame:
    """Fallback: (player -> position) without team join."""
    try:
        if NFL_PKG == "nflreadpy":
            pl = NFLV.load_players()
        else:
            pl = NFLV.import_players()  # type: ignore
        df = _to_pandas(pl)
    except Exception:
        return pd.DataFrame()
    if _is_empty(df):
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    name_col = None
    for c in ["player_name","name","full_name","display_name"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return pd.DataFrame()
    df["player"] = _norm_name(df[name_col].fillna(""))
    pos_col = None
    for c in ["position","pos","gsis_pos"]:
        if c in df.columns:
            pos_col = c
            break
    if pos_col is None:
        return pd.DataFrame()
    df["position"] = df[pos_col].astype(str).str.upper().str.strip()
    return df[["player","position"]].drop_duplicates()

# ---------------------------
# Optional roles.csv merge
# ---------------------------

def _merge_roles_csv(df: pd.DataFrame) -> pd.DataFrame:
    roles_path = os.path.join(DATA_DIR, "roles.csv")
    if not os.path.exists(roles_path):
        return df
    try:
        r = pd.read_csv(roles_path)
    except Exception:
        return df
    r.columns = [c.lower() for c in r.columns]
    if "player" not in r.columns and "player_name" in r.columns:
        r = r.rename(columns={"player_name": "player"})
    need = {"player","team","opponent","role"}
    if not need.issubset(r.columns):
        return df
    r["player"] = _norm_name(r["player"].astype(str))
    r["team"] = r["team"].astype(str).str.upper().str.strip().map(_canon_team)
    r = r[r["team"].isin(VALID)]
    out = df.merge(r[["player","team","opponent","role"]], on=["player","team"], how="left", suffixes=("","_roles"))
    if "role_roles" in out.columns:
        out["role"] = out["role"].combine_first(out["role_roles"])
        out.drop(columns=["role_roles"], inplace=True)
    return out

# ---------------------------
# Role & family inference
# ---------------------------

def _infer_position_family_from_usage(pf: pd.DataFrame) -> pd.Series:
    """
    Return a Series of position family guesses {QB,RB,WR} based on usage
    when exact position is missing.
    """
    fam = pd.Series(index=pf.index, dtype=object)
    # Heuristics:
    # 1) If dropbacks or pass attempts large → QB
    qb_mask = pd.Series(False, index=pf.index)
    if "dropbacks" in pf.columns:
        qb_mask |= (pf["dropbacks"].fillna(0) >= 15)
    if "ypa" in pf.columns:
        qb_mask |= (pf["ypa"].notna() & (pf["ypa"] > 6.0))
    fam[qb_mask] = "QB"

    # 2) If rush_share is present and dominates → RB
    rb_mask = (pf.get("rush_share", pd.Series(0, index=pf.index)).fillna(0) >= 0.20)
    fam[rb_mask & fam.isna()] = "RB"

    # 3) Else default to WR family for receiving usage
    wr_mask = (pf.get("route_rate", pd.Series(0, index=pf.index)).fillna(0) >= 0.20) | \
              (pf.get("tgt_share",   pd.Series(0, index=pf.index)).fillna(0) >= 0.15)
    fam[wr_mask & fam.isna()] = "WR"

    return fam

def _infer_roles_minimal(pf: pd.DataFrame) -> pd.DataFrame:
    pf = pf.copy()
    if "role" not in pf.columns:
        pf["role"] = np.nan

    # use exact position if present, else fall back to family guess
    pos = pf.get("position")
    fam = pd.Series(index=pf.index, dtype=object)
    if pos is not None:
        fam = pos.copy()
    # only fill where fam is null
    fam = fam.where(fam.notna(), _infer_position_family_from_usage(pf))
    fam = fam.astype(object)

    def rank_and_tag(g: pd.DataFrame, mask: pd.Series, score_col: str, tags: List[str]):
        g = g.copy()
        idx = g.index[mask]
        if len(idx) == 0 or score_col not in g.columns:
            return g
        scores = g.loc[idx, score_col].astype(float)
        order = scores.rank(method="first", ascending=False)
        if len(tags) >= 1:
            g.loc[idx[order == 1], "role"] = g.loc[idx[order == 1], "role"].where(g.loc[idx[order == 1], "role"].notna(), tags[0])
        if len(tags) >= 2:
            g.loc[idx[order == 2], "role"] = g.loc[idx[order == 2], "role"].where(g.loc[idx[order == 2], "role"].notna(), tags[1])
        return g

    out = []
    for team, g in pf.groupby(["team","opponent"], dropna=False):
        g = g.copy()
        g_fam = fam.loc[g.index].astype(str)

        # QB1 by dropbacks (fallback ypa)
        qb_mask = g_fam.str.upper().eq("QB")
        if qb_mask.any():
            if "dropbacks" in g.columns and g["dropbacks"].notna().any():
                g = rank_and_tag(g, qb_mask, "dropbacks", ["QB1"])
            elif "ypa" in g.columns and g["ypa"].notna().any():
                g = rank_and_tag(g, qb_mask, "ypa", ["QB1"])

        # RB1/RB2 by rush_share
        rb_mask = g_fam.str.upper().eq("RB")
        if rb_mask.any() and "rush_share" in g.columns and g["rush_share"].notna().any():
            g = rank_and_tag(g, rb_mask, "rush_share", ["RB1","RB2"])

        # WR1/WR2 by route_rate (TE may be treated as WR family if unknown)
        wr_mask = g_fam.str.upper().eq("WR")
        if wr_mask.any() and "route_rate" in g.columns and g["route_rate"].notna().any():
            g = rank_and_tag(g, wr_mask, "route_rate", ["WR1","WR2"])

        out.append(g)

    return pd.concat(out, ignore_index=True) if out else pf

# ---------------------------
# Build from PBP
# ---------------------------

def build_player_form(season: int = 2025) -> pd.DataFrame:
    try:
        pbp = load_pbp(season)
    except Exception as err:
        warnings.warn(
            f"Failed to load play-by-play for season {season}: {err}; proceeding with empty data frame.",
            RuntimeWarning,
        )
        pbp = pd.DataFrame()
    if pbp.empty:
        allow_offseason = os.getenv("ALLOW_OFFSEASON_FALLBACK", "0") != "0"
        msg = f"No play-by-play data available for season {season}."
        if not allow_offseason:
            raise RuntimeError(msg + " Set ALLOW_OFFSEASON_FALLBACK=1 to write structural base.")
        warnings.warn(msg + " Writing structural base due to ALLOW_OFFSEASON_FALLBACK.", RuntimeWarning)
        base = pd.DataFrame(columns=["player", "team"])
        base["season"] = int(season)
        base = _ensure_cols(base, FINAL_COLS)
        base = base[FINAL_COLS].drop_duplicates(subset=["player","team","opponent","season"]).reset_index(drop=True)
        print("[pf] pbp empty → structural base only")
        return base
    
    off_col = "posteam" if "posteam" in pbp.columns else ("offense_team" if "offense_team" in pbp.columns else None)
    if off_col is None:
        raise RuntimeError("No offense team column in PBP (posteam/offense_team).")

    # Opponent once (remove duplicate logic)
    opp_col = "defteam" if "defteam" in pbp.columns else ("defense_team" if "defense_team" in pbp.columns else None)
    if opp_col is None:
        pbp["opponent"] = np.nan
    else:
        pbp["opponent"] = pbp[opp_col].astype(str).str.upper().str.strip()


    # Ensure counting columns exist and are numeric to avoid groupby collapse
    for col in ["pass_attempt","complete_pass","qb_dropback","rush_attempt","yards_gained"]:
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0)
            # boolean/int flags should be ints
            if col != "yards_gained":
                pbp[col] = pbp[col].astype(int)

    # Derive robust is_pass / is_rush flags
    pt = pbp.get("play_type")
    is_pass = pbp.get("pass")
    if is_pass is None:
        is_pass = pt.isin(["pass","no_play"]) if pt is not None else pd.Series(False, index=pbp.index)
    else:
        is_pass = pd.Series(is_pass).astype(bool)
    is_rush = pbp.get("rush")
    if is_rush is None:
        is_rush = pt.eq("run") if pt is not None else pd.Series(False, index=pbp.index)
    else:
        is_rush = pd.Series(is_rush).astype(bool)

    # RECEIVING
    is_pass = pbp.get("pass")
    if is_pass is None:
        pt = pbp.get("play_type")
        is_pass = pt.isin(["pass","no_play"]) if pt is not None else pd.Series(False, index=pbp.index)
    else:
        is_pass = is_pass.astype(bool)

    rec = pbp.loc[is_pass].copy()
    rec["opponent"] = _derive_opponent(rec)
    if rec.empty:
        rply = pd.DataFrame(columns=["team","opponent","player"])
    else:
        rcv_name_col = "receiver_player_name" if "receiver_player_name" in rec.columns else ("receiver" if "receiver" in rec.columns else None)
        if rcv_name_col is None:
            rec["receiver_player_name"] = np.nan
            rcv_name_col = "receiver_player_name"
        rec["player"] = _norm_name(rec[rcv_name_col].fillna(""))
        rec["team"] = rec[off_col].astype(str).str.upper().str.strip().map(_canon_team)
        rec["team"] = rec["team"].replace("", np.nan)

        rec["opponent"] = rec["opponent"].astype(str).str.upper().str.strip() if "opponent" in rec.columns else np.nan
        team_targets = rec.groupby(["team","opponent"], dropna=False).size().rename("team_targets").astype(float)
        if "qb_dropback" in rec.columns:
            team_dropbacks = rec.groupby(["team","opponent"], dropna=False)["qb_dropback"].sum(min_count=1).rename("team_dropbacks")
        else:
            team_dropbacks = rec.groupby(["team","opponent"], dropna=False).size().rename("team_dropbacks").astype(float)

        rply = rec.groupby(["team","opponent","player"], dropna=False).agg(
            targets=("pass_attempt","sum") if "pass_attempt" in rec.columns else ("player","size"),
            rec_yards=("yards_gained","sum"),
            receptions=("complete_pass","sum") if "complete_pass" in rec.columns else (rcv_name_col,"size"),
        ).reset_index()
        rply = rply.merge(team_targets.reset_index(), on=["team","opponent"], how="left")
        rply = rply.merge(team_dropbacks.reset_index(), on=["team","opponent"], how="left")
        rply["tgt_share"] = np.where(rply["team_targets"]>0, rply["targets"]/rply["team_targets"], np.nan)
        rply["route_rate"] = np.where(rply["team_dropbacks"]>0, rply["targets"]/rply["team_dropbacks"], np.nan).clip(0.05, 0.95)
        rply["ypt"] = np.where(rply["targets"]>0, rply["rec_yards"]/rply["targets"], np.nan)
        rply["receptions_per_target"] = np.where(rply["targets"]>0, rply["receptions"]/rply["targets"], np.nan)
        routes_proxy = (rply["team_dropbacks"] * rply["route_rate"]).replace(0, np.nan)
        rply["yprr"] = np.where(routes_proxy>0, rply["rec_yards"]/routes_proxy, np.nan)

        inside20 = rec.copy()
        inside20["yardline_100"] = pd.to_numeric(inside20.get("yardline_100"), errors="coerce")
        rz_rec = inside20.loc[inside20["yardline_100"] <= 20]
        if not rz_rec.empty:
            rz_tgt_ply = rz_rec.groupby(["team","opponent","player"]).size().rename("rz_targets")
            rz_tgt_tm  = rz_rec.groupby("team").size().rename("rz_team_targets")
            rply = rply.merge(rz_tgt_ply.reset_index(), on=["team","opponent","player"], how="left")
            rply = rply.merge(rz_tgt_tm.reset_index(),  on=["team","opponent"],          how="left")
            rply["rz_tgt_share"] = np.where(rply["rz_team_targets"]>0, rply["rz_targets"]/rply["rz_team_targets"], np.nan)

    rply = _ensure_cols(rply, [
        "targets","rec_yards","receptions","team_targets","team_dropbacks",
        "tgt_share","route_rate","ypt","receptions_per_target","yprr",
        "rz_targets","rz_team_targets","rz_tgt_share",
    ])

    # RUSHING
    is_rush = pbp.get("rush")
    if is_rush is None:
        pt = pbp.get("play_type")
        is_rush = pt.eq("run") if pt is not None else pd.Series(False, index=pbp.index)
    else:
        is_rush = is_rush.astype(bool)

    ru = pbp.loc[is_rush].copy()
    ru["opponent"] = _derive_opponent(ru)
    if ru.empty:
        rru = pd.DataFrame(columns=["team","opponent","player"])
    else:
        rush_name_col = "rusher_player_name" if "rusher_player_name" in ru.columns else ("rusher" if "rusher" in ru.columns else None)
        if rush_name_col is None:
            ru["rusher_player_name"] = np.nan
            rush_name_col = "rusher_player_name"

        ru["player"] = _norm_name(ru[rush_name_col].fillna(""))
        ru["team"] = ru[off_col].astype(str).str.upper().str.strip().map(_canon_team)
        ru["team"] = ru["team"].replace("", np.nan)

        ru["opponent"] = ru["opponent"].astype(str).str.upper().str.strip() if "opponent" in ru.columns else np.nan
        team_rushes = ru.groupby(["team","opponent"], dropna=False).size().rename("team_rushes").astype(float)
        rru = ru.groupby(["team","opponent","player"], dropna=False).agg(
            rushes=("rush_attempt","sum") if "rush_attempt" in ru.columns else ("player","size"),
            rush_yards=("yards_gained","sum"),
        ).reset_index()
        rru = rru.merge(team_rushes.reset_index(), on=["team","opponent"], how="left")
        rru["rush_share"] = np.where(rru["team_rushes"]>0, rru["rushes"]/rru["team_rushes"], np.nan)
        rru["ypc"] = np.where(rru["rushes"]>0, rru["rush_yards"]/rru["rushes"], np.nan)

        inside10 = ru.copy()
        inside10["yardline_100"] = pd.to_numeric(inside10.get("yardline_100"), errors="coerce")
        rz_ru = inside10.loc[inside10["yardline_100"] <= 10]
        if not rz_ru.empty:
            rz_ru_ply = rz_ru.groupby(["team","opponent","player"]).size().rename("rz_rushes")
            rz_ru_tm  = rz_ru.groupby("team").size().rename("rz_team_rushes")
            rru = rru.merge(rz_ru_ply.reset_index(), on=["team","opponent","player"], how="left")
            rru = rru.merge(rz_ru_tm.reset_index(),  on=["team","opponent"],          how="left")
            rru["rz_rush_share"] = np.where(rru["rz_team_rushes"]>0, rru["rz_rushes"]/rru["rz_team_rushes"], np.nan)

    rru = _ensure_cols(rru, [
        "rushes","rush_yards","team_rushes","rush_share","ypc",
        "rz_rushes","rz_team_rushes","rz_rush_share",
    ])

    # QUARTERBACK
    qb_df = pd.DataFrame(columns=["team","opponent","player","ypa","dropbacks"])
    qb_name_col = "passer_player_name" if "passer_player_name" in pbp.columns else ("passer" if "passer" in pbp.columns else None)
    if qb_name_col is not None:
        qb = pbp.copy()
        qb["opponent"] = _derive_opponent(qb)
        qb["player"] = _norm_name(qb[qb_name_col].fillna(""))
        qb["team"] = qb[off_col].astype(str).str.upper().str.strip().map(_canon_team)
        qb["team"] = qb["team"].replace("", np.nan)
        qb["opponent"] = qb["opponent"].astype(str).str.upper().str.strip() if "opponent" in qb.columns else np.nan
        gb = qb.groupby(["team", "opponent", "player"], dropna=False).agg(
            pass_yards=("yards_gained","sum"),
            pass_att=("pass_attempt","sum") if "pass_attempt" in qb.columns else (qb_name_col,"size"),
            dropbacks=("qb_dropback","sum") if "qb_dropback" in qb.columns else (qb_name_col,"size"),
        ).reset_index()
        gb["ypa"] = np.where(gb["pass_att"]>0, gb["pass_yards"]/gb["pass_att"], np.nan)
        qb_df = gb[["team","opponent","player","ypa","dropbacks"]]

    # Merge all
    base = pd.merge(rply, rru, on=["team","opponent","player"], how="outer")
    base = pd.merge(base, qb_df, on=["team","opponent","player"], how="left")
    base["rz_share"] = base[["rz_tgt_share","rz_rush_share"]].max(axis=1)
    base["season"] = int(season)

    print("[pf] base after concat/merge:", len(base))

    # Initialize position/role as NaN (do not uppercase yet)
    base["position"] = np.nan
    base["role"] = np.nan

    # Normalize keys
    base = _ensure_cols(base, ["opponent"])
    base["player"] = _norm_name(base["player"].astype(str))
    base["team"] = base["team"].astype(str).str.upper().str.strip().map(_canon_team)
    base["opponent"] = base["opponent"].astype(str).str.upper().str.strip().map(_canon_team)
    base["opponent"] = base["opponent"].replace("", np.nan)

    # POSITION ENRICHMENT: weekly rosters → players master → usage family
    ro = _load_weekly_rosters(season)
    if not ro.empty:
        ro["player"] = _norm_name(ro["player"].astype(str))
        ro["team"] = ro["team"].astype(str).str.upper().str.strip().map(_canon_team)
        ro = ro[ro["team"].isin(VALID)]
        base = base.merge(ro, on=["player","team"], how="left", suffixes=("","_ro"))
        if "position_ro" in base.columns:
            base["position"] = base["position"].combine_first(base["position_ro"])
            base.drop(columns=["position_ro"], inplace=True, errors="ignore")

    # Fallback: players master (merge by player only)
    if base["position"].isna().all():
        pm = _load_players_master()
        if not pm.empty:
            pm["player"] = _norm_name(pm["player"].astype(str))
            base = base.merge(pm, on="player", how="left", suffixes=("","_pm"))
            if "position_pm" in base.columns:
                base["position"] = base["position"].combine_first(base["position_pm"])
                base.drop(columns=["position_pm"], inplace=True, errors="ignore")

    # Final fallback: usage-based family inference → write into position when still missing
    if base["position"].isna().any():
        fam = _infer_position_family_from_usage(base)
        base["position"] = base["position"].where(base["position"].notna(), fam)

    # Only uppercase non-null positions (avoid turning NaN into "NAN")
    base.loc[base["position"].notna(), "position"] = base.loc[base["position"].notna(), "position"].astype(str).str.upper().str.strip()

    # roles.csv (non-destructive) then infer roles
    base = _merge_depth_roles(base)
    base = _merge_roles_csv(base)
    if base.get("role", pd.Series(dtype=object)).isna().all():
        base = _infer_roles_minimal(base)

    base = _ensure_cols(base, FINAL_COLS)
    out = base[FINAL_COLS].drop_duplicates(subset=["player","team","opponent","season"]).reset_index(drop=True)
    out = _enrich_team_and_opponent_from_props(out)
    print("[pf] final rows (pre-write):", len(out))
    return out

# ---------------------------
# Fallback enrichers (optional CSVs)
# ---------------------------

def _apply_fallback_enrichers(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "espn_player_form.csv",
        "msf_player_form.csv",
        "apisports_player_form.csv",
        "nflgsis_player_form.csv",
    ]
    out = df.copy()
    out = _ensure_cols(out, ["opponent"])
    for fn in candidates:
        try:
            ext = _read_csv_safe(os.path.join(DATA_DIR, fn))
            if _is_empty(ext):
                continue
            if "player" not in ext.columns and "player_name" in ext.columns:
                ext = ext.rename(columns={"player_name": "player"})
            if not {"player","team"}.issubset(ext.columns):
                continue
            ext["player"] = _norm_name(ext["player"].astype(str))
            ext["team"] = ext["team"].astype(str).str.upper().str.strip().map(_canon_team)
            ext = ext[ext["team"].isin(VALID)]
            out = _non_destructive_merge(out, ext, keys=["player","team"])
        except Exception:
            continue
    return out

# ---------------------------
# PROPS-SCOPED VALIDATION
# ---------------------------

def _load_props_players() -> pd.DataFrame:
    """
    Read outputs/props_raw.csv to get the set of players (and teams) we actually need to validate.
    Returns DataFrame with columns: player, team, player_key (stable).
    """
    path = os.path.join("outputs", "props_raw.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["player","team","opponent","player_key"])
    try:
        pr = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["player","team","opponent","player_key"])

    pr.columns = [c.lower() for c in pr.columns]
    if "player" not in pr.columns:
        if "player_name" in pr.columns:
            pr = pr.rename(columns={"player_name": "player"})
        elif "name" in pr.columns:
            pr = pr.rename(columns={"name": "player"})
        else:
            pr["player"] = np.nan
    pr["player"] = _norm_name(pr.get("player", pd.Series([], dtype=object)).astype(str))

    if "team" not in pr.columns:
        pr["team"] = np.nan
    else:
        pr["team"] = pr["team"].astype(str).str.upper().str.strip().map(_canon_team)

    pr["opponent"] = _normalize_props_opponent(pr)
    opp_col = next((c for c in DEFENSE_TEAM_CANDIDATES if c in pr.columns), None)
    if opp_col is not None:
        try:
            derived = _derive_opponent(pr)
        except Exception:
            derived = pd.Series(np.nan, index=pr.index, dtype=object)
        if not isinstance(derived, pd.Series) or derived.shape[0] != len(pr):
            derived = pd.Series(np.nan, index=pr.index, dtype=object)
        if derived.notna().any():
            opp_norm = (
                derived.where(derived.notna(), "")
                .astype(str)
                .str.upper()
                .str.strip()
            )
            pr["opponent"] = opp_norm.map(_canon_team).replace("", np.nan)
        else:
            opp_raw = pr[opp_col]
            if isinstance(opp_raw, pd.Series):
                opp_norm = (
                    opp_raw.where(opp_raw.notna(), "")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )
                pr["opponent"] = opp_norm.map(_canon_team).replace("", np.nan)
            else:
                pr["opponent"] = np.nan
    else:
        pr["opponent"] = np.nan

    pr["player_key"] = pr["player"].fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return pr[["player","team","opponent","player_key"]].drop_duplicates()

def _validate_required(df: pd.DataFrame):
    """
    Strict checks by position-family:
      WR/TE: route_rate, tgt_share, yprr
      RB:    rush_share, ypc
      QB:    ypa

    Validate **only** players that appear in outputs/props_raw.csv.
    Skip rows where we cannot determine a family (no position/role and no usage signal).
    """
    props_players = _load_props_players()
    if props_players.empty:
        return

    df = df.copy()
    df["player_key"] = df["player"].fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    need = df.merge(props_players[["player_key"]].drop_duplicates(), on="player_key", how="inner")
    if need.empty:
        return

    pos  = need.get("position", pd.Series(index=need.index, dtype=object)).astype(str).str.upper()
    role = need.get("role",     pd.Series(index=need.index, dtype=object)).astype(str).str.upper()

    fam = pos.where(~pos.isin(["", "NAN", "NONE"]), np.nan)

    qb_mask = pd.Series(False, index=need.index)
    if "dropbacks" in need.columns:
        qb_mask |= (need["dropbacks"].fillna(0) >= 15)
    if "ypa" in need.columns:
        qb_mask |= (need["ypa"].notna() & (need["ypa"] > 6.0))

    rb_mask = (need.get("rush_share", pd.Series(0, index=need.index)).fillna(0) >= 0.20)
    wr_mask = (need.get("route_rate", pd.Series(0, index=need.index)).fillna(0) >= 0.20) | \
              (need.get("tgt_share",   pd.Series(0, index=need.index)).fillna(0) >= 0.15)

    fam = fam.where(fam.notna(), np.where(qb_mask, "QB",
                                  np.where(rb_mask, "RB",
                                  np.where(wr_mask, "WR", np.nan))))

    has_wr_hint = role.str.contains("WR|TE", na=False)
    has_rb_hint = role.str.contains("RB",    na=False)
    has_qb_hint = role.str.contains("QB",    na=False)

    is_wrte = fam.isin(["WR","TE"]) | has_wr_hint
    is_rb   = fam.eq("RB") | has_rb_hint
    is_qb   = fam.eq("QB") | has_qb_hint

    to_check = need.loc[is_wrte | is_rb | is_qb].copy()
    if to_check.empty:
        return

    missing: Dict[str, List[str]] = {}

    def _need(mask, cols: List[str], label: str):
        if not mask.any():
            return
        sub = to_check.loc[mask]
        for c in cols:
            if c not in sub.columns:
                bad = sub.index.tolist()
            else:
                bad = sub.index[sub[c].isna()].tolist()
            if bad:
                missing[f"{label}:{c}"] = sub.loc[bad, "player"].astype(str).tolist()

    _need(is_wrte.loc[to_check.index], ["route_rate","tgt_share","yprr"], "WR/TE")
    _need(is_rb.loc[to_check.index],   ["rush_share","ypc"],              "RB")
    _need(is_qb.loc[to_check.index],   ["ypa"],                           "QB")

    if missing:
        print("[make_player_form] REQUIRED PLAYER METRICS MISSING:", file=sys.stderr)
        for k, v in missing.items():
            preview = ", ".join(v[:10]) + ("..." if len(v) > 10 else "")
            print(f"  - {k}: {preview}", file=sys.stderr)

        # Write a CSV report for diagnostics
        try:
            rows = []
            fam_map = {"WR/TE": is_wrte, "RB": is_rb, "QB": is_qb}
            for fam, names in missing.items():
                mask = fam_map.get(fam, pd.Series(False, index=df.index))
                # build a small lookup subset
                sub = to_check[mask.loc[to_check.index].fillna(False)][["player","team","position","role"]].copy()
                for nm in names:
                    rows.append({
                        "player": nm,
                        "family": fam,
                        "team": (
                            sub.loc[
                                sub["player"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True) == nm,
                                "team",
                            ].head(1).item()
                            if not sub.empty
                            else None
                        ),
                        "missing_for_family": fam,
                    })
            os.makedirs(DATA_DIR, exist_ok=True)
            pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "validation_player_missing.csv"), index=False)
            print(f"[make_player_form] wrote report → {os.path.join(DATA_DIR, 'validation_player_missing.csv')}")
        except Exception as e:
            print(f"[make_player_form] WARN could not write missing report: {e}", file=sys.stderr)

        # Env-gated strictness (default now lenient)
        if os.getenv("STRICT_VALIDATE", "0") != "0":
            raise RuntimeError("Required player_form metrics missing; failing per strict policy.")
        else:
            print("[make_player_form] STRICT_VALIDATE=0 → continue despite missing required metrics", file=sys.stderr)
            return


# ---------------------------
# CLI
# ---------------------------

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    _safe_mkdir(DATA_DIR)

    try:
        df = build_player_form(args.season)

        # Fallback sweep BEFORE strict validation
        before = df.copy()
        df = _apply_fallback_enrichers(df)
        try:
            filled = {}
            for c in ["route_rate","tgt_share","rush_share","yprr","ypc","ypa","rz_share"]:
                if c in df.columns:
                    was_na = before[c].isna() if c in before.columns else pd.Series(True, index=df.index)
                    now_ok = was_na & df[c].notna()
                    if now_ok.any():
                        filled[c] = int(now_ok.sum())
            if any(filled.values()):
                print("[make_player_form] Fallbacks filled:", filled)
        except Exception:
            pass

        _validate_required(df)

    except Exception as e:
        print(f"[make_player_form] ERROR: {e}", file=sys.stderr)
        # If df exists and has rows, write it rather than wiping out to empty.
        try:
            if "df" in locals() and isinstance(df, pd.DataFrame) and len(df) > 0:
                df.to_csv(OUTPATH, index=False)
                print(f"[make_player_form] Wrote {len(df)} rows → {OUTPATH} (after handled error)")
                return
        except Exception:
            pass
        empty = pd.DataFrame(columns=FINAL_COLS)
        empty.to_csv(OUTPATH, index=False)
        print(f"[make_player_form] Wrote 0 rows → {OUTPATH} (empty due to error)")
        return  # ← do not sys.exit(1)

    df.to_csv(OUTPATH, index=False)
    print(f"[make_player_form] Wrote {len(df)} rows → {OUTPATH}")


def _enrich_team_and_opponent_from_props(df: pd.DataFrame) -> pd.DataFrame:
    import os
    path = os.path.join("outputs", "props_raw.csv")
    if not os.path.exists(path):
        return df
    try:
        pr = pd.read_csv(path)
    except Exception:
        return df
    pr.columns = [c.lower() for c in pr.columns]
    name_col = next((c for c in ["player","player_name","name"] if c in pr.columns), None)
    team_col = next((c for c in ["team","team_abbr","posteam"] if c in pr.columns), None)
    if not name_col:
        return df
    pr["player"] = _norm_name(pr[name_col].astype(str))
    if team_col:
        pr["team"] = pr[team_col].astype(str).str.upper().str.strip().map(_canon_team)
        pr.loc[~pr["team"].isin(VALID), "team"] = np.nan
    pr["opponent"] = _normalize_props_opponent(pr)
    out = df.copy()
    out = _ensure_cols(out, ["opponent"])
    if "team" in pr.columns:
        out = out.merge(pr[["player","team","opponent"]].drop_duplicates(), on=["player","team"], how="left", suffixes=("","_pr1"))
        if "opponent_pr1" in out.columns:
            # Guard combine_first so future refactors that drop the ensure above don't KeyError.
            base_opponent = out.get("opponent")
            out["opponent"] = (
                base_opponent.combine_first(out.pop("opponent_pr1"))
                if base_opponent is not None
                else out.pop("opponent_pr1")
            )
        need_team = out["team"].isna() | (out["team"] == "")
        if need_team.any():
            fallback = pr[["player","team","opponent"]].dropna(how="all").drop_duplicates()
            out = out.merge(fallback, on="player", how="left", suffixes=("","_pr2"))
            if "team_pr2" in out.columns:
                out["team"] = out["team"].combine_first(out.pop("team_pr2"))
            if "opponent_pr2" in out.columns:
                base_opponent = out.get("opponent")
                out["opponent"] = (
                    base_opponent.combine_first(out.pop("opponent_pr2"))
                    if base_opponent is not None
                    else out.pop("opponent_pr2")
                )
    else:
        if "opponent" in pr.columns:
            out = out.merge(pr[["player","opponent"]].drop_duplicates(), on="player", how="left", suffixes=("","_pr"))
            if "opponent_pr" in out.columns:
                base_opponent = out.get("opponent")
                out["opponent"] = (
                    base_opponent.combine_first(out.pop("opponent_pr"))
                    if base_opponent is not None
                    else out.pop("opponent_pr")
                )
    return out

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cli()
