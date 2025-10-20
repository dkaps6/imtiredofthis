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

FINAL_COLS = [
    "player",
    "team",
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
    """
    Merge depth chart roles from ESPN and Ourlads into player_form.
    Keeps your original position/role if already present.
    """
    import re
    try:
        roles_espn = pd.read_csv(os.path.join(DATA_DIR, "roles_espn.csv"))
    except Exception:
        roles_espn = pd.DataFrame(columns=["team","player","role"])
    try:
        roles_ourlads = pd.read_csv(os.path.join(DATA_DIR, "roles_ourlads.csv"))
    except Exception:
        roles_ourlads = pd.DataFrame(columns=["team","player","role"])

    roles = pd.concat([roles_espn, roles_ourlads], ignore_index=True)
    if roles.empty:
        print("[make_player_form] No roles_espn or roles_ourlads found, skipping merge.")
        return pf

    # normalize player/team
    roles["player"] = roles["player"].astype(str).str.replace(".", "", regex=False).str.strip()
    roles["team"] = roles["team"].astype(str).str.upper().str.strip()
    roles["role"] = roles["role"].astype(str).str.upper().str.strip()
    roles["position"] = roles["role"].str.replace(r"\d+$", "", regex=True)

    # cleanup: strip jersey prefixes; drop numeric-only rows
    roles["player"] = roles["player"].str.replace(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", "", regex=True)
    roles = roles[~roles["player"].str.fullmatch(r"\d+")]

    # prefer best depth per (team,player)
    def _rank(r):
        m = re.search(r"(\d+)$", str(r))
        return int(m.group(1)) if m else 999

    roles["_rank"] = roles["role"].map(_rank)
    roles = (
        roles.sort_values(["team", "player", "_rank"])
        .drop_duplicates(["team", "player"], keep="first")
        .drop(columns=["_rank"])
    )

    pf = pf.merge(roles, on=["player", "team"], how="left", suffixes=("", "_depth"))
    if "position_depth" in pf.columns:
        pf["position"] = pf["position"].combine_first(pf["position_depth"])
    if "role_depth" in pf.columns:
        pf["role"] = pf["role"].combine_first(pf["role_depth"])
    pf.drop(columns=[c for c in pf.columns if c.endswith("_depth")], inplace=True, errors="ignore")

    try:
        cov = pf["position"].notna().mean()
        print(f"[make_player_form] merged depth roles → coverage now {cov:.2%}")
    except Exception:
        pass
    return pf
# === END ADDITION ===

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

def load_pbp(season: int) -> pd.DataFrame:
    if NFL_PKG == "nflreadpy":
        raw = NFLV.load_pbp(seasons=[season])
    else:
        raw = NFLV.import_pbp_data([season], downcast=True)  # type: ignore
    pbp = _to_pandas(raw)
    pbp.columns = [c.lower() for c in pbp.columns]
    return pbp

# ---------------------------
# Position sources
# ---------------------------

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
    need = {"player","team","role"}
    if not need.issubset(r.columns):
        return df
    r["player"] = _norm_name(r["player"].astype(str))
    r["team"] = r["team"].astype(str).str.upper().str.strip().map(_canon_team)
    r = r[r["team"].isin(VALID)]
    out = df.merge(r[["player","team","role"]], on=["player","team"], how="left", suffixes=("","_roles"))
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
    for team, g in pf.groupby("team", dropna=False):
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
        warnings.warn(
            f"No play-by-play data available for season {season}; returning empty base for fallback hydration.",
            RuntimeWarning,
        )
        base = pd.DataFrame(columns=["player", "team"])
        base["season"] = int(season)
        base = _ensure_cols(base, FINAL_COLS)
        base = base[FINAL_COLS].drop_duplicates(subset=["player","team","season"]).reset_index(drop=True)
    # merge depth roles (non-destructive)
    try:
        base = _merge_depth_roles(base)
    except Exception:
        try:
            pf = _merge_depth_roles(pf)
        except Exception:
            pass

        return base

    off_col = "posteam" if "posteam" in pbp.columns else ("offense_team" if "offense_team" in pbp.columns else None)
    if off_col is None:
        raise RuntimeError("No offense team column in PBP (posteam/offense_team).")

    # RECEIVING
    is_pass = pbp.get("pass")
    if is_pass is None:
        pt = pbp.get("play_type")
        is_pass = pt.isin(["pass","no_play"]) if pt is not None else pd.Series(False, index=pbp.index)
    else:
        is_pass = is_pass.astype(bool)

    rec = pbp.loc[is_pass].copy()
    if rec.empty:
        rply = pd.DataFrame(columns=["team","player"])
    else:
        rcv_name_col = "receiver_player_name" if "receiver_player_name" in rec.columns else ("receiver" if "receiver" in rec.columns else None)
        if rcv_name_col is None:
            rec["receiver_player_name"] = np.nan
            rcv_name_col = "receiver_player_name"
        rec["player"] = _norm_name(rec[rcv_name_col].fillna(""))
        rec["team"] = rec[off_col].astype(str).str.upper().str.strip()

        team_targets = rec.groupby("team", dropna=False).size().rename("team_targets").astype(float)
        if "qb_dropback" in rec.columns:
            team_dropbacks = rec.groupby("team", dropna=False)["qb_dropback"].sum(min_count=1).rename("team_dropbacks")
        else:
            team_dropbacks = rec.groupby("team", dropna=False).size().rename("team_dropbacks").astype(float)

        rply = rec.groupby(["team","player"], dropna=False).agg(
            targets=("pass_attempt","sum") if "pass_attempt" in rec.columns else ("player","size"),
            rec_yards=("yards_gained","sum"),
            receptions=("complete_pass","sum") if "complete_pass" in rec.columns else (rcv_name_col,"size"),
        ).reset_index()
        rply = rply.merge(team_targets.reset_index(), on="team", how="left")
        rply = rply.merge(team_dropbacks.reset_index(), on="team", how="left")
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
            rz_tgt_ply = rz_rec.groupby(["team","player"]).size().rename("rz_targets")
            rz_tgt_tm  = rz_rec.groupby("team").size().rename("rz_team_targets")
            rply = rply.merge(rz_tgt_ply.reset_index(), on=["team","player"], how="left")
            rply = rply.merge(rz_tgt_tm.reset_index(),  on="team",          how="left")
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
    if ru.empty:
        rru = pd.DataFrame(columns=["team","player"])
    else:
        rush_name_col = "rusher_player_name" if "rusher_player_name" in ru.columns else ("rusher" if "rusher" in ru.columns else None)
        if rush_name_col is None:
            ru["rusher_player_name"] = np.nan
            rush_name_col = "rusher_player_name"

        ru["player"] = _norm_name(ru[rush_name_col].fillna(""))
        ru["team"] = ru[off_col].astype(str).str.upper().str.strip()

        team_rushes = ru.groupby("team", dropna=False).size().rename("team_rushes").astype(float)
        rru = ru.groupby(["team","player"], dropna=False).agg(
            rushes=("rush_attempt","sum") if "rush_attempt" in ru.columns else ("player","size"),
            rush_yards=("yards_gained","sum"),
        ).reset_index()
        rru = rru.merge(team_rushes.reset_index(), on="team", how="left")
        rru["rush_share"] = np.where(rru["team_rushes"]>0, rru["rushes"]/rru["team_rushes"], np.nan)
        rru["ypc"] = np.where(rru["rushes"]>0, rru["rush_yards"]/rru["rushes"], np.nan)

        inside10 = ru.copy()
        inside10["yardline_100"] = pd.to_numeric(inside10.get("yardline_100"), errors="coerce")
        rz_ru = inside10.loc[inside10["yardline_100"] <= 10]
        if not rz_ru.empty:
            rz_ru_ply = rz_ru.groupby(["team","player"]).size().rename("rz_rushes")
            rz_ru_tm  = rz_ru.groupby("team").size().rename("rz_team_rushes")
            rru = rru.merge(rz_ru_ply.reset_index(), on=["team","player"], how="left")
            rru = rru.merge(rz_ru_tm.reset_index(),  on="team",          how="left")
            rru["rz_rush_share"] = np.where(rru["rz_team_rushes"]>0, rru["rz_rushes"]/rru["rz_team_rushes"], np.nan)

    rru = _ensure_cols(rru, [
        "rushes","rush_yards","team_rushes","rush_share","ypc",
        "rz_rushes","rz_team_rushes","rz_rush_share",
    ])

    # QUARTERBACK
    qb_df = pd.DataFrame(columns=["team","player","ypa","dropbacks"])
    qb_name_col = "passer_player_name" if "passer_player_name" in pbp.columns else ("passer" if "passer" in pbp.columns else None)
    if qb_name_col is not None:
        qb = pbp.copy()
        qb["player"] = _norm_name(qb[qb_name_col].fillna(""))
        qb["team"] = qb[off_col].astype(str).str.upper().str.strip()
        gb = qb.groupby(["team","player"], dropna=False).agg(
            pass_yards=("yards_gained","sum"),
            pass_att=("pass_attempt","sum") if "pass_attempt" in qb.columns else (qb_name_col,"size"),
            dropbacks=("qb_dropback","sum") if "qb_dropback" in qb.columns else (qb_name_col,"size"),
        ).reset_index()
        gb["ypa"] = np.where(gb["pass_att"]>0, gb["pass_yards"]/gb["pass_att"], np.nan)
        qb_df = gb[["team","player","ypa","dropbacks"]]

    # Merge all
    base = pd.merge(rply, rru, on=["team","player"], how="outer")
    base = pd.merge(base, qb_df, on=["team","player"], how="left")
    base["rz_share"] = base[["rz_tgt_share","rz_rush_share"]].max(axis=1)
    base["season"] = int(season)

    # Initialize position/role as NaN (do not uppercase yet)
    base["position"] = np.nan
    base["role"] = np.nan

    # Normalize keys
    base["player"] = _norm_name(base["player"].astype(str))
    base["team"] = base["team"].astype(str).str.upper().str.strip().map(_canon_team)

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
    out = base[FINAL_COLS].drop_duplicates(subset=["player","team","season"]).reset_index(drop=True)
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
        return pd.DataFrame(columns=["player","team","player_key"])
    try:
        pr = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["player","team","player_key"])

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

    pr["player_key"] = pr["player"].fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return pr[["player","team","player_key"]].drop_duplicates()

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
        raise RuntimeError("Required player_form metrics missing; failing per strict policy.")

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
        empty = pd.DataFrame(columns=FINAL_COLS)
        empty.to_csv(OUTPATH, index=False)
        sys.exit(1)

    df.to_csv(OUTPATH, index=False)
    print(f"[make_player_form] Wrote {len(df)} rows → {OUTPATH}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cli()
