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

Behavior:
- Primary build from nflverse PBP (real data only).
- Fallback sweep (non-destructive merges) BEFORE strict validation:
    data/espn_player_form.csv,
    data/msf_player_form.csv,
    data/apisports_player_form.csv,
    data/nflgsis_player_form.csv
- Strict validator then fails if key per-position metrics are still missing.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPATH = os.path.join(DATA_DIR, "player_form.csv")


# ---------------------------
# Utilities
# ---------------------------

def _safe_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

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
# Build from PBP (real data only)
# ---------------------------

def build_player_form(season: int = 2025) -> pd.DataFrame:
    pbp = load_pbp(season)
    if pbp.empty:
        raise RuntimeError("PBP empty; cannot compute player form.")

    # offense team col
    off_col = "posteam" if "posteam" in pbp.columns else ("offense_team" if "offense_team" in pbp.columns else None)
    if off_col is None:
        raise RuntimeError("No offense team column in PBP (posteam/offense_team).")

    # ---------------- RECEIVING ----------------
    # robust pass flag (real schema only)
    is_pass = pbp.get("pass")
    if is_pass is None:
        pt = pbp.get("play_type")
        is_pass = pt.isin(["pass", "no_play"]) if pt is not None else pd.Series(False, index=pbp.index)
    else:
        is_pass = is_pass.astype(bool)

    rec = pbp.loc[is_pass].copy()
    rcv_name_col = (
        "receiver_player_name"
        if "receiver_player_name" in rec.columns
        else ("receiver" if "receiver" in rec.columns else None)
    )
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

    rply = rec.groupby(["team", "player"], dropna=False).agg(
        targets=("pass_attempt", "sum") if "pass_attempt" in rec.columns else ("player", "size"),
        rec_yards=("yards_gained", "sum"),
        receptions=("complete_pass", "sum") if "complete_pass" in rec.columns else (rcv_name_col, "size"),
    ).reset_index()

    rply = rply.merge(team_targets.reset_index(), on="team", how="left")
    rply = rply.merge(team_dropbacks.reset_index(), on="team", how="left")
    rply["tgt_share"] = np.where(rply["team_targets"] > 0, rply["targets"] / rply["team_targets"], np.nan)
    rply["route_rate"] = np.where(rply["team_dropbacks"] > 0, rply["targets"] / rply["team_dropbacks"], np.nan).clip(0.05, 0.95)
    rply["ypt"] = np.where(rply["targets"] > 0, rply["rec_yards"] / rply["targets"], np.nan)
    rply["receptions_per_target"] = np.where(rply["targets"] > 0, rply["receptions"] / rply["targets"], np.nan)
    routes_proxy = (rply["team_dropbacks"] * rply["route_rate"]).replace(0, np.nan)
    rply["yprr"] = np.where(routes_proxy > 0, rply["rec_yards"] / routes_proxy, np.nan)

    # RZ receiving (inside 20)
    inside20 = rec.copy()
    inside20["yardline_100"] = pd.to_numeric(inside20.get("yardline_100"), errors="coerce")
    rz_rec = inside20.loc[inside20["yardline_100"] <= 20]
    rz_tgt_ply = rz_rec.groupby(["team", "player"]).size().rename("rz_targets")
    rz_tgt_tm = rz_rec.groupby("team").size().rename("rz_team_targets")
    rply = rply.merge(rz_tgt_ply.reset_index(), on=["team", "player"], how="left")
    rply = rply.merge(rz_tgt_tm.reset_index(), on="team", how="left")
    rply["rz_tgt_share"] = np.where(rply["rz_team_targets"] > 0, rply["rz_targets"] / rply["rz_team_targets"], np.nan)

    # ---------------- RUSHING ----------------
    is_rush = pbp.get("rush")
    if is_rush is None:
        pt = pbp.get("play_type")
        is_rush = pt.eq("run") if pt is not None else pd.Series(False, index=pbp.index)
    else:
        is_rush = is_rush.astype(bool)

    ru = pbp.loc[is_rush].copy()
    rush_name_col = (
        "rusher_player_name"
        if "rusher_player_name" in ru.columns
        else ("rusher" if "rusher" in ru.columns else None)
    )
    if rush_name_col is None:
        ru["rusher_player_name"] = np.nan
        rush_name_col = "rusher_player_name"

    ru["player"] = _norm_name(ru[rush_name_col].fillna(""))
    ru["team"] = ru[off_col].astype(str).str.upper().str.strip()

    team_rushes = ru.groupby("team", dropna=False).size().rename("team_rushes").astype(float)
    rru = ru.groupby(["team", "player"], dropna=False).agg(
        rushes=("rush_attempt", "sum") if "rush_attempt" in ru.columns else ("player", "size"),
        rush_yards=("yards_gained", "sum"),
    ).reset_index()
    rru = rru.merge(team_rushes.reset_index(), on="team", how="left")
    rru["rush_share"] = np.where(rru["team_rushes"] > 0, rru["rushes"] / rru["team_rushes"], np.nan)
    rru["ypc"] = np.where(rru["rushes"] > 0, rru["rush_yards"] / rru["rushes"], np.nan)

    # RZ rushing (inside 10)
    inside10 = ru.copy()
    inside10["yardline_100"] = pd.to_numeric(inside10.get("yardline_100"), errors="coerce")
    rz_ru = inside10.loc[inside10["yardline_100"] <= 10]
    rz_ru_ply = rz_ru.groupby(["team", "player"]).size().rename("rz_rushes")
    rz_ru_tm = rz_ru.groupby("team").size().rename("rz_team_rushes")
    rru = rru.merge(rz_ru_ply.reset_index(), on=["team", "player"], how="left")
    rru = rru.merge(rz_ru_tm.reset_index(), on="team", how="left")
    rru["rz_rush_share"] = np.where(rru["rz_team_rushes"] > 0, rru["rz_rushes"] / rru["rz_team_rushes"], np.nan)

    # ---------------- QUARTERBACK ----------------
    qb_name_col = (
        "passer_player_name"
        if "passer_player_name" in pbp.columns
        else ("passer" if "passer" in pbp.columns else None)
    )
    qb_df = pd.DataFrame(columns=["team", "player", "ypa"])
    if qb_name_col is not None:
        qb = pbp.copy()
        qb["player"] = _norm_name(qb[qb_name_col].fillna(""))
        qb["team"] = qb[off_col].astype(str).str.upper().str.strip()
        gb = qb.groupby(["team", "player"], dropna=False).agg(
            pass_yards=("yards_gained", "sum"),
            pass_att=("pass_attempt", "sum") if "pass_attempt" in qb.columns else (qb_name_col, "size"),
        ).reset_index()
        gb["ypa"] = np.where(gb["pass_att"] > 0, gb["pass_yards"] / gb["pass_att"], np.nan)
        qb_df = gb[["team", "player", "ypa"]]

    # Merge all
    base = pd.merge(rply, rru, on=["team", "player"], how="outer")
    base = pd.merge(base, qb_df, on=["team", "player"], how="left")
    base["rz_share"] = base[["rz_tgt_share", "rz_rush_share"]].max(axis=1)

    # attach season + placeholders (position/role can be merged later)
    base["season"] = int(season)
    base["position"] = np.nan
    base["role"] = np.nan

    final_cols = [
        "player", "team", "season", "position", "role",
        "tgt_share", "route_rate", "rush_share",
        "yprr", "ypt", "ypc", "ypa",
        "receptions_per_target",
        "rz_share", "rz_tgt_share", "rz_rush_share",
    ]
    base = _ensure_cols(base, final_cols)
    out = base[final_cols].drop_duplicates(subset=["player", "team", "season"]).reset_index(drop=True)
    return out


# ---------------------------
# Fallback sweep (real CSV enrichers) + strict validation
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
            # normalize keys
            if "player" not in ext.columns and "player_name" in ext.columns:
                ext = ext.rename(columns={"player_name": "player"})
            if not {"player", "team"}.issubset(ext.columns):
                continue
            ext["player"] = ext["player"].astype(str).str.strip()
            ext["team"] = ext["team"].astype(str).str.upper().str.strip()
            out = _non_destructive_merge(out, ext, keys=["player", "team"])
        except Exception:
            # best-effort; keep going
            continue
    return out

def _validate_required(df: pd.DataFrame):
    """
    Strict checks by position-family:
      WR/TE: route_rate, tgt_share, yprr
      RB:    rush_share, ypc
      QB:    ypa
    If position is missing, we use 'role' to infer family.
    """
    pos = df.get("position", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    role = df.get("role", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()

    is_wrte = pos.isin(["WR", "TE"]) | role.str.contains("WR|TE", na=False)
    is_rb   = pos.eq("RB") | role.str.contains("RB", na=False)
    is_qb   = pos.eq("QB") | role.str.contains("QB", na=False)

    missing: Dict[str, List[str]] = {}

    def _need(mask, cols: List[str], label: str):
        if not mask.any():
            return
        sub = df.loc[mask]
        for c in cols:
            if c not in sub.columns:
                bad = sub.index.tolist()
            else:
                bad = sub.index[sub[c].isna()].tolist()
            if bad:
                missing[f"{label}:{c}"] = sub.loc[bad, "player"].astype(str).tolist()

    _need(is_wrte, ["route_rate", "tgt_share", "yprr"], "WR/TE")
    _need(is_rb,   ["rush_share", "ypc"],                "RB")
    _need(is_qb,   ["ypa"],                              "QB")

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
            # quick report of fills
            filled = {}
            for c in ["route_rate", "tgt_share", "rush_share", "yprr", "ypc", "ypa", "rz_share"]:
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
        empty = pd.DataFrame(columns=[
            "player","team","season","position","role",
            "tgt_share","route_rate","rush_share",
            "yprr","ypt","ypc","ypa","receptions_per_target",
            "rz_share","rz_tgt_share","rz_rush_share",
        ])
        empty.to_csv(OUTPATH, index=False)
        sys.exit(1)

    df.to_csv(OUTPATH, index=False)
    print(f"[make_player_form] Wrote {len(df)} rows → {OUTPATH}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cli()
