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

Notes
- Route participation is proxied from targets/dropbacks if true routes aren’t available;
  enrich_player_form.py may overwrite with better participation.
- Red-zone shares: receiving inside 20; rushing inside 10 (goal-line signal).
- Writes schema-correct CSV and exits(1) on fatal errors so strict validator can fail the run.
"""

from __future__ import annotations
import os, sys, warnings
from typing import Tuple
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPATH = os.path.join(DATA_DIR, "player_form.csv")

def _safe_mkdir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

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
                "Neither nflreadpy nor nfl_data_py is available. Please `pip install nflreadpy`."
            ) from e

NFLV, NFL_PKG = _import_nflverse()

def load_pbp_2025() -> pd.DataFrame:
    if NFL_PKG == "nflreadpy":
        pbp = NFLV.load_pbp(seasons=[2025])
    else:
        pbp = NFLV.import_pbp_data([2025], downcast=True)  # type: ignore
    pbp.columns = [c.lower() for c in pbp.columns]
    return pbp

def _norm_name(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.replace(".", "", regex=False).str.strip()
    return out

def _ensure_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def build_player_form() -> pd.DataFrame:
    pbp = load_pbp_2025()
    if pbp.empty:
        raise RuntimeError("PBP empty; cannot compute player form for 2025.")

    # Identify team columns
    off_col = "posteam" if "posteam" in pbp.columns else ("offense_team" if "offense_team" in pbp.columns else None)
    if off_col is None:
        raise RuntimeError("No offense team column in PBP.")

    # RECEIVING (targets, yards, receptions)
    is_pass = pbp.get("pass", pd.Series(False, index=pbp.index)).astype(bool)
    rec = pbp.loc[is_pass].copy()
    rcv_name_col = "receiver_player_name" if "receiver_player_name" in rec.columns else ("receiver" if "receiver" in rec.columns else None)
    if rcv_name_col is None:
        rec["receiver_player_name"] = np.nan
        rcv_name_col = "receiver_player_name"

    rec["player"] = _norm_name(rec[rcv_name_col].fillna(""))
    rec["team"] = rec[off_col].astype(str).str.upper().str.strip()

    # Team totals
    team_targets = rec.groupby("team", dropna=False).size().rename("team_targets").astype(float)
    if "qb_dropback" in rec.columns:
        team_dropbacks = rec.groupby("team", dropna=False)["qb_dropback"].sum(min_count=1).rename("team_dropbacks")
    else:
        team_dropbacks = rec.groupby("team", dropna=False).size().rename("team_dropbacks").astype(float)

    # Player receiving
    rply = rec.groupby(["team","player"], dropna=False).agg(
        targets=("pass_attempt","sum") if "pass_attempt" in rec.columns else ("player","size"),
        rec_yards=("yards_gained","sum"),
        receptions=("complete_pass","sum") if "complete_pass" in rec.columns else ("passer_player_name","size")
    ).reset_index()

    # Merge team totals
    rply = rply.merge(team_targets.reset_index(), on="team", how="left")
    rply = rply.merge(team_dropbacks.reset_index(), on="team", how="left")
    # Rates / efficiency
    rply["tgt_share"] = np.where(rply["team_targets"]>0, rply["targets"]/rply["team_targets"], np.nan)
    rply["route_rate"] = np.where(rply["team_dropbacks"]>0, rply["targets"]/rply["team_dropbacks"], np.nan).clip(0.05, 0.95)
    rply["ypt"] = np.where(rply["targets"]>0, rply["rec_yards"]/rply["targets"], np.nan)
    rply["receptions_per_target"] = np.where(rply["targets"]>0, rply["receptions"]/rply["targets"], np.nan)

    # routes_proxy -> yprr
    routes_proxy = (rply["team_dropbacks"] * rply["route_rate"]).replace(0, np.nan)
    rply["yprr"] = np.where(routes_proxy>0, rply["rec_yards"]/routes_proxy, np.nan)

    # RZ receiving (inside 20)
    inside20 = rec.copy()
    inside20["yardline_100"] = pd.to_numeric(inside20.get("yardline_100"), errors="coerce")
    rz_rec = inside20.loc[inside20["yardline_100"]<=20]
    rz_tgt_ply = rz_rec.groupby(["team","player"]).size().rename("rz_targets")
    rz_tgt_tm = rz_rec.groupby("team").size().rename("rz_team_targets")
    rply = rply.merge(rz_tgt_ply.reset_index(), on=["team","player"], how="left")
    rply = rply.merge(rz_tgt_tm.reset_index(), on="team", how="left")
    rply["rz_tgt_share"] = np.where(rply["rz_team_targets"]>0, rply["rz_targets"]/rply["rz_team_targets"], np.nan)

    # RUSHING (carries, yards)
    is_rush = pbp.get("rush", pd.Series(False, index=pbp.index)).astype(bool)
    ru = pbp.loc[is_rush].copy()
    rush_name_col = "rusher_player_name" if "rusher_player_name" in ru.columns else ("rusher" if "rusher" in ru.columns else None)
    if rush_name_col is None:
        ru["rusher_player_name"] = np.nan
        rush_name_col = "rusher_player_name"

    ru["player"] = _norm_name(ru[rush_name_col].fillna(""))
    ru["team"] = ru[off_col].astype(str).str.upper().str.strip()

    team_rushes = ru.groupby("team", dropna=False).size().rename("team_rushes").astype(float)
    rru = ru.groupby(["team","player"], dropna=False).agg(
        rushes=("rush_attempt","sum") if "rush_attempt" in ru.columns else ("player","size"),
        rush_yards=("yards_gained","sum")
    ).reset_index()
    rru = rru.merge(team_rushes.reset_index(), on="team", how="left")
    rru["rush_share"] = np.where(rru["team_rushes"]>0, rru["rushes"]/rru["team_rushes"], np.nan)
    rru["ypc"] = np.where(rru["rushes"]>0, rru["rush_yards"]/rru["rushes"], np.nan)

    # RZ rushing (inside 10)
    inside10 = ru.copy()
    inside10["yardline_100"] = pd.to_numeric(inside10.get("yardline_100"), errors="coerce")
    rz_ru = inside10.loc[inside10["yardline_100"]<=10]
    rz_ru_ply = rz_ru.groupby(["team","player"]).size().rename("rz_rushes")
    rz_ru_tm = rz_ru.groupby("team").size().rename("rz_team_rushes")
    rru = rru.merge(rz_ru_ply.reset_index(), on=["team","player"], how="left")
    rru = rru.merge(rz_ru_tm.reset_index(), on="team", how="left")
    rru["rz_rush_share"] = np.where(rru["rz_team_rushes"]>0, rru["rz_rushes"]/rru["rz_team_rushes"], np.nan)

    # QUARTERBACK (ypa)
    qb_name_col = "passer_player_name" if "passer_player_name" in pbp.columns else ("passer" if "passer" in pbp.columns else None)
    qb_df = pd.DataFrame(columns=["team","player","ypa"])
    if qb_name_col is not None:
        qb = pbp.copy()
        qb["player"] = _norm_name(qb[qb_name_col].fillna(""))
        qb["team"] = qb[off_col].astype(str).str.upper().str.strip()
        gb = qb.groupby(["team","player"], dropna=False).agg(
            pass_yards=("yards_gained","sum"),
            pass_att=("pass_attempt","sum") if "pass_attempt" in qb.columns else (qb_name_col, "size")
        ).reset_index()
        gb["ypa"] = np.where(gb["pass_att"]>0, gb["pass_yards"]/gb["pass_att"], np.nan)
        qb_df = gb[["team","player","ypa"]]

    # Merge all
    base = pd.merge(rply, rru, on=["team","player"], how="outer", suffixes=("",""))
    base = pd.merge(base, qb_df, on=["team","player"], how="left")

    # Unified RZ share
    base["rz_share"] = base[["rz_tgt_share","rz_rush_share"]].max(axis=1)

    # Compose final schema; leave position/role blank to be filled later by enrich step
    base["season"] = 2025
    base["position"] = np.nan
    base["role"] = np.nan

    final_cols = [
        "player","team","season","position","role",
        "tgt_share","route_rate","rush_share",
        "yprr","ypt","ypc","ypa",
        "receptions_per_target",
        "rz_share","rz_tgt_share","rz_rush_share"
    ]
    # Ensure presence
    base = _ensure_cols(base, final_cols)
    out = base[final_cols].drop_duplicates(subset=["player","team","season"]).reset_index(drop=True)
    return out

def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _safe_mkdir(DATA_DIR)
        try:
            df = build_player_form()
        except Exception as e:
            print(f"[make_player_form] ERROR: {e}", file=sys.stderr)
            # still write an empty schema so validator controls fail/pass
            empty = pd.DataFrame(columns=[
                "player","team","season","position","role",
                "tgt_share","route_rate","rush_share",
                "yprr","ypt","ypc","ypa","receptions_per_target",
                "rz_share","rz_tgt_share","rz_rush_share"
            ])
            empty.to_csv(OUTPATH, index=False)
            sys.exit(1)
        df.to_csv(OUTPATH, index=False)
        print(f"[make_player_form] Wrote {len(df)} rows → {OUTPATH}")

if __name__ == "__main__":
    main()
