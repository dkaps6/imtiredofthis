#!/usr/bin/env python3
from __future__ import annotations
import sys, os, time, traceback
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/player_form.csv")
LOG_DIR = Path("logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
ERR_LOG = LOG_DIR / "nfl_pbp_error.txt"

def _safe_write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame(columns=[
            "player","team","position","role",
            "target_share","rush_share","route_rate",
            "rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","qb_ypa"
        ]).to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)

# --- PBP fetcher: prefer nflreadpy (2025), fall back to nfl_data_py; 404 -> prior season ---
def _fetch_pbp_with_retry(season: int, tries: int = 3, wait: int = 4) -> pd.DataFrame:
    seasons_to_try = [season, season - 1, season - 2]
    last = None
    try:
        import nflreadpy as nfr
        use_readpy = True
    except Exception:
        use_readpy = False
        import nfl_data_py as nfl  # noqa: F401

    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== PBP fetch wanted season={season} ===\n")

    for s in seasons_to_try:
        for i in range(1, tries + 1):
            try:
                if use_readpy:
                    pf = nfr.load_pbp([s])
                    df = pf.to_pandas()
                else:
                    df = nfl.import_pbp_data([s])

                if isinstance(df, pd.DataFrame) and len(df):
                    with open(ERR_LOG, "a", encoding="utf-8") as f:
                        f.write(f"season {s} try {i}: OK rows={len(df)} via "
                                f"{'nflreadpy' if use_readpy else 'nfl_data_py'}\n")
                    if s != season:
                        print(f"[player_form] NOTE: using season {s} as fallback for {season}", flush=True)
                    return df
                raise RuntimeError("empty dataframe")
            except Exception as e:
                last = e
                with open(ERR_LOG, "a", encoding="utf-8") as f:
                    f.write(f"season {s} try {i}: {type(e).__name__}: {e}\n{traceback.format_exc()}\n")
                if (not use_readpy) and ("HTTP Error 404" in str(e)):
                    break
                time.sleep(wait)
    raise RuntimeError(f"PBP fetch failed: {type(last).__name__}: {last}")

def build_from_nflverse(season: int) -> pd.DataFrame:
    pbp = _fetch_pbp_with_retry(season)
    pbp = pbp.loc[pbp["season"]==season].copy()

    pbp["posteam"] = pbp["posteam"].astype(str).str.upper()
    pbp["receiver"] = pbp.get("receiver_player_name","").fillna("").astype(str)
    pbp["rusher"]   = pbp.get("rusher_player_name","").fillna("").astype(str)
    pbp["passer"]   = pbp.get("passer_player_name","").fillna("").astype(str)

    pbp["is_pass"] = (pbp.get("pass",0)==1) | (pbp.get("pass_attempt",0)==1) | (pbp.get("play_type","")=="pass")
    pbp["is_rush"] = (pbp.get("rush",0)==1) | (pbp.get("play_type","")=="run")
    pbp["in_rz"]   = (pbp.get("yardline_100").fillna(100) <= 20)

    # team totals
    targs = pbp.loc[pbp["is_pass"] & (pbp["receiver"]!=""), ["game_id","posteam","receiver"]].copy()
    targs["targets"] = 1
    team_tgts = targs.groupby(["game_id","posteam"])["targets"].sum().rename("team_targets")
    rec_tgts  = targs.groupby(["game_id","posteam","receiver"])["targets"].sum().rename("player_targets")

    rush = pbp.loc[pbp["is_rush"] & (pbp["rusher"]!=""), ["game_id","posteam","rusher","yards_gained"]].copy()
    rush["rush_att"] = 1
    team_rush = rush.groupby(["game_id","posteam"])["rush_att"].sum().rename("team_rush_att")
    ply_rush  = rush.groupby(["game_id","posteam","rusher"])["rush_att"].sum().rename("player_rush_att")
    ply_rush_yds = rush.groupby(["game_id","posteam","rusher"])["yards_gained"].sum().rename("rush_yards")

    rec_yards = pbp.loc[pbp["receiver"]!=""].groupby(["game_id","posteam","receiver"])["yards_gained"].sum().rename("rec_yards")

    rz_targs = pbp.loc[pbp["is_pass"] & pbp["in_rz"] & (pbp["receiver"]!=""), ["game_id","posteam","receiver"]].copy()
    rz_targs["rz_tgt"] = 1
    ply_rz_tgt = rz_targs.groupby(["game_id","posteam","receiver"])["rz_tgt"].sum().rename("player_rz_tgt")
    team_rz_tgt = rz_targs.groupby(["game_id","posteam"])["rz_tgt"].sum().rename("team_rz_tgt")

    rz_rush = pbp.loc[pbp["is_rush"] & pbp["in_rz"] & (pbp["rusher"]!=""), ["game_id","posteam","rusher"]].copy()
    rz_rush["rz_carry"] = 1
    ply_rz_carry = rz_rush.groupby(["game_id","posteam","rusher"])["rz_carry"].sum().rename("player_rz_carry")
    team_rz_carry = rz_rush.groupby(["game_id","posteam"])["rz_carry"].sum().rename("team_rz_carry")

    pass_att = pbp.loc[pbp["is_pass"] & (pbp["passer"]!=""), ["game_id","posteam","passer","yards_gained"]].copy()
    pass_att["att"] = 1
    team_pass_yds = pass_att.groupby(["game_id","posteam"])["yards_gained"].sum().rename("team_pass_yds")
    team_atts = pass_att.groupby(["game_id","posteam"])["att"].sum().rename("team_pass_att")
    team_ypa = (team_pass_yds.groupby("posteam").sum() / team_atts.groupby("posteam").sum()).rename("qb_ypa_all")

    rec_level  = rec_tgts.reset_index().rename(columns={"receiver":"player"})
    rush_level = ply_rush.reset_index().rename(columns={"rusher":"player"})
    players = pd.concat(
        [rec_level[["game_id","posteam","player","player_targets"]],
         rush_level[["game_id","posteam","player","player_rush_att"]]],
        ignore_index=True
    ).fillna(0)

    merges = [
        team_tgts.reset_index(),
        team_rush.reset_index(),
        ply_rz_tgt.reset_index().rename(columns={"receiver":"player"}),
        team_rz_tgt.reset_index(),
        ply_rz_carry.reset_index().rename(columns={"rusher":"player"}),
        team_rz_carry.reset_index(),
        rec_yards.reset_index().rename(columns={"receiver":"player"}),
        ply_rush_yds.reset_index().rename(columns={"rusher":"player"}),
    ]
    for m in merges:
        players = players.merge(m, on=[c for c in ["game_id","posteam","player"] if c in m.columns], how="left")

    for c in ["player_targets","team_targets","player_rush_att","team_rush_att",
              "player_rz_tgt","team_rz_tgt","player_rz_carry","team_rz_carry",
              "rec_yards","rush_yards"]:
        if c not in players.columns: players[c]=0
        players[c] = players[c].fillna(0)

    agg = players.groupby(["posteam","player"]).sum(numeric_only=True).reset_index()
    agg["team"] = agg["posteam"].astype(str).str.upper()

    agg["target_share"]   = (agg["player_targets"] / agg["team_targets"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["rush_share"]     = (agg["player_rush_att"] / agg["team_rush_att"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["rz_tgt_share"]   = (agg["player_rz_tgt"] / agg["team_rz_tgt"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["rz_carry_share"] = (agg["player_rz_carry"] / agg["team_rz_carry"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    agg["yprr_proxy"] = (agg["rec_yards"] / np.where(agg["player_targets"]>0, agg["player_targets"], np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["ypc"]        = (agg["rush_yards"] / np.where(agg["player_rush_att"]>0, agg["player_rush_att"], np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    agg["position"] = np.where(agg["player_rush_att"] > agg["player_targets"], "RB", "WR")
    agg["route_rate"] = np.nan
    agg["role"] = ""

    team_ypa = team_ypa.reset_index().rename(columns={"posteam":"team","qb_ypa_all":"qb_ypa"})
    out = agg.merge(team_ypa, on="team", how="left")
    out["qb_ypa"] = out["qb_ypa"].fillna(out["qb_ypa"].mean() if np.isfinite(out["qb_ypa"].mean()) else 6.9)

    out = out.rename(columns={"player":"player"})[[
        "player","team","position","role",
        "target_share","rush_share","route_rate",
        "rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","qb_ypa"
    ]].sort_values(["team","player"]).reset_index(drop=True)

    # optional PFR/mirrors enrich for real route/adot if you provide it
    try:
        enrich_path = Path("data/pfr_player_enrich.csv")
        if enrich_path.exists():
            enrich = pd.read_csv(enrich_path)
            rename_map = {
                "team_abbr": "team",
                "routes_db": "routes_per_dropback",
                "aDOT": "aDOT",
            }
            enrich = enrich.rename(columns={k:v for k,v in rename_map.items() if k in enrich.columns})
            keep = ["player","team","routes_per_dropback","aDOT","slot_rate","snap_share"]
            enrich = enrich[[c for c in keep if c in enrich.columns]].copy()
            for col in ["routes_per_dropback","slot_rate","snap_share"]:
                if col in enrich.columns:
                    enrich[col] = pd.to_numeric(enrich[col], errors="coerce")
            out = out.merge(enrich, on=["player","team"], how="left", suffixes=("","_enrich"))
            if "routes_per_dropback" in out.columns:
                out["route_rate"] = out["route_rate"].fillna(out["routes_per_dropback"])
    except Exception as e:
        print(f"[player_form] enrich skipped: {type(e).__name__}: {e}", flush=True)

    return out

def cli(season: int) -> int:
    try:
        df = build_from_nflverse(season)
        if os.getenv("NFL_FORM_STRICT") == "1":
            if df is None or df.empty or df["team"].nunique() < 8 or len(df) < 50:
                raise RuntimeError("[player_form] looks empty/stub — check logs/nfl_pbp_error.txt")
    except Exception as e:
        print(f"[player_form] fatal error: {type(e).__name__}: {e}", flush=True)
        df = pd.DataFrame()
    _safe_write(df, OUT)
    print(f"[player_form] wrote rows={len(df)} → {OUT}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    a = ap.parse_args()
    sys.exit(cli(a.season))
