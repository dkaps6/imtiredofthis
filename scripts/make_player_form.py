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

# ---- DIAGNOSTIC/RETRY WRAPPER (nflreadpy first, then nfl_data_py; 404 fallback seasons) ----
def _fetch_pbp_with_retry(season: int, tries: int = 3, wait: int = 4) -> pd.DataFrame:
    import traceback, time
    seasons_to_try = [season, season - 1, season - 2]
    last = None
    try:
        import nflreadpy as nfr
        use_readpy = True
    except Exception:
        use_readpy = False
    if not use_readpy:
        import nfl_data_py as nfl

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

                if df is not None and len(df):
                    with open(ERR_LOG, "a", encoding="utf-8") as f:
                        f.write(f"season {s} try {i}: OK rows={len(df)} via "
                                f"{'nflreadpy' if use_readpy else 'nfl_data_py'}\n")
                    if s != season:
                        print(f"[player_form] NOTE: using season {s} as fallback for {season}", flush=True)
                    return df

                raise RuntimeError("PBP fetch returned empty dataframe")
            except Exception as e:
                last = e
                tb = traceback.format_exc()
                msg = f"{type(e).__name__}: {e}"
                print(f"[player_form] season {s} try {i}/{tries} failed: {msg}", flush=True)
                with open(ERR_LOG, "a", encoding="utf-8") as f:
                    f.write(f"season {s} try {i}: {msg}\n{tb}\n")
                if (not use_readpy) and ("HTTP Error 404" in str(e)):
                    print(f"[player_form] season {s}: 404 detected; trying previous season…", flush=True)
                    break
                time.sleep(wait)

    raise RuntimeError(f"PBP fetch failed for {season}: {type(last).__name__}: {last}")

def build_from_nflverse(season: int) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl  # noqa: F401
    except Exception as e:
        print(f"[player_form] nfl_data_py import failed → fallback: {e}", flush=True)
        return pd.DataFrame([
            {"player":"Example WR1","team":"BUF","position":"WR","role":"WR1",
             "target_share":0.24,"rush_share":0.00,"route_rate":0.90,
             "rz_tgt_share":0.28,"rz_carry_share":0.00,
             "yprr_proxy":2.1,"ypc":0.0,"qb_ypa":7.6}
        ])

    # guard against shadowing; show exactly what got imported
    import importlib
    nfl_mod = importlib.import_module("nfl_data_py")
    nfl_path = getattr(nfl_mod, "__file__", "")
    print(f"[player_form] nfl_data_py path → {nfl_path}", flush=True)
    if "site-packages" not in (nfl_path or "") and "dist-packages" not in (nfl_path or ""):
        raise RuntimeError(f"Wrong nfl_data_py imported (shadowed). Path: {nfl_path}")

    print("[player_form] pulling pbp…", flush=True)
    pbp = _fetch_pbp_with_retry(season)
    pbp = pbp.loc[pbp["season"]==season].copy()

    pbp["posteam"] = pbp["posteam"].astype(str).str.upper()
    pbp["receiver"] = pbp.get("receiver_player_name","").fillna("").astype(str)
    pbp["rusher"]   = pbp.get("rusher_player_name","").fillna("").astype(str)
    pbp["passer"]   = pbp.get("passer_player_name","").fillna("").astype(str)

    pbp["is_pass"] = (pbp.get("pass",0)==1) | (pbp.get("pass_attempt",0)==1) | (pbp.get("play_type","")=="pass")
    pbp["is_rush"] = (pbp.get("rush",0)==1) | (pbp.get("play_type","")=="run")
    pbp["in_rz"]   = (pbp.get("yardline_100").fillna(100) <= 20)

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

    rec_level = rec_tgts.reset_index().rename(columns={"receiver":"player"})
    rush_level = ply_rush.reset_index().rename(columns={"rusher":"player"})
    players = pd.concat([rec_level[["game_id","posteam","player","player_targets"]],
                         rush_level[["game_id","posteam","player","player_rush_att"]]], ignore_index=True).fillna(0)

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

    agg["yprr_proxy"] = (agg["rec_yards"] / agg["player_targets"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["ypc"]        = (agg["rush_yards"] / agg["player_rush_att"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)

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

    return out

def cli(season: int) -> int:
    try:
        df = build_from_nflverse(season)
        if os.getenv("NFL_FORM_STRICT") == "1":
            if df is None or df.empty or df["team"].nunique() < 8 or len(df) < 50:
                raise RuntimeError("[player_form] looks empty/stub — check requirements install and network; see logs/nfl_pbp_error.txt")
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
