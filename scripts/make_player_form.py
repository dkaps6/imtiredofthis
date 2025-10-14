#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/player_form.csv")

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

def build_from_nflverse(season: int) -> pd.DataFrame:
    """
    Pull real season PBP and compute simple, robust player volume/efficiency:
      - target_share = targets / team targets
      - rush_share   = rush attempts / team rush att
      - rz_tgt_share / rz_carry_share based on yardline_100 <= 20
      - yprr_proxy   = receiving yards / targets  (proxy)
      - ypc          = rush yards / rush att
      - qb_ypa       = team QB passing yards / attempts (mapped to all players on that team)
    """
    try:
        import nfl_data_py as nfl
    except Exception as e:
        print(f"[player_form] nfl_data_py not installed; falling back to stub: {e}", flush=True)
        return pd.DataFrame([
            {"player":"Example WR1","team":"BUF","position":"WR","role":"WR1",
             "target_share":0.24,"rush_share":0.00,"route_rate":0.90,
             "rz_tgt_share":0.28,"rz_carry_share":0.00,
             "yprr_proxy":2.1,"ypc":0.0,"qb_ypa":7.6}
        ])

    pbp = nfl.import_pbp_data([season]).copy()
    pbp = pbp.loc[pbp["season"]==season].copy()

    # Normalize names
    pbp["posteam"] = pbp["posteam"].astype(str).str.upper()
    pbp["defteam"] = pbp["defteam"].astype(str).str.upper()
    pbp["receiver"] = pbp.get("receiver_player_name","").fillna("").astype(str)
    pbp["rusher"]   = pbp.get("rusher_player_name","").fillna("").astype(str)
    pbp["passer"]   = pbp.get("passer_player_name","").fillna("").astype(str)

    pbp["is_pass"] = (pbp.get("pass", 0).fillna(0)==1) | (pbp.get("pass_attempt",0).fillna(0)==1) | (pbp.get("play_type","")=="pass")
    pbp["is_rush"] = (pbp.get("rush", 0).fillna(0)==1) | (pbp.get("play_type","")=="run")

    # Targets: credited when receiver name is present on a pass
    targs = pbp.loc[pbp["is_pass"] & (pbp["receiver"]!=""), ["game_id","posteam","receiver"]].copy()
    targs["targets"] = 1
    team_tgts = targs.groupby(["game_id","posteam"])["targets"].sum().rename("team_targets")
    rec_tgts  = targs.groupby(["game_id","posteam","receiver"])["targets"].sum().rename("player_targets")

    # Rush attempts
    rush = pbp.loc[pbp["is_rush"] & (pbp["rusher"]!=""), ["game_id","posteam","rusher","yards_gained"]].copy()
    rush["rush_att"] = 1
    team_rush = rush.groupby(["game_id","posteam"])["rush_att"].sum().rename("team_rush_att")
    ply_rush  = rush.groupby(["game_id","posteam","rusher"])["rush_att"].sum().rename("player_rush_att")
    ply_rush_yds = rush.groupby(["game_id","posteam","rusher"])["yards_gained"].sum().rename("rush_yards")

    # Receiving yards per player (for yprr proxy)
    rec_yards = pbp.loc[pbp["receiver"]!=""].groupby(["game_id","posteam","receiver"])["yards_gained"].sum().rename("rec_yards")

    # Red-zone flags
    pbp["in_rz"] = (pbp.get("yardline_100").fillna(100) <= 20)
    rz_targs = pbp.loc[pbp["is_pass"] & pbp["in_rz"] & (pbp["receiver"]!=""), ["game_id","posteam","receiver"]].copy()
    rz_targs["rz_tgt"] = 1
    ply_rz_tgt = rz_targs.groupby(["game_id","posteam","receiver"])["rz_tgt"].sum().rename("player_rz_tgt")
    team_rz_tgt = rz_targs.groupby(["game_id","posteam"])["rz_tgt"].sum().rename("team_rz_tgt")

    rz_rush = pbp.loc[pbp["is_rush"] & pbp["in_rz"] & (pbp["rusher"]!=""), ["game_id","posteam","rusher"]].copy()
    rz_rush["rz_carry"] = 1
    ply_rz_carry = rz_rush.groupby(["game_id","posteam","rusher"])["rz_carry"].sum().rename("player_rz_carry")
    team_rz_carry = rz_rush.groupby(["game_id","posteam"])["rz_carry"].sum().rename("team_rz_carry")

    # QB YPA per team
    pass_att = pbp.loc[pbp["is_pass"] & (pbp["passer"]!=""), ["game_id","posteam","passer","yards_gained"]].copy()
    pass_att["att"] = 1
    team_pass_yds = pass_att.groupby(["game_id","posteam"])["yards_gained"].sum().rename("team_pass_yds")
    team_atts = pass_att.groupby(["game_id","posteam"])["att"].sum().rename("team_pass_att")
    team_ypa = (team_pass_yds.groupby("posteam").sum() / team_atts.groupby("posteam").sum()).rename("qb_ypa_all")

    # Build player rows by union of receivers & rushers (by game then aggregate season)
    rec_level = rec_tgts.reset_index().rename(columns={"receiver":"player"})
    rush_level = ply_rush.reset_index().rename(columns={"rusher":"player"})
    players = pd.concat([rec_level[["game_id","posteam","player","player_targets"]],
                         rush_level[["game_id","posteam","player","player_rush_att"]]], ignore_index=True).fillna(0)

    # Merge all per-game stats
    players = players.merge(team_tgts.reset_index(), on=["game_id","posteam"], how="left")
    players = players.merge(team_rush.reset_index(), on=["game_id","posteam"], how="left")
    players = players.merge(ply_rz_tgt.reset_index().rename(columns={"receiver":"player"}), on=["game_id","posteam","player"], how="left")
    players = players.merge(team_rz_tgt.reset_index(), on=["game_id","posteam"], how="left")
    players = players.merge(ply_rz_carry.reset_index().rename(columns={"rusher":"player"}), on=["game_id","posteam","player"], how="left")
    players = players.merge(team_rz_carry.reset_index(), on=["game_id","posteam"], how="left")
    players = players.merge(rec_yards.reset_index().rename(columns={"receiver":"player"}), on=["game_id","posteam","player"], how="left")
    players = players.merge(ply_rush_yds.reset_index().rename(columns={"rusher":"player"}), on=["game_id","posteam","player"], how="left")

    # fill NaNs
    for c in ["player_targets","team_targets","player_rush_att","team_rush_att",
              "player_rz_tgt","team_rz_tgt","player_rz_carry","team_rz_carry",
              "rec_yards","rush_yards"]:
        if c not in players.columns: players[c]=0
        players[c] = players[c].fillna(0)

    # Aggregate to season
    agg = players.groupby(["posteam","player"]).sum(numeric_only=True).reset_index()
    agg["team"] = agg["posteam"].astype(str).str.upper()

    # Shares
    agg["target_share"]   = (agg["player_targets"] / agg["team_targets"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["rush_share"]     = (agg["player_rush_att"] / agg["team_rush_att"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["rz_tgt_share"]   = (agg["player_rz_tgt"] / agg["team_rz_tgt"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["rz_carry_share"] = (agg["player_rz_carry"] / agg["team_rz_carry"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # Efficiencies
    agg["yprr_proxy"] = (agg["rec_yards"] / agg["player_targets"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["ypc"]        = (agg["rush_yards"] / agg["player_rush_att"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # Position guess (very light; your roles.csv can override elsewhere)
    # If mostly rushing → RB; if mostly targeted → WR/TE heuristic
    agg["position"] = np.where(agg["player_rush_att"] > agg["player_targets"], "RB", "WR")
    # route_rate not in PBP → leave NaN; your downstream fills default (0.75) if missing
    agg["route_rate"] = np.nan

    # Role unknown here
    agg["role"] = ""

    # Map team QB YPA to each player’s team
    team_ypa = team_ypa.reset_index().rename(columns={"posteam":"team","qb_ypa_all":"qb_ypa"})
    out = agg.merge(team_ypa, on="team", how="left")
    out["qb_ypa"] = out["qb_ypa"].fillna(out["qb_ypa"].mean() if np.isfinite(out["qb_ypa"].mean()) else 6.9)

    # Final columns in your schema order
    out = out.rename(columns={"player":"player"})
    out = out[[
        "player","team","position","role",
        "target_share","rush_share","route_rate",
        "rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","qb_ypa"
    ]].sort_values(["team","player"]).reset_index(drop=True)

    return out

def cli(season: int) -> int:
    try:
        df = build_from_nflverse(season)
    except Exception as e:
        print(f"[player_form] fatal error: {e}", flush=True)
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
