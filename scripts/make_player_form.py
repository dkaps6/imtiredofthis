from __future__ import annotations
import argparse, pandas as pd, numpy as np
from pathlib import Path

try:
    from scripts.utils.names import normalize_team, normalize_player
except Exception:
    def normalize_team(t: str) -> str:
        if not isinstance(t, str): return ""
        t = t.strip().upper()
        return {"JAX":"JAC","WSH":"WAS","LA":"LAR","ARZ":"ARI","CLV":"CLE"}.get(t,t)
    def normalize_player(s: str) -> str:
        return " ".join(str(s or "").split())

def _nflverse_player(season:int) -> pd.DataFrame:
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data([season])
    pbp = pbp.assign(team=pbp.posteam)

    rec = pbp[pbp["pass"]==1].copy()
    rec["player"] = rec["receiver_player_name"].fillna("")
    tgt = rec[rec["player"]!=""].groupby(["team","player"]).size().rename("targets").reset_index()
    tgt_team = rec.groupby("team").size().rename("team_targets").reset_index()

    ru = pbp[pbp["rush"]==1].copy()
    ru["player"] = ru["rusher_player_name"].fillna("")
    car = ru[ru["player"]!=""].groupby(["team","player"]).size().rename("carries").reset_index()
    car_team = ru.groupby("team").size().rename("team_carries").reset_index()

    rz = pbp[(pbp["yardline_100"]<=20)].copy()
    rz_rush = rz[rz["rush"]==1].copy(); rz_rush["player"]=rz_rush["rusher_player_name"].fillna("")
    rz_car = rz_rush[rz_rush["player"]!=""].groupby(["team","player"]).size().rename("rz_carries").reset_index()
    rz_pass = rz[rz["pass"]==1].copy(); rz_pass["player"]=rz_pass["receiver_player_name"].fillna("")
    rz_tgt = rz_pass[rz_pass["player"]!=""].groupby(["team","player"]).size().rename("rz_tgts").reset_index()
    rz_car_team = rz_rush.groupby("team").size().rename("team_rz_carries").reset_index()
    rz_tgt_team = rz_pass.groupby("team").size().rename("team_rz_tgts").reset_index()

    df = pd.merge(tgt, tgt_team, on="team", how="outer")
    df = pd.merge(df, car, on=["team","player"], how="outer")
    df = pd.merge(df, car_team, on="team", how="outer")
    df = pd.merge(df, rz_car, on=["team","player"], how="left")
    df = pd.merge(df, rz_tgt, on=["team","player"], how="left")
    df = pd.merge(df, rz_car_team, on="team", how="left")
    df = pd.merge(df, rz_tgt_team, on="team", how="left")

    df["target_share"]    = (df["targets"]    / df["team_targets"]).fillna(0.0).clip(0,1)
    df["rush_share"]      = (df["carries"]    / df["team_carries"]).fillna(0.0).clip(0,1)
    df["rz_carry_share"]  = (df["rz_carries"] / df["team_rz_carries"]).fillna(0.0).clip(0,1)
    df["rz_tgt_share"]    = (df["rz_tgts"]    / df["team_rz_tgts"]).fillna(0.0).clip(0,1)

    qb = pbp.groupby("posteam")["yards_gained"].mean().rename("qb_ypa").reset_index().rename(columns={"posteam":"team"})
    ypc = ru.groupby("rusher_player_name")["yards_gained"].mean().rename("ypc").reset_index().rename(columns={"rusher_player_name":"player"})
    ypt = rec.groupby("receiver_player_name")["yards_gained"].mean().rename("ypt").reset_index().rename(columns={"receiver_player_name":"player"})
    df = df.merge(qb, on="team", how="left").merge(ypt, on="player", how="left").merge(ypc, on="player", how="left")
    return df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--season", type=int, required=True); a=ap.parse_args()
    Path("data").mkdir(exist_ok=True); Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    try:
        df = _nflverse_player(a.season)
        print(f"[player_form] nflverse rows={len(df)}")
    except Exception as e:     # <- no bare 'Error'
        print(f"[player_form] nflverse error: {e}")
        df = pd.DataFrame(columns=["team","player"])
    if not df.empty:
        df["team"]=df["team"].map(normalize_team)
        df["player"]=df["player"].map(normalize_player)
    df.to_csv("data/player_form.csv", index=False)
    df.to_csv("outputs/metrics/player_form.csv", index=False)
    print(f"[player_form] wrote rows={len(df)} → data/player_form.csv")

if __name__=="__main__":
    main()
