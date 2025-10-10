from __future__ import annotations
import argparse, pandas as pd, numpy as np
from pathlib import Path

try:
    from scripts.utils.names import normalize_team
except Exception:
    def normalize_team(t: str) -> str:
        if not isinstance(t, str): return ""
        t = t.strip().upper()
        return {"JAX":"JAC","WSH":"WAS","LA":"LAR","ARZ":"ARI","CLV":"CLE"}.get(t,t)

def _z(x): return (x - x.mean())/x.std(ddof=0) if x.std(ddof=0)>0 else x*0

def _nflverse_team(season:int) -> pd.DataFrame:
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data([season])
    grp = pbp.groupby("defteam").agg(
        def_pass_epa = ("epa", lambda s: s[pbp.loc[s.index,"pass"]==1].mean()),
        def_rush_epa = ("epa", lambda s: s[pbp.loc[s.index,"rush"]==1].mean()),
        def_sack_rate = ("sack", "mean"),
    ).reset_index().rename(columns={"defteam":"team"})
    nu = pbp[(pbp.down.isin([1,2])) & (pbp.score_differential.abs()<=7)]
    pace = nu.groupby("posteam")["game_seconds_remaining"].apply(lambda s: s.diff().abs().dropna().mean()).reset_index()
    pace.columns=["team","pace"]
    proe = nu.groupby("posteam")["pass"].mean().reset_index().rename(columns={"posteam":"team","pass":"proe"})
    df = grp.merge(pace, on="team", how="left").merge(proe, on="team", how="left")
    return df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--season", type=int, required=True); a=ap.parse_args()
    Path("data").mkdir(exist_ok=True); Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    try:
        df = _nflverse_team(a.season)
        print(f"[team_form] nflverse rows={len(df)}")
    except Exception as e:   # <- no bare 'Error'
        print(f"[team_form] nflverse error: {e}")
        df = pd.DataFrame(columns=["team"])
    if not df.empty:
        df["team"]=df["team"].map(normalize_team)
        for c in ["def_pass_epa","def_rush_epa","def_sack_rate","pace","proe"]:
            if c in df.columns:
                df[c+"_z"] = _z(df[c].fillna(df[c].median()))
    df.to_csv("data/team_form.csv", index=False)
    df.to_csv("outputs/metrics/team_form.csv", index=False)
    print(f"[team_form] wrote rows={len(df)} → data/team_form.csv")

if __name__=="__main__":
    main()
