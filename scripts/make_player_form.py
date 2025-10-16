# scripts/make_player_form.py
"""
Create player_form.csv: per-player usage & efficiency metrics.

Inputs:
- Play-by-play via nflreadpy
- Optional enrichers (ESPN, MySportsFeeds, API-Sports, NFLGSIS)
Outputs:
- data/player_form.csv
"""

from __future__ import annotations
import argparse, os, sys, warnings
import pandas as pd, numpy as np

def _import_nflverse():
    try:
        import nflreadpy as nflv
        return nflv, "nflreadpy"
    except Exception:
        import nfl_data_py as nflv  # fallback
        return nflv, "nfl_data_py"

NFLV, NFLPKG = _import_nflverse()
DATA_DIR = "data"
OUTPATH = os.path.join(DATA_DIR, "player_form.csv")

def _safe_mkdir(p): os.makedirs(p, exist_ok=True) if not os.path.exists(p) else None
def safe_div(n,d): 
    n,d = n.astype(float), d.astype(float)
    with np.errstate(divide="ignore",invalid="ignore"): return np.where(d==0,np.nan,n/d)

def load_pbp(season:int)->pd.DataFrame:
    if NFLPKG=="nflreadpy": df=NFLV.load_pbp(seasons=[season])
    else: df=NFLV.import_pbp_data([season], downcast=True)
    df.columns=[c.lower() for c in df.columns]; return df

def load_participation(season:int)->pd.DataFrame:
    try:
        if NFLPKG=="nflreadpy": p=NFLV.load_participation(seasons=[season])
        else: return pd.DataFrame()
        p.columns=[c.lower() for c in p.columns]; return p
    except Exception: return pd.DataFrame()

def compute_player_usage(pbp:pd.DataFrame)->pd.DataFrame:
    """Target share, rush share, red-zone share, efficiency."""
    df=pbp.copy(); df.columns=[c.lower() for c in df.columns]
    if "posteam" not in df or "player_name" not in df: return pd.DataFrame()

    # identify plays
    is_pass=df.get("pass",pd.Series(False,index=df.index)).astype(bool)
    is_rush=df.get("rush",pd.Series(False,index=df.index)).astype(bool)
    off=df["posteam"]

    # receiving targets
    rec=df.loc[is_pass & df["receiver_player_name"].notna(),["posteam","receiver_player_name","yardline_100","yards_gained"]]
    rec["is_rz"]=rec["yardline_100"]<=20
    tgt=rec.groupby(["posteam","receiver_player_name"]).size().rename("targets")
    rz_tgt=rec.loc[rec["is_rz"]].groupby(["posteam","receiver_player_name"]).size().rename("rz_targets")
    ypt=rec.groupby(["posteam","receiver_player_name"])["yards_gained"].mean().rename("ypt")

    # rushing attempts
    rush=df.loc[is_rush & df["rusher_player_name"].notna(),["posteam","rusher_player_name","yardline_100","yards_gained"]]
    rush["is_rz"]=rush["yardline_100"]<=20
    att=rush.groupby(["posteam","rusher_player_name"]).size().rename("rush_att")
    rz_carry=rush.loc[rush["is_rz"]].groupby(["posteam","rusher_player_name"]).size().rename("rz_carries")
    ypc=rush.groupby(["posteam","rusher_player_name"])["yards_gained"].mean().rename("ypc")

    # combine per-team totals for shares
    team_tgts=tgt.groupby(level=0).sum()
    team_atts=att.groupby(level=0).sum()

    rec_tbl=pd.concat([tgt,rz_tgt,ypt],axis=1).reset_index().rename(columns={"receiver_player_name":"player"})
    rush_tbl=pd.concat([att,rz_carry,ypc],axis=1).reset_index().rename(columns={"rusher_player_name":"player"})
    rec_tbl["rush_att"]=0; rush_tbl["targets"]=0
    base=pd.concat([rec_tbl,rush_tbl],axis=0,ignore_index=True)
    base["team"]=base["posteam"]
    base=base.groupby(["team","player"],dropna=False).sum().reset_index()

    base["target_share"]=safe_div(base["targets"],base.groupby("team")["targets"].transform("sum"))
    base["rush_share"]=safe_div(base["rush_att"],base.groupby("team")["rush_att"].transform("sum"))
    base["rz_tgt_share"]=safe_div(base["rz_targets"],base.groupby("team")["rz_targets"].transform("sum"))
    base["rz_carry_share"]=safe_div(base["rz_carries"],base.groupby("team")["rz_carries"].transform("sum"))
    base["yprr_proxy"]=np.where(base["targets"]>0,base["ypt"],np.nan)
    base["route_rate"]=np.nan  # to be enriched later
    return base

def enrich_with_participation(base:pd.DataFrame,part:pd.DataFrame)->pd.DataFrame:
    """Add route_rate from participation if available."""
    if part.empty or "offense_team" not in part: return base
    p=part.copy(); p.columns=[c.lower() for c in p.columns]
    if not {"offense_team","player"}.issubset(p.columns): return base
    # crude route_rate proxy: routes/plays per player
    if "routes_run" in p.columns and "plays" in p.columns:
        rr=p.groupby(["offense_team","player"]).apply(lambda x:safe_div(x["routes_run"].sum(),x["plays"].sum()).mean())
        rr=rr.rename("route_rate").reset_index().rename(columns={"offense_team":"team"})
        return base.merge(rr,on=["team","player"],how="left")
    return base

def fallback_from_external(base:pd.DataFrame)->pd.DataFrame:
    """Merge optional enrichers without overwriting existing values."""
    enrichers=["espn_player_form.csv","msf_player_form.csv","apisports_player_form.csv","nflgsis_player_form.csv"]
    for f in enrichers:
        path=os.path.join(DATA_DIR,f)
        if not os.path.exists(path): continue
        try:
            e=pd.read_csv(path)
        except Exception: continue
        e.columns=[c.lower() for c in e.columns]
        keep=[c for c in ["team","player","target_share","rush_share","route_rate","ypt","ypc","yprr_proxy"] if c in e.columns]
        if not keep: continue
        e=e[keep]
        base=base.merge(e,on=["team","player"],how="left",suffixes=("","_ext"))
        for c in [k for k in keep if k not in ["team","player"]]:
            base[c]=base[c].combine_first(base.get(f"{c}_ext"))
            if f"{c}_ext" in base: base.drop(columns=f"{c}_ext",inplace=True)
    return base

def build_player_form(season:int)->pd.DataFrame:
    print(f"[make_player_form] Loading PBP for {season} ({NFLPKG}) ...")
    pbp=load_pbp(season)
    if pbp.empty: raise RuntimeError("Empty PBP.")
    base=compute_player_usage(pbp)
    part=load_participation(season)
    base=enrich_with_participation(base,part)
    base=fallback_from_external(base)
    base["season"]=season
    order=["player","team","season","target_share","rush_share","rz_tgt_share","rz_carry_share","ypt","ypc","yprr_proxy","route_rate"]
    for c in order:
        if c not in base.columns: base[c]=np.nan
    base=base[order].sort_values(["team","player"]).reset_index(drop=True)
    return base

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--season",type=int,default=2025)
    args=parser.parse_args()
    _safe_mkdir(DATA_DIR)
    try:
        df=build_player_form(args.season)
    except Exception as e:
        print(f"[make_player_form] ERROR: {e}",file=sys.stderr)
        df=pd.DataFrame(columns=["player","team","season","target_share","rush_share","rz_tgt_share","rz_carry_share","ypt","ypc","yprr_proxy","route_rate"])
    df.to_csv(OUTPATH,index=False)
    print(f"[make_player_form] Wrote {len(df)} rows → {OUTPATH}")

if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
