#!/usr/bin/env python3
"""Build Stage-1 authority-exact source rows for Market-Relative Selector V1."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scripts.backtest.build_authority_exact_vegas_projection_trace_v1 import validate_qb, validate_te

KEYS=["season","week","team","player_clean_key"]

def read(path: Path) -> pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x

def schedule(files:list[Path]) -> pd.DataFrame:
    x=pd.concat([read(p) for p in files],ignore_index=True)
    need={"season","week","team","opponent","game_id"}
    miss=sorted(need-set(x.columns))
    if miss: raise RuntimeError(f"schedule missing columns: {miss}")
    x=x[["season","week","team","opponent","game_id"]].drop_duplicates()
    if x.duplicated(["season","week","team"]).any():
        raise RuntimeError("duplicate schedule team-week rows")
    return x

def attach(x:pd.DataFrame,s:pd.DataFrame,label:str)->pd.DataFrame:
    y=x.copy()
    if "opponent" in y.columns:
        y=y.rename(columns={"opponent":"authority_opponent"})
    y=y.merge(s,on=["season","week","team"],how="left",validate="many_to_one")
    if y["game_id"].isna().any():
        raise RuntimeError(f"{label} schedule attachment failed")
    if "authority_opponent" in y.columns:
        bad=y["authority_opponent"].astype(str).str.upper().ne(y["opponent"].astype(str).str.upper())
        if bad.any(): raise RuntimeError(f"{label} opponent mismatch")
        y=y.drop(columns=["authority_opponent"])
    return y

def out_rows(x,position,market,proj_col,actual_col,authority,scope):
    o=x[["season","week","team","opponent","game_id","player_clean_key"]].copy()
    o["player"]=x["player"].astype(str) if "player" in x.columns else x["player_clean_key"].astype(str)
    o["position"]=position
    o["market"]=market
    o["model_projection"]=pd.to_numeric(x[proj_col],errors="raise")
    o["actual"]=pd.to_numeric(x[actual_col],errors="raise")
    o["model_authority"]=authority
    o["authority_scope"]=scope
    return o

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--qb-casebook",type=Path,required=True)
    ap.add_argument("--te-casebook",type=Path,required=True)
    ap.add_argument("--schedule-file",type=Path,action="append",required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    qb=read(a.qb_casebook); te=read(a.te_casebook)
    q_audit=validate_qb(qb); t_audit=validate_te(te)
    s=schedule(a.schedule_file)

    q=attach(qb.loc[qb.season.isin([2024,2025])].copy(),s,"QB")
    t=attach(te.loc[te.season.isin([2024,2025])].copy(),s,"TE")
    rows=[
      out_rows(q,"QB","pass_yards","football_synthesis","actual_pass_yards","QB_PASS_SYNTHESIS_V1","exact M89 OOS 2024-2025"),
      out_rows(t,"TE","rec_yards","candidate_rec_yards_r5p","rec_yards","TE_R5P_PRODUCTION_MODEL_V1","exact TE-R5P OOS 2024-2025"),
      out_rows(t,"TE","receptions","candidate_receptions_r5p","receptions","TE_R5P_PRODUCTION_MODEL_V1","exact TE-R5P OOS 2024-2025"),
    ]
    out=pd.concat(rows,ignore_index=True)
    key=["season","week","team","player_clean_key","market"]
    if out.duplicated(key).any(): raise RuntimeError("duplicate Stage-1 source identities")
    if sorted(out.season.unique().tolist()) != [2024,2025]: raise RuntimeError("unexpected season scope")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.out,index=False)
    audit=pd.DataFrame([
      {**q_audit,"stage1_eligible":True},
      {**t_audit,"stage1_eligible":True},
      {"position":"WR","authority":"WR_R15_PRODUCTION_MODEL_V1","scope":"2025 confirmation forbidden","status":"SOURCE_REPLAY_BLOCKED","stage1_eligible":False},
      {"position":"RB","authority":"RB_P3_R26_R22_WEEK1_2026","scope":"no 2024-2025 retrospective promoted authority","status":"SOURCE_REPLAY_BLOCKED","stage1_eligible":False},
    ])
    audit.to_csv(a.out.with_name("market_relative_stage1_source_audit.csv"),index=False)
    print(out.groupby(["season","position","market"]).size().rename("rows").reset_index().to_string(index=False))

if __name__=="__main__":
    main()
