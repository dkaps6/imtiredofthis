#!/usr/bin/env python3
"""RB market benchmark v2: frozen M94C vs exact archived DK/FD rushing_yards lines.

Mechanical repair only from run 33498879907:
- exact `rushing_yards` market only (no milestones/combo markets)
- source names are abbreviated (e.g. d.henry), so join by team + week + first-initial/surname key
No football/model logic is changed.
"""
from __future__ import annotations

import argparse
import io
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

SOURCE_URL = (
    "https://raw.githubusercontent.com/gcampb41/nfl_data-/main/"
    "data/processed/football/nfl/player_props/2025.parquet"
)
BOOKS = {68: "draftkings", 69: "fanduel"}
FULL_GAME_PERIODS = {"0", "0.0", "game", "full", "fullgame", "full_game", "full game", "event", "match", "all"}
TEAM_MAP = {"OAK":"LV", "SD":"LAC", "STL":"LA", "LAR":"LA", "JAX":"JAC", "ARZ":"ARI", "WSH":"WAS"}


def num(s): return pd.to_numeric(s, errors="coerce")
def text(s): return s.astype("string").fillna("").str.strip().str.lower()

def first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lookup={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in lookup: return str(lookup[n.lower()])
    return None

def team(v):
    if pd.isna(v): return ""
    s=str(v).strip().upper()
    if not s or s in {"NAN","NONE","<NA>"}: return ""
    return TEAM_MAP.get(s,s)

def short_name(v):
    raw=str(v or "").lower().strip().replace("'","")
    raw=re.sub(r"[^a-z0-9.\- ]","",raw)
    toks=[t for t in re.split(r"[\s.\-]+",raw) if t and t not in {"jr","sr","ii","iii","iv"}]
    if not toks: return ""
    if len(toks)==1: return re.sub(r"[^a-z0-9]","",toks[0])
    return re.sub(r"[^a-z0-9]","",toks[0][0]+toks[-1])

def metrics(actual,pred):
    q=pd.DataFrame({"actual":num(actual),"pred":num(pred)}).dropna()
    if q.empty: return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":np.nan,"actual_mean":np.nan,"pred_mean":np.nan}
    e=q.pred-q.actual
    corr=q.actual.corr(q.pred) if len(q)>=3 and q.actual.nunique()>1 and q.pred.nunique()>1 else np.nan
    return {"n":len(q),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.square(e).mean())),"bias":float(e.mean()),
            "corr":float(corr) if pd.notna(corr) else np.nan,"actual_mean":float(q.actual.mean()),"pred_mean":float(q.pred.mean())}


def load_market():
    r=requests.get(SOURCE_URL,timeout=120,headers={"User-Agent":"imtiredofthis-rb-market-benchmark/2.0"}); r.raise_for_status()
    raw=pd.read_parquet(io.BytesIO(r.content)); raw.columns=[str(c).strip().lower() for c in raw.columns]
    bet=first_existing(raw,["bet_type","market","market_key"])
    if not bet: raise RuntimeError("archive missing bet type")
    bet_audit=text(raw[bet]).value_counts().rename_axis("bet_type").rename("rows").reset_index()
    x=raw.copy()
    if "season" in x: x=x.loc[num(x.season).eq(2025)].copy()
    x["week"]=num(x.week); x=x.loc[x.week.between(1,18,inclusive="both")].copy()
    # exact straight player rushing-yard prop only
    x=x.loc[text(x[bet]).eq("rushing_yards")].copy()
    exact_market_rows=len(x)
    if x.empty: raise RuntimeError("no exact rushing_yards rows")
    if "period" in x:
        p=text(x.period).str.replace(r"\s+"," ",regex=True); full=p.isin(FULL_GAME_PERIODS)
        if not full.any(): raise RuntimeError(f"no recognized full-game rushing_yards period: {sorted(set(p))[:20]}")
        x=x.loc[full].copy()
    full_game_rows=len(x)
    x["book_id"]=num(x.book_id); x=x.loc[x.book_id.isin(BOOKS)].copy(); x["book"]=x.book_id.map(BOOKS)
    dkfd_rows=len(x)
    line_col=first_existing(x,["value","line","point"]); side_col=first_existing(x,["side","label","outcome"])
    name_col=first_existing(x,["join_name","player_name","player","name","full_name"]); team_col=first_existing(x,["team","team_abbr","team_abbreviation"])
    if not all([line_col,side_col,name_col]): raise RuntimeError(f"missing line/side/name: {line_col}/{side_col}/{name_col}")
    x["line"]=num(x[line_col]); s=text(x[side_col]); x["side_norm"]=np.select([s.isin(["over","o"]),s.isin(["under","u"])],["OVER","UNDER"],default="")
    x=x.loc[x.line.notna()&x.line.gt(0)&x.side_norm.ne("")].copy()
    x["source_player"]=x[name_col].astype("string").fillna("").str.strip(); x["short_key"]=x.source_player.map(short_name)
    x["source_team"]=x[team_col].map(team) if team_col else ""; x["week"]=x.week.astype(int)
    x=x.loc[x.short_key.ne("")&x.source_team.ne("")].copy()
    rows=[]; conflicts=0; one_sided=0
    for (week,t,sk,book),g in x.groupby(["week","source_team","short_key","book"],dropna=False):
        lines=sorted(set(num(g.line).dropna().round(6)))
        if len(lines)!=1: conflicts+=1; continue
        if not {"OVER","UNDER"}.issubset(set(g.side_norm)): one_sided+=1; continue
        rows.append({"season":2025,"week":int(week),"team":str(t),"short_key":str(sk),"book":str(book),"line":float(lines[0]),
                     "source_player":str(g.source_player.iloc[-1]),"source_line_definition":"archived_latest_per_book"})
    out=pd.DataFrame(rows)
    audit=pd.DataFrame([{"source_url":SOURCE_URL,"download_bytes":len(r.content),"raw_rows":len(raw),"exact_rushing_yards_rows":exact_market_rows,
                         "full_game_rushing_yards_rows":full_game_rows,"dk_fd_full_game_rows":dkfd_rows,"eligible_book_player_rows":len(out),
                         "conflicting_player_week_team_book_groups_dropped":conflicts,"one_sided_groups_dropped":one_sided,
                         "line_definition":"archived_latest_per_book_closing_like_not_fixed_timestamp"}])
    return out,audit,bet_audit


def load_m94c(root:Path):
    hits=list(root.rglob("m94c_2025_rb_trace.csv"))
    if len(hits)!=1: raise RuntimeError(f"expected one M94C trace; found {len(hits)}")
    x=pd.read_csv(hits[0],low_memory=False); x.columns=[str(c).lower() for c in x.columns]
    x=x.loc[num(x.season).eq(2025)].copy()
    if "position" in x: x=x.loc[x.position.astype(str).str.upper().isin(["RB","FB"])].copy()
    x["week"]=num(x.week).astype(int); x["team"]=x.team.map(team); x["short_key"]=x.player.map(short_name)
    x["candidate_rush_yards"]=num(x.candidate_rush_yards); x["actual_rush_yards"]=num(x.actual_rush_yards); x["actual_rush_att"]=num(x.actual_rush_att)
    x=x.loc[x.short_key.ne("")&x.candidate_rush_yards.notna()&x.actual_rush_yards.notna()].copy()
    keys=["season","week","team","short_key"]
    amb=x.groupby(keys).size(); amb=amb[amb.gt(1)]
    if len(amb): raise RuntimeError(f"ambiguous M94C short keys: {amb.head(20).to_dict()}")
    return x[["season","week","team","short_key","player","opponent","candidate_rush_yards","actual_rush_yards","actual_rush_att"]]


def join(m94,market):
    j=market.merge(m94,on=["season","week","team","short_key"],how="left",validate="many_to_one")
    matched=j.player.notna(); unmatched=j.loc[~matched,["week","team","short_key","source_player","book","line"]].copy(); j=j.loc[matched].copy()
    idx=["season","week","team","short_key","player","opponent","candidate_rush_yards","actual_rush_yards","actual_rush_att"]
    p=j.pivot_table(index=idx,columns="book",values="line",aggfunc="first").reset_index()
    for c in ["draftkings","fanduel"]:
        if c not in p: p[c]=np.nan
    p["consensus_line"]=p[["draftkings","fanduel"]].median(axis=1,skipna=True)
    both=p.draftkings.notna()&p.fanduel.notna(); p["two_book_consensus"]=np.where(both,(p.draftkings+p.fanduel)/2,np.nan)
    p["market_books"]=p[["draftkings","fanduel"]].notna().sum(axis=1)
    p["model_minus_market"]=p.candidate_rush_yards-p.consensus_line; p["abs_disagreement"]=p.model_minus_market.abs()
    p["model_abs_error"]=(p.candidate_rush_yards-p.actual_rush_yards).abs(); p["market_abs_error"]=(p.consensus_line-p.actual_rush_yards).abs()
    p["winner"]=np.select([p.model_abs_error<p.market_abs_error,p.market_abs_error<p.model_abs_error],["MODEL","MARKET"],default="TIE")
    audit=pd.DataFrame([{"m94c_rows":len(m94),"eligible_market_book_rows":len(market),"matched_book_rows":int(matched.sum()),"unmatched_book_rows":int((~matched).sum()),
                         "final_consensus_player_games":int(p.consensus_line.notna().sum()),"two_book_player_games":int(p.two_book_consensus.notna().sum()),
                         "draftkings_player_games":int(p.draftkings.notna().sum()),"fanduel_player_games":int(p.fanduel.notna().sum()),
                         "unmatched_examples":unmatched.head(25).to_json(orient="records")}])
    return p,audit


def summarize(z):
    common=z.loc[z.consensus_line.notna()].copy()
    rows=[]
    for arm,col in [("M94C_MODEL","candidate_rush_yards"),("VEGAS_CONSENSUS","consensus_line"),("DRAFTKINGS","draftkings"),("FANDUEL","fanduel"),("TWO_BOOK_CONSENSUS","two_book_consensus")]:
        # For the headline M94C row, restrict to the same market-covered rows.
        base=common if arm=="M94C_MODEL" else z
        rows.append({"arm":arm,**metrics(base.actual_rush_yards,base[col])})
    summary=pd.DataFrame(rows)
    w=common.winner.value_counts().to_dict(); non_tie=w.get("MODEL",0)+w.get("MARKET",0)
    h2h=pd.DataFrame([{"n":len(common),"model_closer":w.get("MODEL",0),"market_closer":w.get("MARKET",0),"ties":w.get("TIE",0),
                       "model_closer_rate_ex_ties":w.get("MODEL",0)/max(non_tie,1),
                       "mean_market_abs_error_minus_model_abs_error":float((common.market_abs_error-common.model_abs_error).mean()),
                       "median_market_abs_error_minus_model_abs_error":float((common.market_abs_error-common.model_abs_error).median())}])
    common["disagreement_bucket"]=pd.cut(common.abs_disagreement,[-np.inf,5,10,15,np.inf],labels=["lt5","5_to_lt10","10_to_lt15","15_plus"],right=False)
    d=[]
    for b,q in common.groupby("disagreement_bucket",observed=False):
        if q.empty: continue
        ww=q.winner.value_counts().to_dict(); nt=ww.get("MODEL",0)+ww.get("MARKET",0)
        d.append({"bucket":str(b),"n":len(q),"mean_abs_disagreement":q.abs_disagreement.mean(),"model_mae":q.model_abs_error.mean(),"market_mae":q.market_abs_error.mean(),
                  "model_mae_gain_vs_market":q.market_abs_error.mean()-q.model_abs_error.mean(),"model_closer":ww.get("MODEL",0),"market_closer":ww.get("MARKET",0),"ties":ww.get("TIE",0),
                  "model_closer_rate_ex_ties":ww.get("MODEL",0)/max(nt,1)})
    disagreement=pd.DataFrame(d)
    sig=[]
    for t in [0,5,10,15]:
        for direction in ["MODEL_OVER_MARKET","MODEL_UNDER_MARKET"]:
            if direction=="MODEL_OVER_MARKET":
                q=common.loc[common.model_minus_market.ge(t) if t else common.model_minus_market.gt(0)].copy(); correct=q.actual_rush_yards>q.consensus_line
            else:
                q=common.loc[common.model_minus_market.le(-t) if t else common.model_minus_market.lt(0)].copy(); correct=q.actual_rush_yards<q.consensus_line
            push=q.actual_rush_yards.eq(q.consensus_line); ww=q.winner.value_counts().to_dict(); nt=ww.get("MODEL",0)+ww.get("MARKET",0)
            sig.append({"threshold":t,"direction":direction,"n":len(q),"pushes":int(push.sum()),
                        "directional_market_side_accuracy_ex_pushes":float(correct.loc[~push].mean()) if (~push).any() else np.nan,
                        "model_closer_rate_ex_ties":ww.get("MODEL",0)/max(nt,1),"model_mae":q.model_abs_error.mean() if len(q) else np.nan,
                        "market_mae":q.market_abs_error.mean() if len(q) else np.nan})
    signal=pd.DataFrame(sig)
    week=[]
    for wk,q in common.groupby("week"):
        week.append({"week":wk,"n":len(q),"model_mae":q.model_abs_error.mean(),"market_mae":q.market_abs_error.mean(),"model_gain_vs_market":q.market_abs_error.mean()-q.model_abs_error.mean()})
    return summary,h2h,disagreement,signal,pd.DataFrame(week)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m94c-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    market,source,bet=load_market(); m94=load_m94c(a.m94c_root); case,join_audit=join(m94,market); summary,h2h,disagreement,signal,week=summarize(case)
    market.to_csv(a.out_dir/"rb_market_normalized_book_lines.csv",index=False); source.to_csv(a.out_dir/"rb_market_source_audit.csv",index=False); bet.to_csv(a.out_dir/"rb_market_bet_type_audit.csv",index=False)
    join_audit.to_csv(a.out_dir/"rb_market_join_audit.csv",index=False); case.sort_values(["week","player"]).to_csv(a.out_dir/"rb_market_casebook.csv",index=False)
    summary.to_csv(a.out_dir/"rb_market_summary.csv",index=False); h2h.to_csv(a.out_dir/"rb_market_head_to_head.csv",index=False); disagreement.to_csv(a.out_dir/"rb_market_disagreement_buckets.csv",index=False)
    signal.to_csv(a.out_dir/"rb_market_directional_signal.csv",index=False); week.to_csv(a.out_dir/"rb_market_by_week.csv",index=False)
    print("=== source ===\n",source.to_string(index=False)); print("=== join ===\n",join_audit.to_string(index=False)); print("=== summary ===\n",summary.to_string(index=False)); print("=== h2h ===\n",h2h.to_string(index=False)); print("=== disagreement ===\n",disagreement.to_string(index=False)); print("=== signal ===\n",signal.to_string(index=False)); print("=== week ===\n",week.to_string(index=False))

if __name__=="__main__": main()
