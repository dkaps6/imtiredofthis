#!/usr/bin/env python3
"""Freeze a genuinely market-independent stable-QB football frontier.

Canonical v1 contained market-trained residual corrections. The first v2 attempt
removed those corrections but recovered its rows from a trace whose cohort had
already been filtered to games with a non-null market total. V3 fixes the cohort
itself: it consumes the pre-market historical component traces produced directly
by `decompose_qb_passing_error.py` / `component_predictions.predict_week`, then
applies only the M47 realized stable-primary definition (actual primary QB and
>=80% of team official QB pass attempts).

No sportsbook field is loaded or used for row selection, projection, or output.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

SNAPSHOT_ID = "qb_frontier_canonical_v3_football_only"

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()
def num(x): return pd.to_numeric(x, errors="coerce")
def key(s): return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
def met(a,p):
    z=pd.DataFrame({"a":num(a),"p":num(p)}).dropna(); e=z.p-z.a
    return {"n":len(z),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),
            "bias":float(e.mean()),"corr":float(z.a.corr(z.p)) if len(z)>2 else np.nan,
            "tail100":int(e.abs().ge(100).sum())}

def load_one(trace_path: Path, logs_path: Path, season: int) -> pd.DataFrame:
    t=pd.read_csv(trace_path, low_memory=False); l=pd.read_csv(logs_path, low_memory=False)
    t.columns=[str(c).strip().lower() for c in t.columns]; l.columns=[str(c).strip().lower() for c in l.columns]
    req_t={"week","team","player_clean_key","mc_proj","pred_attempts","pred_ypa","actual_pass_att","actual_pass_yards_raw"}
    req_l={"season","week","team","player","pass_att"}
    if req_t-set(t): raise RuntimeError(f"{season} premarket trace missing {sorted(req_t-set(t))}")
    if req_l-set(l): raise RuntimeError(f"{season} player logs missing {sorted(req_l-set(l))}")
    if "market" in t:
        t=t[t.market.astype(str).eq("pass_yards")].copy()
    t["season"]=season; t["week"]=num(t.week).astype(int); t["player_clean_key"]=key(t.player_clean_key)
    # The premarket trace must itself contain no schedule-market variables reachable by this freezer.
    prohibited=[c for c in t if c.startswith("market_") or c in {"spread_line","total_line","moneyline"}]
    # Presence in a broad diagnostic trace is tolerated only because no such column is selected below;
    # row selection is performed solely from football trace keys and realized QB participation.
    l=l[num(l.season).eq(season)].copy(); l["week"]=num(l.week).astype(int); l["player_clean_key"]=key(l.get("player_clean_key",l.player)); l["pa"]=num(l.pass_att).fillna(0)
    totals=l.groupby(["week","team"],as_index=False).pa.sum().rename(columns={"pa":"team_pa"})
    q=l.merge(totals,on=["week","team"],how="left")
    q["share"]=q.pa/q.team_pa.replace(0,np.nan)
    prim=q.sort_values(["week","team","pa"],ascending=[True,True,False]).drop_duplicates(["week","team"])[["week","team","player_clean_key","pa","team_pa","share"]]
    prim=prim.rename(columns={"player_clean_key":"actual_primary_key","pa":"actual_primary_attempts","share":"actual_qb_attempt_share"})
    x=t.merge(prim,on=["week","team"],how="left",validate="many_to_one")
    x=x[x.player_clean_key.eq(x.actual_primary_key) & num(x.actual_qb_attempt_share).ge(.80)].copy()
    if x.duplicated(["week","team","player_clean_key"]).any(): raise RuntimeError(f"{season} duplicate stable-QB keys")
    opponent=x.opponent.astype(str) if "opponent" in x else pd.Series("",index=x.index)
    out=pd.DataFrame({
      "season":season,"week":num(x.week).astype(int),"team":x.team.astype(str),"opponent":opponent,
      "player_clean_key":x.player_clean_key.astype(str),"actual_pass_yards":num(x.actual_pass_yards_raw),
      "actual_attempts":num(x.actual_pass_att),"pred_pass_yards":num(x.mc_proj),"pred_attempts":num(x.pred_attempts),"pred_ypa":num(x.pred_ypa),
      "actual_qb_attempt_share":num(x.actual_qb_attempt_share),
    })
    out["actual_ypa"]=out.actual_pass_yards/out.actual_attempts.replace(0,np.nan)
    out["implied_pred_ypa"]=out.pred_pass_yards/out.pred_attempts.replace(0,np.nan)
    out["det_pass_yards"]=out.pred_attempts*out.pred_ypa
    if out[["actual_pass_yards","actual_attempts","pred_pass_yards","pred_attempts"]].isna().any().any(): raise RuntimeError(f"{season} missing core stable-QB values")
    return out.sort_values(["week","team","player_clean_key"]).reset_index(drop=True), prohibited

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace-2024",type=Path,required=True); ap.add_argument("--logs-2024",type=Path,required=True)
    ap.add_argument("--trace-2025",type=Path,required=True); ap.add_argument("--logs-2025",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args()
    f24,p24=load_one(a.trace_2024,a.logs_2024,2024); f25,p25=load_one(a.trace_2025,a.logs_2025,2025)
    snap=pd.concat([f24,f25],ignore_index=True).sort_values(["season","week","team","player_clean_key"]).reset_index(drop=True)
    if len(f24)<250 or len(f25)<250: raise RuntimeError(f"stable-QB cohort unexpectedly small: 2024={len(f24)} 2025={len(f25)}")
    if snap.duplicated(["season","week","team","player_clean_key"]).any(): raise RuntimeError("v3 key uniqueness failed")
    if any(any(k in c.lower() for k in ["market","spread","moneyline","total_line"]) for c in snap.columns): raise RuntimeError("market field leaked into v3 output")
    a.out_dir.mkdir(parents=True,exist_ok=True); p=a.out_dir/f"{SNAPSHOT_ID}.csv"; snap.to_csv(p,index=False,float_format="%.10g")
    rows=[]
    for label,g in [("combined",snap),("2024",f24),("2025",f25)]:
        cur=met(g.actual_pass_yards,g.pred_pass_yards); oa=met(g.actual_pass_yards,g.actual_attempts*g.implied_pred_ypa); oy=met(g.actual_pass_yards,g.pred_attempts*g.actual_ypa)
        rows += [{"season":label,"candidate":"football_only_current",**cur,"mae_gain_vs_current":0.0},
                 {"season":label,"candidate":"oracle_actual_attempts",**oa,"mae_gain_vs_current":cur["mae"]-oa["mae"]},
                 {"season":label,"candidate":"oracle_actual_ypa",**oy,"mae_gain_vs_current":cur["mae"]-oy["mae"]}]
    pd.DataFrame(rows).to_csv(a.out_dir/"football_only_oracle_summary.csv",index=False)
    manifest={"snapshot_id":SNAPSHOT_ID,"schema_version":3,"row_count":len(snap),"season_rows":{"2024":len(f24),"2025":len(f25)},"snapshot_sha256":sha256(p),
      "cohort_definition":"projected pass-yards QB who was realized team primary and handled >=80% of team QB official attempts; selected before any market join/filter",
      "projection_source":"decompose_qb_passing_error.py -> component_predictions.predict_week pre-market historical pipeline",
      "market_boundary":"No sportsbook field is used for row selection, projection, or output.",
      "diagnostic_market_columns_present_but_unread":{"2024":p24,"2025":p25}}
    (a.out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2)); print(pd.DataFrame(rows).to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
