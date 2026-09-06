#!/usr/bin/env python3
"""WR-R3 frozen walk-forward individual player-error persistence diagnostic."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 12396
HIST = 8
MIN_PRIOR = 4
MIN_ROWS = 7000
SEASONS = list(range(2020, 2026))


def _one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, got {len(hits)}")
    return hits[0]


def _read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"empty {path}")
    return x


def _key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_walkforward(paired: pd.DataFrame) -> pd.DataFrame:
    q = paired.loc[paired["market"].astype(str).str.lower().eq("rec_yards") & paired["position"].astype(str).str.upper().eq("WR")].copy()
    if len(q) != EXPECTED_ROWS:
        raise RuntimeError(f"M38 row parity expected={EXPECTED_ROWS} got={len(q)}")
    q["season"] = _num(q["season"]).astype(int)
    q["week"] = _num(q["week"]).astype(int)
    q["player_key"] = q["player_clean_key"].map(_key)
    q["actual"] = _num(q["actual_m38"])
    q["proj"] = _num(q["mc_proj_m38"])
    q["error"] = q["proj"] - q["actual"]
    q["abs_error"] = q["error"].abs()
    q["miss30"] = q["abs_error"].ge(30).astype(int)
    if sorted(q["season"].unique().tolist()) != SEASONS or q[["actual","proj"]].isna().any().any():
        raise RuntimeError("canonical M38 integrity drift")
    q = q.sort_values(["season","week","player_key"], kind="stable").reset_index(drop=True)

    histories: dict[str, list[dict]] = {}
    rows = []
    for r in q.itertuples(index=False):
        h = histories.get(r.player_key, [])[-HIST:]
        rec = {
            "season":r.season,"week":r.week,"team":r.team,"player":r.player,"player_key":r.player_key,
            "actual":r.actual,"proj":r.proj,"target_error":r.error,"target_abs_error":r.abs_error,"target_miss30":r.miss30,
            "prior_games":len(h),
        }
        if h:
            d = pd.DataFrame(h)
            rec["prior8_m38_bias"] = float(d["error"].mean())
            rec["prior8_m38_mae"] = float(d["abs_error"].mean())
            rec["prior8_m38_miss30_rate"] = float(d["miss30"].mean())
            rec["last_prior_season"] = int(d.iloc[-1]["season"])
            rec["last_prior_week"] = int(d.iloc[-1]["week"])
        else:
            for c in ["prior8_m38_bias","prior8_m38_mae","prior8_m38_miss30_rate","last_prior_season","last_prior_week"]:
                rec[c]=np.nan
        rows.append(rec)
        histories.setdefault(r.player_key, []).append({"season":r.season,"week":r.week,"error":r.error,"abs_error":r.abs_error,"miss30":r.miss30})
    out=pd.DataFrame(rows)
    leak=out.loc[out["last_prior_season"].notna() & ((out["last_prior_season"]>out["season"]) | ((out["last_prior_season"]==out["season"]) & (out["last_prior_week"]>=out["week"])))]
    if len(leak):
        raise RuntimeError(f"leakage rows={len(leak)}")
    return out


def _gap(g:pd.DataFrame, feature:str, outcome:str)->float:
    f=_num(g[feature]); y=_num(g[outcome]); q25=float(f.quantile(.25)); q75=float(f.quantile(.75))
    return float(y.loc[f.ge(q75)].mean()-y.loc[f.le(q25)].mean())


def _season_gap(g,feature,outcome):
    d={}
    for s in SEASONS:
        q=g.loc[g["season"].eq(s)]
        d[s]=_gap(q,feature,outcome) if len(q)>=100 and _num(q[feature]).nunique()>=4 else np.nan
    return d


def score(wf:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    g=wf.loc[wf["prior_games"].ge(MIN_PRIOR)].copy()
    rows=[]

    sp=float(_num(g["prior8_m38_bias"]).corr(_num(g["target_error"]),method="spearman")); gap=_gap(g,"prior8_m38_bias","target_error")
    sq=g.loc[_num(g["prior8_m38_bias"]).abs().ge(3)]
    sign=float((np.sign(_num(sq["prior8_m38_bias"]))==np.sign(_num(sq["target_error"]))).mean()) if len(sq) else np.nan
    sg=_season_gap(g,"prior8_m38_bias","target_error"); pos=sum(np.isfinite(v) and v>0 for v in sg.values())
    pa=bool(len(g)>=MIN_ROWS and sp>=.08 and gap>=6 and sign>=.55 and pos>=4 and sg.get(2024,-np.inf)>0 and sg.get(2025,-np.inf)>0)
    rows.append({"diagnostic":"DIRECTIONAL_BIAS_PERSISTENCE","rows":len(g),"spearman":sp,"quartile_gap":gap,"sign_agreement":sign,"positive_seasons":pos,"gap_2024":sg.get(2024),"gap_2025":sg.get(2025),"enrichment":np.nan,"passes":pa})

    spb=float(_num(g["prior8_m38_mae"]).corr(_num(g["target_abs_error"]),method="spearman")); gapb=_gap(g,"prior8_m38_mae","target_abs_error")
    sgb=_season_gap(g,"prior8_m38_mae","target_abs_error"); posb=sum(np.isfinite(v) and v>0 for v in sgb.values())
    pb=bool(len(g)>=MIN_ROWS and spb>=.08 and gapb>=4 and posb>=4 and sgb.get(2024,-np.inf)>0 and sgb.get(2025,-np.inf)>0)
    rows.append({"diagnostic":"INDIVIDUAL_DIFFICULTY_PERSISTENCE","rows":len(g),"spearman":spb,"quartile_gap":gapb,"sign_agreement":np.nan,"positive_seasons":posb,"gap_2024":sgb.get(2024),"gap_2025":sgb.get(2025),"enrichment":np.nan,"passes":pb})

    f=_num(g["prior8_m38_miss30_rate"]); q75=float(f.quantile(.75)); high=f.ge(q75)
    overall=float(g["target_miss30"].mean()); enrich=float(g.loc[high,"target_miss30"].mean()/overall) if overall>0 else np.nan
    season_enrich={}; pose=0
    for s in SEASONS:
        q=g.loc[g["season"].eq(s)]; fv=_num(q["prior8_m38_miss30_rate"])
        if len(q)>=100 and fv.nunique()>=2 and float(q["target_miss30"].mean())>0:
            hi=fv.ge(float(fv.quantile(.75))); e=float(q.loc[hi,"target_miss30"].mean()/q["target_miss30"].mean()); season_enrich[s]=e; pose+=int(e>1)
        else: season_enrich[s]=np.nan
    pc=bool(len(g)>=MIN_ROWS and np.isfinite(enrich) and enrich>=1.25 and pose>=4 and season_enrich.get(2024,0)>1 and season_enrich.get(2025,0)>1)
    rows.append({"diagnostic":"EXTREME_MISS_PERSISTENCE","rows":len(g),"spearman":np.nan,"quartile_gap":np.nan,"sign_agreement":np.nan,"positive_seasons":pose,"gap_2024":season_enrich.get(2024),"gap_2025":season_enrich.get(2025),"enrichment":enrich,"passes":pc})

    metrics=pd.DataFrame(rows); winners=metrics.loc[metrics["passes"],"diagnostic"].tolist()
    summary={"migration":"WR_R3_PLAYER_ERROR_PERSISTENCE","source_rows":len(wf),"scoreable_rows":len(g),"players":int(wf["player_key"].nunique()),"history_window":HIST,"minimum_prior_games":MIN_PRIOR,"walk_forward_leakage_violations":0,"sportsbook_inputs_used":False,"model_fitting_used":False,"production_changed":False,"passing_diagnostics":winners,"disposition":"WR_PLAYER_ERROR_PERSISTENCE_DETECTED" if winners else "NO_ACTIONABLE_WR_PLAYER_ERROR_PERSISTENCE"}
    return metrics,summary


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--wr-r1-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,default=Path("data/backtests/wr_r3_player_error_persistence")); a=ap.parse_args()
    paired=_read(_one(a.wr_r1_root,"wr_r1_paired_wr_casebook.csv")); wf=build_walkforward(paired); metrics,summary=score(wf)
    a.out_dir.mkdir(parents=True,exist_ok=True); wf.to_csv(a.out_dir/"wr_r3_walkforward_casebook.csv",index=False); metrics.to_csv(a.out_dir/"wr_r3_metrics.csv",index=False); (a.out_dir/"wr_r3_result.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    print(metrics.to_string(index=False)); print(json.dumps(summary,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
