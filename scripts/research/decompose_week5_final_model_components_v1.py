#!/usr/bin/env python3
"""Week-5 final-model opportunity vs efficiency decomposition.

Uses immutable Week-5 final point projections and the pregame football workload
state. Outcomes enter only after all pregame quantities are loaded. Diagnostic
only; no fitting or production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly

SEASON=2026
WEEK=5
TOL=1e-10
CELLS=[
    ("QB","pass_yards","pass_attempts"),
    ("RB","rush_yards","carries"),
    ("RB","rec_yards","targets"),
    ("RB","receptions","targets"),
    ("WR","rec_yards","targets"),
    ("WR","receptions","targets"),
    ("TE","rec_yards","targets"),
    ("TE","receptions","targets"),
]


def _read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def _num(s): return pd.to_numeric(s,errors="coerce")


def _to_pandas(obj:Any)->pd.DataFrame:
    if isinstance(obj,pd.DataFrame): return obj.copy()
    if hasattr(obj,"to_dicts"): return pd.DataFrame(obj.to_dicts())
    if hasattr(obj,"to_pandas"): return obj.to_pandas()
    return pd.DataFrame(obj)


def _pos(v)->str:
    p=str(v or "").strip().upper()
    if p in {"HB","FB","TB"} or p.startswith("RB"): return "RB"
    if p in {"LWR","RWR","SWR"} or p.startswith("WR"): return "WR"
    if p.startswith("TE"): return "TE"
    if p.startswith("QB"): return "QB"
    return p


def _score(actual,pred,prefix):
    y=_num(actual);p=_num(pred);ok=y.notna()&p.notna();y=y[ok];p=p[ok]
    if not len(y):
        return {f"{prefix}_rows":0,f"{prefix}_mae":np.nan,f"{prefix}_bias":np.nan,f"{prefix}_rmse":np.nan}
    e=p-y
    return {
        f"{prefix}_rows":int(len(e)),
        f"{prefix}_mae":float(e.abs().mean()),
        f"{prefix}_bias":float(e.mean()),
        f"{prefix}_rmse":float(np.sqrt(np.mean(e**2))),
    }


def _final_rush_probabilities(fsu:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for team,g in fsu.groupby("team",sort=True):
        g=g.drop_duplicates(["team","player_clean_key"],keep="last").copy()
        plays=_num(g["rules_plays_est"]).dropna()
        rate=_num(g["rules_pass_rate"]).dropna()
        if not len(plays) or not len(rate):
            continue
        if float(plays.max()-plays.min())>1e-8 or float(rate.max()-rate.min())>1e-8:
            raise RuntimeError(f"team rule-input drift team={team}")
        team_rush=float(plays.iloc[0])*(1.0-float(rate.iloc[0]))
        raw=_num(g["rules_rush_share"]).fillna(0.0).clip(0.0,0.95).to_numpy(float)
        if len(raw)>5:
            order=np.argsort(-raw,kind="stable")
            keep=set(order[:5].tolist())
            raw=np.asarray([v if i in keep else 0.0 for i,v in enumerate(raw)],float)
        s=float(raw.sum())
        if s>0.95:
            raw*=0.95/s
        for (_,r),share in zip(g.iterrows(),raw):
            rows.append({
                "team":team,
                "player_clean_key":str(r.player_clean_key),
                "predicted_carries":team_rush*float(share),
                "predicted_rush_probability":float(share),
                "predicted_team_rush_attempts":team_rush,
            })
    return pd.DataFrame(rows)


def _target_opportunities(fsu:pd.DataFrame,ent:pd.DataFrame)->pd.DataFrame:
    team_inputs=(fsu.drop_duplicates(["team","player_clean_key"],keep="last")
        .groupby("team",as_index=False)
        .agg(rules_plays_est=("rules_plays_est","first"),rules_pass_rate=("rules_pass_rate","first")))
    team_inputs["predicted_team_pass_attempts"]=_num(team_inputs["rules_plays_est"])*_num(team_inputs["rules_pass_rate"])
    x=ent[["team","player_clean_key","entitlement_tgt_share"]].copy()
    x["team"]=x["team"].map(canon_team)
    x=x.merge(team_inputs[["team","predicted_team_pass_attempts"]],on="team",how="left",validate="many_to_one")
    x["predicted_targets"]=_num(x["entitlement_tgt_share"])*_num(x["predicted_team_pass_attempts"])
    return x


def _qb_attempts(priced:pd.DataFrame)->pd.DataFrame:
    q=priced.loc[priced["market"].eq("pass_yards")].copy()
    rows=[]
    for (team,key),g in q.groupby(["team","player_clean_key"],sort=True):
        vals=_num(g["qb_pred_attempts"]).dropna().unique()
        if len(vals)!=1:
            raise RuntimeError(f"QB attempt state drift {team} {key}: {vals}")
        rows.append({"team":team,"player_clean_key":key,"predicted_pass_attempts":float(vals[0])})
    return pd.DataFrame(rows)


def _actual_usage()->pd.DataFrame:
    import nflreadpy as nfl
    raw=_to_pandas(nfl.load_player_stats(seasons=[SEASON],summary_level="week"))
    x=_normalize_weekly(raw,SEASON)
    x=x.loc[_num(x["week"]).eq(WEEK)].copy()
    x["team"]=x["team"].map(canon_team)
    x["player_id"]=x["player_id"].fillna("").astype(str).str.strip()
    x["position_family"]=x["position"].map(_pos)
    return x[[
        "team","player_id","player_identity_key","position_family",
        "pass_att","pass_yards","rushes","rush_yards",
        "targets","receptions","rec_yards"
    ]].copy()


def build_rows(full,priced,fsu,ent)->pd.DataFrame:
    full=full.loc[full["football_settlement_status"].eq("SETTLED")].copy()
    full["position_family"]=full["position"].map(_pos)
    full["team"]=full["team"].map(canon_team)
    priced["team"]=priced["team"].map(canon_team)
    fsu["team"]=fsu["team"].map(canon_team)
    ent["team"]=ent["team"].map(canon_team)

    rush=_final_rush_probabilities(fsu)
    tgt=_target_opportunities(fsu,ent)
    qb=_qb_attempts(priced)
    actual=_actual_usage()
    actual_map={(str(r.team),str(r.player_id)):r for r in actual.itertuples(index=False)}

    rows=[]
    for pos,market,opp_type in CELLS:
        z=full.loc[full["position_family"].eq(pos)&full["market"].eq(market)].copy()
        for r in z.itertuples(index=False):
            pred=np.nan
            if opp_type=="pass_attempts":
                q=qb.loc[qb["team"].eq(r.team)&qb["player_clean_key"].eq(r.player_clean_key),"predicted_pass_attempts"]
                if len(q)==1: pred=float(q.iloc[0])
            elif opp_type=="carries":
                q=rush.loc[rush["team"].eq(r.team)&rush["player_clean_key"].eq(r.player_clean_key),"predicted_carries"]
                if len(q)==1: pred=float(q.iloc[0])
            else:
                q=tgt.loc[tgt["team"].eq(r.team)&tgt["player_clean_key"].eq(r.player_clean_key),"predicted_targets"]
                if len(q)==1: pred=float(q.iloc[0])

            a=actual_map.get((str(r.team),str(r.gsis_id)))
            if a is not None:
                if opp_type=="pass_attempts": act=float(a.pass_att)
                elif opp_type=="carries": act=float(a.rushes)
                else: act=float(a.targets)
            else:
                # The full-board GSIS settlement already verified these rows as
                # legitimate zero-usage outcomes when no weekly stat row exists.
                act=0.0

            baseline=float(r.model_proj); actual_output=float(r.actual)
            model_eff=(baseline/pred) if np.isfinite(pred) and pred>0 else np.nan
            actual_eff=(actual_output/act) if act>0 else np.nan
            opp_oracle=act*model_eff if np.isfinite(model_eff) else np.nan
            eff_oracle=pred*actual_eff if np.isfinite(actual_eff) and np.isfinite(pred) else np.nan
            rows.append({
                "player":r.player,"player_clean_key":r.player_clean_key,"gsis_id":r.gsis_id,
                "team":r.team,"opponent":r.opponent,"position_family":pos,"market":market,
                "opportunity_type":opp_type,
                "baseline_projection":baseline,"actual_output":actual_output,
                "predicted_opportunities":pred,"actual_opportunities":act,
                "model_effective_efficiency":model_eff,"actual_efficiency":actual_eff,
                "opportunity_oracle":opp_oracle,"efficiency_oracle":eff_oracle,
                "baseline_abs_error":abs(baseline-actual_output),
                "opportunity_oracle_abs_error":abs(opp_oracle-actual_output) if np.isfinite(opp_oracle) else np.nan,
                "efficiency_oracle_abs_error":abs(eff_oracle-actual_output) if np.isfinite(eff_oracle) else np.nan,
                "opportunity_error":pred-act if np.isfinite(pred) else np.nan,
            })
    out=pd.DataFrame(rows)
    if out["predicted_opportunities"].isna().any():
        bad=out.loc[out["predicted_opportunities"].isna(),["player","team","position_family","market"]]
        raise RuntimeError(f"missing predicted opportunity state: {bad.to_dict('records')}")
    return out


def summarize(g:pd.DataFrame)->dict:
    out={"rows":int(len(g))}
    out.update(_score(g["actual_output"],g["baseline_projection"],"baseline"))
    out.update(_score(g["actual_output"],g["opportunity_oracle"],"opportunity_oracle"))
    b=out["baseline_mae"];o=out["opportunity_oracle_mae"]
    out["opportunity_oracle_mae_improvement"]=float(b-o)
    out["opportunity_oracle_fraction_mae_removed"]=float((b-o)/b) if b>0 else np.nan

    e=g.loc[g["actual_efficiency"].notna()&g["efficiency_oracle"].notna()].copy()
    out["efficiency_eligible_rows"]=int(len(e))
    if len(e):
        eb=_score(e["actual_output"],e["baseline_projection"],"efficiency_subset_baseline")
        eo=_score(e["actual_output"],e["efficiency_oracle"],"efficiency_oracle")
        out.update(eb);out.update(eo)
        bm=eb["efficiency_subset_baseline_mae"];em=eo["efficiency_oracle_mae"]
        out["efficiency_oracle_mae_improvement"]=float(bm-em)
        out["efficiency_oracle_fraction_mae_removed"]=float((bm-em)/bm) if bm>0 else np.nan
    else:
        out["efficiency_oracle_mae_improvement"]=np.nan
        out["efficiency_oracle_fraction_mae_removed"]=np.nan
    oe=g["opportunity_error"];be=g["baseline_projection"]-g["actual_output"]
    out["opportunity_error_vs_output_error_pearson"]=float(oe.corr(be)) if oe.nunique()>1 and be.nunique()>1 else np.nan
    out["mean_predicted_opportunities"]=float(g["predicted_opportunities"].mean())
    out["mean_actual_opportunities"]=float(g["actual_opportunities"].mean())
    return out


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--full-board-detail",type=Path,required=True)
    p.add_argument("--priced",type=Path,required=True)
    p.add_argument("--football-universe",type=Path,required=True)
    p.add_argument("--entitlement-trace",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    full=_read(a.full_board_detail,"full board")
    priced=_read(a.priced,"priced")
    fsu=_read(a.football_universe,"football universe")
    ent=_read(a.entitlement_trace,"entitlement trace")
    rows=build_rows(full,priced,fsu,ent)
    summaries=[]
    for (pos,market),g in rows.groupby(["position_family","market"],sort=True):
        rec={"position":pos,"market":market};rec.update(summarize(g));summaries.append(rec)
    s=pd.DataFrame(summaries)
    case=rows.copy()
    case["opp_oracle_improvement"]=case["baseline_abs_error"]-case["opportunity_oracle_abs_error"]
    case["eff_oracle_improvement"]=case["baseline_abs_error"]-case["efficiency_oracle_abs_error"]
    case=case.sort_values("baseline_abs_error",ascending=False).head(60)
    rows.to_csv(a.out_dir/"week5_component_decomposition_rows.csv",index=False)
    s.to_csv(a.out_dir/"week5_component_decomposition_scorecard.csv",index=False)
    case.to_csv(a.out_dir/"week5_component_decomposition_casebook.csv",index=False)
    result={
        "version":"WEEK5_FINAL_MODEL_COMPONENT_DECOMPOSITION_V1",
        "season":SEASON,"week":WEEK,
        "scorecard":summaries,
        "rows":int(len(rows)),
        "model_fitting":False,
        "sportsbook_inputs_used_for_feature_construction":False,
        "production_changed":False,
    }
    (a.out_dir/"week5_component_decomposition_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
