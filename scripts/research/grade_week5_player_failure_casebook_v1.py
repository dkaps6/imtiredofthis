#!/usr/bin/env python3
"""Week-5 player failure casebook V1.

Postgame diagnostic only. Reconstructs frozen pregame opportunity/efficiency
components for the top-40 Week-5 final-model misses and compares them with
official nflverse weekly stats.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly
from scripts.simulation_v2 import _sharpen_wr_target_shares, _top_n_shares

VERSION="WEEK5_PLAYER_FAILURE_CASEBOOK_V1"
SEASON=2026
WEEK=5
TOP_N=40
RB_POS={"RB","HB","FB"}


def _num(s:pd.Series)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def _clean_key(v:Any)->str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def _finite(v:Any)->float | None:
    try:
        x=float(v)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def _read_csv(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def _load_actuals()->pd.DataFrame:
    import nflreadpy as nfl
    raw=nfl.load_player_stats(seasons=[SEASON],summary_level="week")
    if hasattr(raw,"to_dicts"):
        raw=pd.DataFrame(raw.to_dicts())
    elif hasattr(raw,"to_pandas"):
        raw=raw.to_pandas()
    x=_normalize_weekly(raw,SEASON)
    x=x.loc[_num(x["week"]).eq(WEEK)].copy()
    if x.empty:
        raise RuntimeError("nflverse Week-5 player stats are empty")
    keep=[
        "player_identity_key","player_id","player","player_clean_key","team","position",
        "targets","receptions","rec_yards","rushes","rush_yards","pass_att","pass_yards",
        "team_targets","team_rushes","tgt_share_game","rush_share_game","ypt_game",
        "ypc_game","catch_rate_game",
    ]
    keep=[c for c in keep if c in x.columns]
    return x[keep].drop_duplicates(["player_identity_key","team"],keep="last")


def _unique_scalar(g:pd.DataFrame,col:str)->float | None:
    if col not in g.columns:
        return None
    vals=_num(g[col]).dropna().unique()
    if len(vals)==0:
        return None
    if len(vals)>1 and np.nanmax(vals)-np.nanmin(vals)>1e-9:
        # Component inputs are expected to be identical across a player's market rows.
        return float(np.nanmedian(vals))
    return float(vals[0])


def _build_player_components(metrics:pd.DataFrame)->pd.DataFrame:
    req={"team","opponent","player","player_clean_key","player_identity_key","position",
         "rules_plays_est","rules_pass_rate","rules_tgt_share","rules_rush_share",
         "rules_ypt","rules_ypc","rules_ypa","rules_catch_rate"}
    miss=req-set(metrics.columns)
    if miss:
        raise RuntimeError(f"model inputs missing required fields: {sorted(miss)}")

    metrics=metrics.copy()
    metrics["team"]=metrics["team"].map(canon_team)
    metrics["opponent"]=metrics["opponent"].map(canon_team)
    if "event_id" not in metrics.columns:
        metrics["event_id"]=metrics.apply(
            lambda r:"|".join(sorted([str(r.get("team","")),str(r.get("opponent",""))])),
            axis=1,
        )

    rows=[]
    grouped=metrics.sort_values(["event_id","team","player_clean_key"]).groupby(
        ["event_id","team","player_clean_key"],dropna=False,sort=True
    )
    for (event_id,team,pkey),g in grouped:
        last=g.iloc[-1]
        rec={
            "event_id":event_id,
            "team":team,
            "opponent":canon_team(last.get("opponent")),
            "player":str(last.get("player") or ""),
            "player_clean_key":str(pkey),
            "player_identity_key":str(last.get("player_identity_key") or ""),
            "position":str(last.get("position") or "").upper(),
        }
        for c in (
            "rules_plays_est","rules_pass_rate","rules_tgt_share","rules_rush_share",
            "rules_ypt","rules_ypc","rules_ypa","rules_catch_rate"
        ):
            rec[c]=_unique_scalar(g,c)
        rows.append(rec)

    players=pd.DataFrame(rows)
    if players.empty:
        raise RuntimeError("zero frozen player component rows")

    target_probs=[]
    rush_probs=[]
    for (event_id,team),g in players.groupby(["event_id","team"],sort=False):
        idx=g.index.to_numpy()
        raw_tgt=np.array([
            float(x) if x is not None and np.isfinite(x) else 0.0
            for x in g["rules_tgt_share"].tolist()
        ],dtype=float)
        sharpened=_sharpen_wr_target_shares(g,raw_tgt)
        sharpened=np.clip(np.nan_to_num(sharpened,nan=0.0,posinf=0.0,neginf=0.0),0.0,0.95)
        if sharpened.sum()>0.95:
            sharpened*=0.95/float(sharpened.sum())

        raw_rush=np.array([
            float(x) if x is not None and np.isfinite(x) else 0.0
            for x in g["rules_rush_share"].tolist()
        ],dtype=float)
        rush=_top_n_shares(raw_rush,5)
        rush=np.clip(np.nan_to_num(rush,nan=0.0,posinf=0.0,neginf=0.0),0.0,0.95)
        if rush.sum()>0.95:
            rush*=0.95/float(rush.sum())

        target_probs.extend(zip(idx,sharpened))
        rush_probs.extend(zip(idx,rush))

    players["final_target_probability"]=np.nan
    players["final_rush_probability"]=np.nan
    for idx,val in target_probs:
        players.loc[idx,"final_target_probability"]=float(val)
    for idx,val in rush_probs:
        players.loc[idx,"final_rush_probability"]=float(val)

    plays=_num(players["rules_plays_est"])
    pr=_num(players["rules_pass_rate"])
    players["pred_team_pass_att"]=plays*pr
    players["pred_team_rush_att"]=plays*(1.0-pr)
    players["pred_targets"]=players["pred_team_pass_att"]*_num(players["final_target_probability"])
    players["pred_carries"]=players["pred_team_rush_att"]*_num(players["final_rush_probability"])
    return players


def _qb_promoted(priced:pd.DataFrame)->pd.DataFrame:
    cols={"team","player_clean_key"}
    if not cols.issubset(priced.columns):
        return pd.DataFrame(columns=["team","player_clean_key","qb_pred_attempts","qb_pred_ypa","production_selected"])
    x=priced.copy()
    x["team"]=x["team"].map(canon_team)
    out=[]
    for (team,pkey),g in x.groupby(["team","player_clean_key"],dropna=False):
        def scalar(c):
            if c not in g.columns:
                return np.nan
            v=_num(g[c]).dropna()
            return float(v.iloc[0]) if len(v) else np.nan
        selected=False
        if "production_decision" in g.columns:
            selected=g["production_decision"].astype(str).str.upper().eq("BET").any()
        out.append({
            "team":team,"player_clean_key":pkey,
            "qb_pred_attempts":scalar("qb_pred_attempts"),
            "qb_pred_ypa":scalar("qb_pred_ypa"),
            "production_selected":bool(selected),
        })
    return pd.DataFrame(out)


def _component_label(opp:float | None,eff:float | None)->str:
    if opp is None or eff is None or not np.isfinite(opp) or not np.isfinite(eff):
        return "INSUFFICIENT_COMPONENT_EVIDENCE"
    ao=abs(opp); ae=abs(eff)
    if ao>=1.5*ae:
        return "OPPORTUNITY_DOMINANT"
    if ae>=1.5*ao:
        return "EFFICIENCY_DOMINANT"
    return "MIXED_OPPORTUNITY_EFFICIENCY"


def _market_components(r:pd.Series)->dict[str,Any]:
    market=str(r["market"])
    out={
        "pred_opportunity":np.nan,
        "actual_opportunity":np.nan,
        "pred_efficiency":np.nan,
        "actual_efficiency":np.nan,
        "opportunity_contribution":np.nan,
        "efficiency_contribution":np.nan,
        "rush_opportunity_contribution":np.nan,
        "rush_efficiency_contribution":np.nan,
        "rec_opportunity_contribution":np.nan,
        "rec_efficiency_contribution":np.nan,
        "component_label":"INSUFFICIENT_COMPONENT_EVIDENCE",
    }

    if market=="pass_yards":
        pred_opp=r.get("qb_pred_attempts")
        if not np.isfinite(pred_opp):
            pred_opp=r.get("pred_team_pass_att")
        pred_eff=r.get("qb_pred_ypa")
        if not np.isfinite(pred_eff):
            pred_eff=r.get("rules_ypa")
        act_opp=r.get("pass_att")
        act_yards=r.get("pass_yards")
        act_eff=(act_yards/act_opp) if np.isfinite(act_opp) and act_opp>0 and np.isfinite(act_yards) else np.nan
        if all(np.isfinite(x) for x in [pred_opp,pred_eff,act_opp,act_eff]):
            oc=(act_opp-pred_opp)*pred_eff
            ec=act_opp*(act_eff-pred_eff)
            out.update({
                "pred_opportunity":pred_opp,"actual_opportunity":act_opp,
                "pred_efficiency":pred_eff,"actual_efficiency":act_eff,
                "opportunity_contribution":oc,"efficiency_contribution":ec,
                "component_label":_component_label(oc,ec),
            })
        return out

    if market=="rush_yards":
        pred_opp=r.get("pred_carries"); pred_eff=r.get("rules_ypc")
        act_opp=r.get("rushes"); act_yards=r.get("rush_yards")
        act_eff=(act_yards/act_opp) if np.isfinite(act_opp) and act_opp>0 and np.isfinite(act_yards) else np.nan
        if all(np.isfinite(x) for x in [pred_opp,pred_eff,act_opp,act_eff]):
            oc=(act_opp-pred_opp)*pred_eff
            ec=act_opp*(act_eff-pred_eff)
            out.update({
                "pred_opportunity":pred_opp,"actual_opportunity":act_opp,
                "pred_efficiency":pred_eff,"actual_efficiency":act_eff,
                "opportunity_contribution":oc,"efficiency_contribution":ec,
                "component_label":_component_label(oc,ec),
            })
        return out

    if market in {"rec_yards","receptions"}:
        pred_opp=r.get("pred_targets")
        act_opp=r.get("targets")
        if market=="rec_yards":
            pred_eff=r.get("rules_ypt"); act_val=r.get("rec_yards")
            act_eff=(act_val/act_opp) if np.isfinite(act_opp) and act_opp>0 and np.isfinite(act_val) else np.nan
        else:
            pred_eff=r.get("rules_catch_rate"); act_val=r.get("receptions")
            act_eff=(act_val/act_opp) if np.isfinite(act_opp) and act_opp>0 and np.isfinite(act_val) else np.nan
        if all(np.isfinite(x) for x in [pred_opp,pred_eff,act_opp,act_eff]):
            oc=(act_opp-pred_opp)*pred_eff
            ec=act_opp*(act_eff-pred_eff)
            out.update({
                "pred_opportunity":pred_opp,"actual_opportunity":act_opp,
                "pred_efficiency":pred_eff,"actual_efficiency":act_eff,
                "opportunity_contribution":oc,"efficiency_contribution":ec,
                "component_label":_component_label(oc,ec),
            })
        elif np.isfinite(act_opp) and act_opp==0 and np.isfinite(pred_opp) and np.isfinite(pred_eff):
            oc=(0-pred_opp)*pred_eff
            out.update({
                "pred_opportunity":pred_opp,"actual_opportunity":0.0,
                "pred_efficiency":pred_eff,
                "opportunity_contribution":oc,
                "component_label":"OPPORTUNITY_DOMINANT",
            })
        return out

    if market=="rush_rec_yards":
        pc=r.get("pred_carries"); py=r.get("rules_ypc")
        ac=r.get("rushes"); ary=r.get("rush_yards")
        pt=r.get("pred_targets"); pyp=r.get("rules_ypt")
        at=r.get("targets"); arv=r.get("rec_yards")
        rush_eff=(ary/ac) if np.isfinite(ac) and ac>0 and np.isfinite(ary) else np.nan
        rec_eff=(arv/at) if np.isfinite(at) and at>0 and np.isfinite(arv) else np.nan

        roc=(ac-pc)*py if all(np.isfinite(x) for x in [ac,pc,py]) else np.nan
        recoc=(at-pt)*pyp if all(np.isfinite(x) for x in [at,pt,pyp]) else np.nan
        rec_eff_c=at*(rec_eff-pyp) if all(np.isfinite(x) for x in [at,rec_eff,pyp]) else (0.0 if np.isfinite(at) and at==0 else np.nan)
        rush_eff_c=ac*(rush_eff-py) if all(np.isfinite(x) for x in [ac,rush_eff,py]) else (0.0 if np.isfinite(ac) and ac==0 else np.nan)

        opp_vals=[x for x in [roc,recoc] if np.isfinite(x)]
        eff_vals=[x for x in [rush_eff_c,rec_eff_c] if np.isfinite(x)]
        if opp_vals and eff_vals:
            opp_total=sum(opp_vals); eff_total=sum(eff_vals)
            label=_component_label(opp_total,eff_total)
        else:
            opp_total=np.nan;eff_total=np.nan;label="INSUFFICIENT_COMPONENT_EVIDENCE"
        out.update({
            "rush_opportunity_contribution":roc,
            "rush_efficiency_contribution":rush_eff_c,
            "rec_opportunity_contribution":recoc,
            "rec_efficiency_contribution":rec_eff_c,
            "opportunity_contribution":opp_total,
            "efficiency_contribution":eff_total,
            "component_label":label,
        })
        return out

    return out


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worst-misses",type=Path,required=True)
    ap.add_argument("--model-inputs",type=Path,required=True)
    ap.add_argument("--priced-board",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    worst=_read_csv(a.worst_misses,"worst misses").head(TOP_N).copy()
    metrics=_read_csv(a.model_inputs,"frozen model inputs")
    priced=_read_csv(a.priced_board,"frozen priced board")

    players=_build_player_components(metrics)
    qb=_qb_promoted(priced)
    players=players.merge(qb,on=["team","player_clean_key"],how="left",validate="one_to_one")

    actual=_load_actuals()
    actual["team"]=actual["team"].map(canon_team)

    # Map casebook names to frozen player identity using exact team + canonicalized display name.
    players["case_name_key"]=players["player"].map(_clean_key)
    worst["team"]=worst["team"].map(canon_team)
    worst["case_name_key"]=worst["player"].map(_clean_key)
    ident=players.sort_values(["team","case_name_key"]).drop_duplicates(["team","case_name_key"],keep="last")
    case=worst.merge(
        ident,
        on=["team","case_name_key"],
        how="left",
        suffixes=("","_frozen"),
        validate="many_to_one",
    )

    if case["player_identity_key"].astype(str).isin({"","nan","None"}).any():
        missing=case.loc[
            case["player_identity_key"].astype(str).isin({"","nan","None"}),
            ["player","team","market"],
        ].to_dict("records")
        raise RuntimeError(f"casebook frozen identity miss: {missing}")

    case=case.merge(
        actual,
        on=["player_identity_key","team"],
        how="left",
        suffixes=("","_actual"),
        validate="many_to_one",
    )

    rows=[]
    for _,r in case.iterrows():
        comp=_market_components(r)
        rec=r.to_dict()
        rec.update(comp)
        med_line=r.get("median_book_line")
        actual_val=r.get("actual")
        rec["median_line_abs_error"]=(
            abs(float(med_line)-float(actual_val))
            if np.isfinite(med_line) and np.isfinite(actual_val) else np.nan
        )
        rec["model_closer_than_line"]=(
            bool(float(r["abs_error"]) < rec["median_line_abs_error"])
            if np.isfinite(rec["median_line_abs_error"]) else False
        )
        rows.append(rec)

    out=pd.DataFrame(rows)
    out["case_rank"]=np.arange(1,len(out)+1)

    counts=(
        out.groupby(["position","market","component_label"],dropna=False)
        .size().reset_index(name="rows")
        .sort_values(["position","market","rows"],ascending=[True,True,False])
    )

    watson=out.loc[
        out["player"].astype(str).str.casefold().eq("deshaun watson")
        & out["market"].eq("pass_yards")
    ]

    summary={
        "version":VERSION,
        "season":SEASON,
        "week":WEEK,
        "top_n":TOP_N,
        "case_rows":int(len(out)),
        "component_label_counts":{
            str(k):int(v) for k,v in out["component_label"].value_counts(dropna=False).items()
        },
        "watson_case":watson[[
            "player","team","market","model_proj","actual","abs_error",
            "qb_pred_attempts","pass_att","qb_pred_ypa",
            "opportunity_contribution","efficiency_contribution","component_label",
        ]].to_dict("records"),
        "nflverse_week5_rows":int(len(actual)),
        "sportsbook_refetch":False,
        "model_fitting":False,
        "production_changed":False,
    }

    out.to_csv(a.out_dir/"week5_player_failure_casebook_v1_rows.csv",index=False,float_format="%.12f")
    counts.to_csv(a.out_dir/"week5_player_failure_casebook_v1_counts.csv",index=False)
    (a.out_dir/"week5_player_failure_casebook_v1_result.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
