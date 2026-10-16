#!/usr/bin/env python3
"""Score frozen WR targeted-route full-stack integration V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS=["season","week","team","player_clean_key","market"]


def read(p: Path, label: str) -> pd.DataFrame:
    if not p.exists() or not p.stat().st_size:
        raise RuntimeError(f"missing {label}: {p}")
    x=pd.read_csv(p,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def score(actual, pred) -> dict:
    z=pd.DataFrame({"a":pd.to_numeric(actual,errors="coerce"),"p":pd.to_numeric(pred,errors="coerce")}).dropna()
    if z.empty:
        return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"correlation":np.nan}
    e=z.p-z.a
    corr=float(z.p.corr(z.a)) if len(z)>1 and z.p.nunique()>1 and z.a.nunique()>1 else np.nan
    return {"n":int(len(z)),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":corr}


def coverage(df: pd.DataFrame,prefix: str,level: int)->float:
    lo,hi=f"{prefix}_lo{level}",f"{prefix}_hi{level}"
    q=df.loc[df[lo].notna()&df[hi].notna()].copy()
    return float(((q["actual"]>=q[lo])&(q["actual"]<=q[hi])).mean()) if len(q) else np.nan


def miss_rate(actual,pred,threshold)->float:
    a=pd.to_numeric(actual,errors="coerce");p=pd.to_numeric(pred,errors="coerce")
    m=a.notna()&p.notna()
    return float((p[m].sub(a[m]).abs()>=threshold).mean()) if m.any() else np.nan


def trace_pair(base: pd.DataFrame,cand: pd.DataFrame,eligible: pd.DataFrame,market: str)->pd.DataFrame:
    need=KEYS+["ensemble_proj","actual"]
    for label,x in (("baseline",base),("candidate",cand)):
        miss=sorted(set(need)-set(x.columns))
        if miss: raise RuntimeError(f"{label} trace missing {miss}")
    b=base.loc[base["market"].astype(str).eq(market),need].copy().rename(columns={"ensemble_proj":"baseline_full","actual":"actual_base"})
    c=cand.loc[cand["market"].astype(str).eq(market),need].copy().rename(columns={"ensemble_proj":"candidate_full","actual":"actual_candidate"})
    e=eligible.loc[eligible["market"].astype(str).eq(market),["season","week","team","player_clean_key","market","wr1_anchor","actual"]].copy()
    if e.duplicated(KEYS).any(): raise RuntimeError(f"eligible {market} identities duplicate")
    z=e.merge(b,on=KEYS,how="left",validate="one_to_one").merge(c,on=KEYS,how="left",validate="one_to_one")
    if z[["baseline_full","candidate_full"]].isna().any().any():
        raise RuntimeError(f"full-stack trace missing eligible {market} rows")
    # direct actual is source authority; assert trace actual parity where finite.
    for col in ("actual_base","actual_candidate"):
        q=pd.to_numeric(z[col],errors="coerce")
        m=q.notna()&pd.to_numeric(z["actual"],errors="coerce").notna()
        if m.any() and float((q[m]-pd.to_numeric(z.loc[m,"actual"],errors="raise")).abs().max())>1e-10:
            raise RuntimeError(f"actual parity failure {market} {col}")
    return z


def player_diag(z: pd.DataFrame)->dict:
    rows=[]
    for pid,g in z.groupby("player_clean_key"):
        if len(g)<4: continue
        b=float((g["baseline_full"]-g["actual"]).abs().mean())
        c=float((g["candidate_full"]-g["actual"]).abs().mean())
        rows.append((pid,len(g),b,c,c-b))
    p=pd.DataFrame(rows,columns=["player_clean_key","games","baseline_mae","candidate_mae","delta"])
    return {
        "qualifying_players_ge4":int(len(p)),
        "median_player_baseline_mae":float(p["baseline_mae"].median()) if len(p) else np.nan,
        "median_player_candidate_mae":float(p["candidate_mae"].median()) if len(p) else np.nan,
        "median_player_mae_delta":float(p["delta"].median()) if len(p) else np.nan,
        "fraction_players_improved":float((p["delta"]<0).mean()) if len(p) else np.nan,
        "rows":p,
    }


def arm_result(direct: pd.DataFrame,z: pd.DataFrame,market: str)->dict:
    ds_base=score(direct["actual"],direct["baseline_mean"])
    ds_cand=score(direct["actual"],direct["candidate_mean"])
    fs_base=score(z["actual"],z["baseline_full"])
    fs_cand=score(z["actual"],z["candidate_full"])
    phase={}
    for name,mask in {
        "weeks_5_12":z["week"].between(5,12),
        "weeks_13_18":z["week"].between(13,18),
        "wr1":z["wr1_anchor"].astype(bool),
        "wr2plus":~z["wr1_anchor"].astype(bool),
    }.items():
        q=z.loc[mask]
        phase[name]={"baseline":score(q["actual"],q["baseline_full"]),"candidate":score(q["actual"],q["candidate_full"])}
    pdg=player_diag(z)
    crps_base=float(pd.to_numeric(direct["baseline_crps"],errors="coerce").mean())
    crps_cand=float(pd.to_numeric(direct["candidate_crps"],errors="coerce").mean())
    cov={
        "baseline_80":coverage(direct,"baseline",80),"candidate_80":coverage(direct,"candidate",80),
        "baseline_90":coverage(direct,"baseline",90),"candidate_90":coverage(direct,"candidate",90),
    }
    if market=="rec_yards":
        tails={f"miss{t}":{"baseline":miss_rate(z["actual"],z["baseline_full"],t),"candidate":miss_rate(z["actual"],z["candidate_full"],t)} for t in (20,30,40)}
    else:
        tails={f"miss{t}":{"baseline":miss_rate(z["actual"],z["baseline_full"],t),"candidate":miss_rate(z["actual"],z["candidate_full"],t)} for t in (2,3)}
    return {
        "market":market,"eligible_player_games":int(len(z)),"named_players":int(z["player_clean_key"].nunique()),
        "direct_mc":{"baseline":ds_base,"candidate":ds_cand,
            "mae_improvement_fraction":float((ds_base["mae"]-ds_cand["mae"])/ds_base["mae"]) if ds_base["mae"] else np.nan},
        "full_stack":{"baseline":fs_base,"candidate":fs_cand,
            "mae_improvement_fraction":float((fs_base["mae"]-fs_cand["mae"])/fs_base["mae"]) if fs_base["mae"] else np.nan},
        "phase_role":phase,"tails":tails,
        "distribution":{"baseline_crps":crps_base,"candidate_crps":crps_cand,**cov},
        "player":{k:v for k,v in pdg.items() if k!="rows"},
        "_player_rows":pdg["rows"],
    }


def gate_arm(r:dict)->tuple[dict,str]:
    m=r["market"]; d=r["direct_mc"]; f=r["full_stack"]; p=r["player"]; pr=r["phase_role"]; dist=r["distribution"]
    gates={
        "coverage_named_players_ge150":r["named_players"]>=150,
        "coverage_player_games_ge1500":r["eligible_player_games"]>=1500,
        "direct_mc_mae_improve_ge_0_5pct":d["mae_improvement_fraction"]>=0.005,
        "direct_mc_rmse_nonworse":d["candidate"]["rmse"]<=d["baseline"]["rmse"]+1e-12,
        "full_stack_mae_improves":f["candidate"]["mae"]<f["baseline"]["mae"],
        "full_stack_rmse_nonworse":f["candidate"]["rmse"]<=f["baseline"]["rmse"]+1e-12,
        "full_stack_abs_bias_nonworse":abs(f["candidate"]["bias"])<=abs(f["baseline"]["bias"])+1e-12,
        "full_stack_corr_nonworse_tol_005":f["candidate"]["correlation"]>=f["baseline"]["correlation"]-0.005,
        "weeks_5_12_mae_nonworse":pr["weeks_5_12"]["candidate"]["mae"]<=pr["weeks_5_12"]["baseline"]["mae"]+1e-12,
        "weeks_13_18_mae_nonworse":pr["weeks_13_18"]["candidate"]["mae"]<=pr["weeks_13_18"]["baseline"]["mae"]+1e-12,
        "wr1_mae_nonworse":pr["wr1"]["candidate"]["mae"]<=pr["wr1"]["baseline"]["mae"]+1e-12,
        "wr2plus_mae_nonworse":pr["wr2plus"]["candidate"]["mae"]<=pr["wr2plus"]["baseline"]["mae"]+1e-12,
        "crps_nonworse":dist["candidate_crps"]<=dist["baseline_crps"]+1e-12,
        "coverage80_gap_nonworse":abs(dist["candidate_80"]-0.80)<=abs(dist["baseline_80"]-0.80)+1e-12,
        "coverage90_gap_nonworse":abs(dist["candidate_90"]-0.90)<=abs(dist["baseline_90"]-0.90)+1e-12,
        "median_player_delta_negative":p["median_player_mae_delta"]<0,
        "fraction_players_improved_gt_50pct":p["fraction_players_improved"]>0.50,
    }
    for k,v in r["tails"].items():
        gates[f"{k}_nonworse"]=v["candidate"]<=v["baseline"]+1e-12
    ok=all(gates.values())
    if m=="rec_yards":
        disp="WR_ROUTE_MATCHUP_YPT_FULL_STACK_CONFIRMED" if ok else "NO_ACTIONABLE_WR_ROUTE_MATCHUP_YPT_INTEGRATION"
    else:
        disp="WR_ROUTE_MATCHUP_CATCH_FULL_STACK_CONFIRMED" if ok else "NO_ACTIONABLE_WR_ROUTE_MATCHUP_CATCH_INTEGRATION"
    return gates,disp


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-trace",type=Path,required=True)
    ap.add_argument("--ypt-trace",type=Path,required=True)
    ap.add_argument("--catch-trace",type=Path,required=True)
    ap.add_argument("--direct-rows",type=Path,required=True)
    ap.add_argument("--audit",type=Path,required=True)
    ap.add_argument("--betas",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    base=read(a.baseline_trace,"baseline full-stack trace")
    ypt=read(a.ypt_trace,"YPT full-stack trace")
    catch=read(a.catch_trace,"catch full-stack trace")
    direct=read(a.direct_rows,"direct MC rows")
    audit=read(a.audit,"integration audit")
    betas=json.loads(a.betas.read_text())

    integrity={
        "audit_rows":int(len(audit)),
        "audit_all_pass":bool((audit["status"].astype(str)=="PASS").all()),
        "te_pool_max_abs_gap":float(pd.to_numeric(audit["te_pool_max_abs_gap"],errors="coerce").max()),
        "wr1_anchor_max_abs_gap":float(pd.to_numeric(audit["wr1_anchor_max_abs_gap"],errors="coerce").max()),
        "wr2plus_pool_max_abs_gap":float(pd.to_numeric(audit["wr2plus_pool_max_abs_gap"],errors="coerce").max()),
        "wr_room_mass_max_abs_gap":float(pd.to_numeric(audit["wr_room_mass_max_abs_gap"],errors="coerce").max()),
        "team_modeled_mass_max_abs_gap":float(pd.to_numeric(audit["team_modeled_mass_max_abs_gap"],errors="coerce").max()),
        "ypt_target_share_max_abs_delta":float(pd.to_numeric(audit["ypt_target_share_max_abs_delta"],errors="coerce").max()),
        "catch_target_share_max_abs_delta":float(pd.to_numeric(audit["catch_target_share_max_abs_delta"],errors="coerce").max()),
        "ypt_non_wr_max_abs_delta":float(pd.to_numeric(audit["ypt_non_wr_max_abs_delta"],errors="coerce").max()),
        "catch_non_wr_max_abs_delta":float(pd.to_numeric(audit["catch_non_wr_max_abs_delta"],errors="coerce").max()),
    }
    integrity["hard_pass"]=bool(
        integrity["audit_all_pass"]
        and max(integrity[k] for k in (
            "te_pool_max_abs_gap","wr1_anchor_max_abs_gap","wr2plus_pool_max_abs_gap",
            "wr_room_mass_max_abs_gap","team_modeled_mass_max_abs_gap"
        ))<=1e-10
        and max(integrity[k] for k in (
            "ypt_target_share_max_abs_delta","catch_target_share_max_abs_delta",
            "ypt_non_wr_max_abs_delta","catch_non_wr_max_abs_delta"
        ))<=1e-12
    )

    rz=direct.loc[direct["market"].eq("rec_yards")].copy()
    cp=direct.loc[direct["market"].eq("receptions")].copy()
    yz=trace_pair(base,ypt,direct,"rec_yards")
    cz=trace_pair(base,catch,direct,"receptions")
    yr=arm_result(rz,yz,"rec_yards")
    cr=arm_result(cp,cz,"receptions")
    yg,yd=gate_arm(yr); cg,cd=gate_arm(cr)
    if not integrity["hard_pass"]:
        yg["baseline_integrity_pass"]=False; cg["baseline_integrity_pass"]=False
        yd="NO_ACTIONABLE_WR_ROUTE_MATCHUP_YPT_INTEGRATION"
        cd="NO_ACTIONABLE_WR_ROUTE_MATCHUP_CATCH_INTEGRATION"
    else:
        yg["baseline_integrity_pass"]=True; cg["baseline_integrity_pass"]=True
    ypass=yd=="WR_ROUTE_MATCHUP_YPT_FULL_STACK_CONFIRMED"
    cpass=cd=="WR_ROUTE_MATCHUP_CATCH_FULL_STACK_CONFIRMED"
    overall=("WR_ROUTE_MATCHUP_FULL_STACK_BOTH_ARMS_CONFIRMED" if ypass and cpass else
             "WR_ROUTE_MATCHUP_FULL_STACK_PARTIAL_CONFIRMATION" if ypass or cpass else
             "NO_ACTIONABLE_WR_ROUTE_MATCHUP_FULL_STACK_INTEGRATION")

    for r,g,d in ((yr,yg,yd),(cr,cg,cd)):
        r["gates"]=g;r["failed_gates"]=[k for k,v in g.items() if not v];r["disposition"]=d
        r.pop("_player_rows",None)

    result={
        "version":"WR_TARGETED_ROUTE_MATCHUP_FULL_STACK_INTEGRATION_V1",
        "disposition":overall,"integrity":integrity,"betas":betas,
        "ypt_arm":yr,"catch_arm":cr,
        "sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_read":0,"production_changed":False,
    }
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/"wr_route_matchup_full_stack_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    pd.DataFrame([
        {
            "arm":"YPT","disposition":yd,"named_players":yr["named_players"],"player_games":yr["eligible_player_games"],
            "direct_mae_base":yr["direct_mc"]["baseline"]["mae"],"direct_mae_candidate":yr["direct_mc"]["candidate"]["mae"],
            "direct_improvement":yr["direct_mc"]["mae_improvement_fraction"],
            "full_mae_base":yr["full_stack"]["baseline"]["mae"],"full_mae_candidate":yr["full_stack"]["candidate"]["mae"],
            "full_improvement":yr["full_stack"]["mae_improvement_fraction"],
            "median_player_delta":yr["player"]["median_player_mae_delta"],"fraction_players_improved":yr["player"]["fraction_players_improved"],
            "failed_gates":";".join(yr["failed_gates"]),
        },
        {
            "arm":"CATCH","disposition":cd,"named_players":cr["named_players"],"player_games":cr["eligible_player_games"],
            "direct_mae_base":cr["direct_mc"]["baseline"]["mae"],"direct_mae_candidate":cr["direct_mc"]["candidate"]["mae"],
            "direct_improvement":cr["direct_mc"]["mae_improvement_fraction"],
            "full_mae_base":cr["full_stack"]["baseline"]["mae"],"full_mae_candidate":cr["full_stack"]["candidate"]["mae"],
            "full_improvement":cr["full_stack"]["mae_improvement_fraction"],
            "median_player_delta":cr["player"]["median_player_mae_delta"],"fraction_players_improved":cr["player"]["fraction_players_improved"],
            "failed_gates":";".join(cr["failed_gates"]),
        },
    ]).to_csv(a.out_dir/"wr_route_matchup_full_stack_summary.csv",index=False)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
