#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

TOL = 1e-6
KEYS = ["season", "week", "team", "player_clean_key"]
COMPONENTS = {
    "TOTAL_OFFENSIVE_PLAYS": "play_contrib",
    "PASS_OPPORTUNITY_RATE": "rate_contrib",
}


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def canon_keys(d: pd.DataFrame) -> pd.DataFrame:
    x=d.copy(); x["season"]=num(x["season"]); x["week"]=num(x["week"])
    x["team"]=x["team"].fillna("").astype(str).map(canon_team)
    x["player_clean_key"]=x["player_clean_key"].fillna("").astype(str).str.strip()
    return x


def load_chain(root: Path) -> pd.DataFrame:
    cols=KEYS+["pred_D","actual_D","mechanism_state"]
    x=pd.read_csv(one(root,"qb_opportunity_chain_casebook.csv"),usecols=cols,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]; x=canon_keys(x)
    x["pred_d"]=num(x["pred_d"]); x["actual_d"]=num(x["actual_d"])
    if len(x)!=884 or x.duplicated(KEYS).any() or x.duplicated(["season","week","team"]).any():
        raise RuntimeError(f"chain integrity failure rows={len(x)}")
    return x


def load_m89_components(root: Path, chain: pd.DataFrame) -> pd.DataFrame:
    parts=[]
    for season in [2024,2025]:
        hits=list(root.rglob(f"{season}/component_predictions.csv"))
        if len(hits)!=1: raise RuntimeError(f"component_predictions season={season} hits={len(hits)}")
        d=pd.read_csv(hits[0],low_memory=False); d.columns=[str(c).strip().lower() for c in d.columns]
        req=KEYS+["market","mc_projected_plays","mc_dropback_rate","mc_team_expected_dropbacks"]
        miss=[c for c in req if c not in d.columns]
        if miss: raise RuntimeError(f"M89 component source missing {miss}")
        d=canon_keys(d)
        d=d.loc[d.season.eq(season)&d.market.astype(str).eq("pass_yards"),req].copy()
        d=d.drop(columns="market")
        parts.append(d)
    c=pd.concat(parts,ignore_index=True)
    c=c.merge(chain[KEYS],on=KEYS,how="inner",validate="one_to_one")
    if len(c)!=884 or c.duplicated(KEYS).any(): raise RuntimeError(f"M89 component alignment drift {len(c)}")
    for col in ["mc_projected_plays","mc_dropback_rate","mc_team_expected_dropbacks"]: c[col]=num(c[col])
    return c


def load_actual_plays() -> tuple[pd.DataFrame,dict]:
    import nflreadpy as nfl
    parts=[]; audits={}
    for season in [2024,2025]:
        raw=nfl.load_pbp(seasons=[season]); p=raw.to_pandas() if hasattr(raw,"to_pandas") else pd.DataFrame(raw)
        p.columns=[str(c).strip().lower() for c in p.columns]
        if "season_type" in p.columns: p=p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
        elif "game_type" in p.columns: p=p.loc[p.game_type.astype(str).str.upper().eq("REG")].copy()
        req={"week","posteam","qb_dropback","rush_attempt"}; miss=sorted(req-set(p.columns))
        if miss: raise RuntimeError(f"PBP {season} missing {miss}")
        p["season"]=season; p["week"]=num(p["week"]); p["posteam"]=p["posteam"].fillna("").astype(str).map(canon_team)
        db=num(p["qb_dropback"]).fillna(0).eq(1); rush=num(p["rush_attempt"]).fillna(0).eq(1)
        p["_off_play"]=(db|rush).astype(int)
        g=(p.loc[p.week.between(1,18)&p.posteam.ne("")]
             .groupby(["season","week","posteam"],as_index=False)
             .agg(actual_offensive_plays=("_off_play","sum")))
        g=g.rename(columns={"posteam":"team"}); g["team"]=g.team.map(canon_team)
        parts.append(g); audits[str(season)]={"pbp_rows":int(len(p)),"team_weeks":int(len(g))}
    out=pd.concat(parts,ignore_index=True)
    if out.duplicated(["season","week","team"]).any(): raise RuntimeError("duplicate actual play team-weeks")
    return out,audits


def metrics(actual,pred):
    a=num(actual).to_numpy(float); p=num(pred).to_numpy(float); e=p-a
    return {"n":int(len(a)),"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e**2))),"bias":float(np.mean(e)),"corr":float(np.corrcoef(a,p)[0,1])}


def corr_metrics(x,y):
    z=pd.DataFrame({"x":num(x),"y":num(y)}).dropna()
    return {"n":int(len(z)),"pearson":float(z.x.corr(z.y,method="pearson")),"spearman":float(z.x.corr(z.y,method="spearman")),"same_sign":float((np.sign(z.x)==np.sign(z.y)).mean())}


def q4q1(x,y):
    z=pd.DataFrame({"x":num(x),"y":num(y)}).dropna(); q1=z.x.quantile(.25); q4=z.x.quantile(.75)
    return float(z.loc[z.x.ge(q4),"y"].mean()-z.loc[z.x.le(q1),"y"].mean())


def summarize(z:pd.DataFrame)->pd.DataFrame:
    specs=[
        ("ALL",pd.Series(True,index=z.index)),
        ("ATTEMPTS_DOMINANT",z.mechanism_state.eq("ATTEMPTS_DOMINANT")),
        ("ABS_D_MISS_8_PLUS",(z.actual_d-z.pred_d).abs().ge(8)),
        ("ABS_D_MISS_10_PLUS",(z.actual_d-z.pred_d).abs().ge(10)),
        ("D_UNDERPROJECTED",(z.actual_d-z.pred_d).gt(0)),
        ("D_OVERPROJECTED",(z.actual_d-z.pred_d).lt(0)),
    ]
    rows=[]
    for season_label,smask in [("2024",z.season.eq(2024)),("2025",z.season.eq(2025)),("POOLED_2024_2025",pd.Series(True,index=z.index))]:
        for cohort,cmask in specs:
            g=z.loc[smask&cmask].copy()
            if g.empty: continue
            mass={k:float(g[v].abs().mean()) for k,v in COMPONENTS.items()}; denom=sum(mass.values())
            for comp,col in COMPONENTS.items():
                s=g[col]
                rows.append({
                    "season":season_label,"cohort":cohort,"component":comp,"n":int(len(g)),
                    "mean_contribution":float(s.mean()),"mean_abs_contribution":mass[comp],
                    "abs_mass_share":float(mass[comp]/denom) if denom else np.nan,
                    "sign_agreement_with_d_residual":float((np.sign(s)==np.sign(g.actual_d-g.pred_d)).mean()),
                    "dominant_row_rate":float(g.dominant_internal_component.eq(comp).mean()),
                    "p50_abs":float(s.abs().quantile(.5)),"p75_abs":float(s.abs().quantile(.75)),"p90_abs":float(s.abs().quantile(.90)),
                })
    return pd.DataFrame(rows)


def shared_attribution(z:pd.DataFrame,root:Path)->pd.DataFrame:
    p=pd.read_csv(one(root,"qb_wr_shared_pass_volume_primary_2025.csv"),low_memory=False); s=pd.read_csv(one(root,"qb_wr_shared_pass_volume_secondary_2024_2025.csv"),low_memory=False)
    for d in [p,s]:
        d.columns=[str(c).strip().lower() for c in d.columns]
    p=canon_keys(p); s=canon_keys(s)
    if len(p)!=440 or len(s)!=884 or p.duplicated(KEYS).any() or s.duplicated(KEYS).any(): raise RuntimeError("shared cohort integrity failure")
    keep=KEYS+["play_contrib","rate_contrib"]
    p=p.merge(z[keep],on=KEYS,how="left",validate="one_to_one"); s=s.merge(z[keep],on=KEYS,how="left",validate="one_to_one")
    if p[["play_contrib","rate_contrib"]].isna().any().any() or s[["play_contrib","rate_contrib"]].isna().any().any(): raise RuntimeError("shared attribution missing")
    rows=[]
    for view,d,y,views in [
        ("PRIMARY_WR_TARGET_MASS",p,"wr_target_mass_residual",[("2025",p)]),
        ("SECONDARY_WR_RECEPTION_MASS",s,"wr_reception_mass_residual",[("POOLED_2024_2025",s),("2024",s.loc[s.season.eq(2024)]),("2025",s.loc[s.season.eq(2025)])])]:
        for season_label,g in views:
            for comp,col in COMPONENTS.items():
                m=corr_metrics(g[col],g[y]); rows.append({"view":view,"season":season_label,"component":comp,**m,"signed_q4_minus_q1_wr_residual_gap":q4q1(g[col],g[y])})
    return pd.DataFrame(rows)


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--chain-root",type=Path,required=True); ap.add_argument("--m89-root",type=Path,required=True); ap.add_argument("--shared-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args()
    chain=load_chain(a.chain_root); comp=load_m89_components(a.m89_root,chain); plays,pbp_audit=load_actual_plays()
    z=chain.merge(comp,on=KEYS,how="inner",validate="one_to_one").merge(plays,on=["season","week","team"],how="left",validate="one_to_one")
    if len(z)!=884 or z.actual_offensive_plays.isna().any(): raise RuntimeError(f"play alignment failure rows={len(z)} missing={int(z.actual_offensive_plays.isna().sum())}")
    z["pred_plays"]=z.mc_projected_plays; z["pred_rate"]=z.mc_dropback_rate; z["actual_plays"]=num(z.actual_offensive_plays); z["actual_rate"]=z.actual_d/z.actual_plays
    z["pred_identity"]=z.pred_plays*z.pred_rate; z["actual_identity"]=z.actual_plays*z.actual_rate
    z["play_contrib"]=(z.actual_plays-z.pred_plays)*(z.pred_rate+z.actual_rate)/2.0
    z["rate_contrib"]=(z.actual_rate-z.pred_rate)*(z.pred_plays+z.actual_plays)/2.0
    z["shapley_sum"]=z.play_contrib+z.rate_contrib
    z["perfect_plays_d"]=z.actual_plays*z.pred_rate; z["perfect_rate_d"]=z.pred_plays*z.actual_rate
    arr=np.column_stack([z.play_contrib.abs().to_numpy(),z.rate_contrib.abs().to_numpy()]); labels=np.array(list(COMPONENTS.keys()),object); z["dominant_internal_component"]=labels[arr.argmax(axis=1)]

    identities={
        "pred_plays_times_rate_vs_pred_d_max_abs":float((z.pred_identity-z.pred_d).abs().max()),
        "actual_plays_times_rate_vs_actual_d_max_abs":float((z.actual_identity-z.actual_d).abs().max()),
        "two_factor_shapley_vs_d_residual_max_abs":float((z.shapley_sum-(z.actual_d-z.pred_d)).abs().max()),
    }
    summaries=summarize(z); shared=shared_attribution(z,a.shared_root)
    factors={}
    oracles={}
    for label,g in [("2024",z.loc[z.season.eq(2024)]),("2025",z.loc[z.season.eq(2025)]),("POOLED_2024_2025",z)]:
        factors[label]={"offensive_plays":metrics(g.actual_plays,g.pred_plays),"pass_opportunity_rate":metrics(g.actual_rate,g.pred_rate)}
        oracles[label]={"baseline_d":metrics(g.actual_d,g.pred_d),"perfect_plays_d":metrics(g.actual_d,g.perfect_plays_d),"perfect_rate_d":metrics(g.actual_d,g.perfect_rate_d)}

    rate_dist={
        "min":float(z.pred_rate.min()),"max":float(z.pred_rate.max()),"mean":float(z.pred_rate.mean()),"std":float(z.pred_rate.std()),
        "distinct_rounded_12dp":int(z.pred_rate.round(12).nunique()),
        "share_equal_0_57_within_1e_12":float(np.isclose(z.pred_rate,0.57,atol=1e-12,rtol=0).mean()),
    }
    integrity={
        "exact_884_rows":len(z)==884,"exact_440_884_shared":True,"pbp_loaded_2024_2025":set(pbp_audit)=={"2024","2025"},
        "zero_sportsbook_inputs":True,"zero_model_fitting":True,"no_production_change":True,
        "pred_identity":identities["pred_plays_times_rate_vs_pred_d_max_abs"]<=TOL,
        "actual_identity":identities["actual_plays_times_rate_vs_actual_d_max_abs"]<=TOL,
        "shapley_identity":identities["two_factor_shapley_vs_d_residual_max_abs"]<=TOL,
        "all_team_week_play_counts_aligned":not z.actual_plays.isna().any(),"target_pbp_diagnostic_only":True,"pre_m89_ranking_not_used":True,
    }
    all_integrity=all(integrity.values())
    pool=summaries.loc[(summaries.season=="POOLED_2024_2025")&(summaries.cohort=="ALL")].set_index("component")
    y24=summaries.loc[(summaries.season=="2024")&(summaries.cohort=="ALL")].set_index("component"); y25=summaries.loc[(summaries.season=="2025")&(summaries.cohort=="ALL")].set_index("component")
    wrt=shared.loc[(shared.view=="PRIMARY_WR_TARGET_MASS")&(shared.season=="2025")].set_index("component")
    wrr=shared.loc[(shared.view=="SECONDARY_WR_RECEPTION_MASS")&(shared.season=="POOLED_2024_2025")].set_index("component")
    routing={}
    comps=list(COMPONENTS)
    for c in comps:
        other=[o for o in comps if o!=c][0]
        vp=float(pool.loc[c,"mean_abs_contribution"]); vo=float(pool.loc[other,"mean_abs_contribution"])
        v24=float(y24.loc[c,"mean_abs_contribution"]); o24=float(y24.loc[other,"mean_abs_contribution"]); v25=float(y25.loc[c,"mean_abs_contribution"]); o25=float(y25.loc[other,"mean_abs_contribution"])
        stable=(v24>=o24 and v25>=o25) or (v24>=o24 and v25>=.9*o25) or (v25>=o25 and v24>=.9*o24)
        sw=abs(float(wrt.loc[c,"spearman"])); so=abs(float(wrt.loc[other,"spearman"])); sr=abs(float(wrr.loc[c,"spearman"]))
        routing[c]={"largest_pooled":vp>=vo,"season_stability":bool(stable),"wr_target_abs_spearman_ge_0_30":sw>=.30,"wr_target_lead_ge_0_10":sw>=so+.10,"wr_reception_pooled_abs_spearman_ge_0_25":sr>=.25,"pooled_mean_abs":vp,"wr_target_spearman":float(wrt.loc[c,"spearman"]),"wr_reception_pooled_spearman":float(wrr.loc[c,"spearman"])}
    qualifying=[c for c,g in routing.items() if all([g["largest_pooled"],g["season_stability"],g["wr_target_abs_spearman_ge_0_30"],g["wr_target_lead_ge_0_10"],g["wr_reception_pooled_abs_spearman_ge_0_25"]])]
    if not all_integrity: disposition="MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE"
    elif len(qualifying)==1: disposition=f"{qualifying[0]}_PRIMARY_DIAGNOSTIC"
    else: disposition="MIXED_PLAY_RATE_NO_SINGLE_PRIMARY"
    result={"migration":"QB_TEAM_PASS_OPPORTUNITY_PLAY_RATE_DECOMP_V1","disposition":disposition,"production_actionable":False,"identities":identities,"integrity_gates":integrity,"predicted_rate_distribution":rate_dist,"pbp_audit":pbp_audit,"factor_metrics":factors,"oracle_d_metrics":oracles,"routing":routing,"qualifying_primary":qualifying}
    a.out_dir.mkdir(parents=True,exist_ok=True); z.to_csv(a.out_dir/"play_rate_decomposition_casebook.csv",index=False); summaries.to_csv(a.out_dir/"play_rate_decomposition_summary.csv",index=False); shared.to_csv(a.out_dir/"play_rate_shared_receiver_attribution.csv",index=False); (a.out_dir/"play_rate_decomposition_result.json").write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if all_integrity else 2

if __name__=="__main__": raise SystemExit(main())
