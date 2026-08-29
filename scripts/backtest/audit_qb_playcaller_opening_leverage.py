#!/usr/bin/env python3
"""Migration 68: playcaller + opening-script + leverage new-information audit.

M68 is diagnostic only. Fixed models fit 2024 and test 2025. Any combined
existing+new claim must clear the Raw/base gate AND an incremental existing-only
control gate so old information cannot be misattributed to M68.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import scripts.backtest.audit_qb_research_frontier as m66
from scripts._opponent_map import canon_team

INVALID_EXISTING={"coach_change","coach_tenure_games"}


def num(v):
    return pd.to_numeric(v,errors="coerce")


def lower(x: pd.DataFrame) -> pd.DataFrame:
    y=x.copy()
    y.columns=[str(c).strip().lower() for c in y.columns]
    return y


def read(path: Path) -> pd.DataFrame:
    return lower(pd.read_csv(path))


def canon(v) -> str:
    t=canon_team(v)
    return "WAS" if t=="WSH" else t


def metrics(actual,pred,miss_threshold=None) -> dict:
    z=pd.DataFrame({"a":num(actual),"p":num(pred)}).dropna()
    if z.empty:
        return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":np.nan,"miss":0}
    e=z.p-z.a
    return {
        "n":int(len(z)),
        "mae":float(e.abs().mean()),
        "rmse":float(np.sqrt(np.mean(np.square(e)))),
        "bias":float(e.mean()),
        "corr":float(z.a.corr(z.p)) if len(z)>1 else np.nan,
        "miss":int(e.abs().ge(miss_threshold).sum()) if miss_threshold is not None else 0,
    }


def model_specs(target: str):
    return {
        f"ridge_{target}":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=50.0)),
        f"histgb_{target}":make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="absolute_error",max_iter=150,learning_rate=.04,max_depth=2,
                min_samples_leaf=15,l2_regularization=5.0,random_state=68,
            ),
        ),
    }


def usable_cols(x: pd.DataFrame,cols:list[str]) -> list[str]:
    train=x[num(x.season).eq(2024)]
    out=[]
    for c in cols:
        if c not in x:
            continue
        x[c]=num(x[c])
        if train[c].notna().sum()>=100 and train[c].nunique(dropna=True)>1:
            out.append(c)
    return out


def prepare(g:pd.DataFrame,sf:pd.DataFrame,nf:pd.DataFrame):
    base,existing=m66.merge_feature_universe(g.copy(),sf.copy())
    for q in (base,nf):
        q["team"]=q.team.map(canon)
    existing=[c for c in existing if c not in INVALID_EXISTING]
    new_cols=[c for c in nf if c not in {"season","week","team","opponent","playcaller_current_name"}]
    x=base.merge(nf[["season","week","team"]+new_cols],on=["season","week","team"],how="left",validate="many_to_one")
    if len(x)!=len(base):
        raise RuntimeError("M68 feature merge changed canonical population")
    opening=usable_cols(x,[c for c in new_cols if c.startswith("opening_")])
    caller=usable_cols(x,[c for c in new_cols if c.startswith("playcaller_")])
    leverage=usable_cols(x,[c for c in new_cols if c.startswith("leverage_")])
    existing=usable_cols(x,existing)
    families={
        "opening_script_live":opening,
        "verified_playcaller_live":caller,
        "playoff_leverage_live":leverage,
        "playcaller_plus_opening_live":list(dict.fromkeys(caller+opening)),
        "all_m68_new_live":list(dict.fromkeys(caller+opening+leverage)),
        "existing_only_control":existing,
        "existing_plus_m68_new":list(dict.fromkeys(existing+caller+opening+leverage)),
    }
    for name,cols in families.items():
        if not cols:
            raise RuntimeError(f"M68 family {name} has zero usable features")
    return x,families,existing


def family_coverage(x:pd.DataFrame,cols:list[str]) -> float:
    new=[c for c in cols if c.startswith(("opening_","playcaller_","leverage_"))]
    if not new:
        return 1.0
    return float(np.median([num(x[c]).notna().mean() for c in new]))


def fit_family_target(x,families,target_kind:str) -> pd.DataFrame:
    train=x[num(x.season).eq(2024)].copy()
    test=x[num(x.season).eq(2025)].copy()

    if target_kind=="pass":
        actual_col="actual"
        base_col="m64_pass_raw_reference" if "m64_pass_raw_reference" in x else "mc_proj_attempts_raw_only"
        threshold=100
        clip=None
    elif target_kind=="attempt":
        actual_col="actual_pass_att"; base_col="attempts_raw"; threshold=10; clip=(15.0,55.0)
    elif target_kind=="dbr":
        actual_col="m64_actual_dropback_rate"; base_col="m64_pred_dropback_rate_neutral"; threshold=None; clip=(.25,.90)
    else:
        raise ValueError(target_kind)

    train["_resid"]=num(train[actual_col])-num(train[base_col])
    test["_resid"]=num(test[actual_col])-num(test[base_col])
    base=metrics(test[actual_col],test[base_col],threshold)
    rows=[]
    for family,cols in families.items():
        cov=family_coverage(test,cols)
        for model_name,model in model_specs(f"{target_kind}_residual").items():
            model.fit(train[cols],train._resid)
            phat=model.predict(test[cols])
            corrected=num(test[base_col]).to_numpy(float)+phat
            if clip:
                corrected=np.clip(corrected,*clip)
            mm=metrics(test[actual_col],corrected,threshold)
            rc=float(pd.Series(test._resid.to_numpy(float)).corr(pd.Series(phat)))
            rows.append({
                "target":target_kind,"family":family,"model":model_name,
                "feature_count":len(cols),"new_feature_median_coverage":cov,
                "residual_corr":rc,"corrected_mae":mm["mae"],"corrected_rmse":mm["rmse"],
                "corrected_bias":mm["bias"],"corrected_corr":mm["corr"],"corrected_misses":mm["miss"],
                "base_mae":base["mae"],"base_rmse":base["rmse"],"base_bias":base["bias"],
                "base_corr":base["corr"],"base_misses":base["miss"],
                "mae_gain_vs_base":base["mae"]-mm["mae"],
                "corr_gain_vs_base":mm["corr"]-base["corr"],
            })
    return pd.DataFrame(rows)


def raw_gate(r: pd.Series) -> bool:
    if float(r.new_feature_median_coverage)<.75:
        return False
    if r.target=="pass":
        return bool(r.residual_corr>=.20 and r.mae_gain_vs_base>=1.0 and r.corr_gain_vs_base>=.03 and r.corrected_misses<=r.base_misses)
    if r.target=="dbr":
        return bool(r.residual_corr>=.20 and r.mae_gain_vs_base>=.0075 and r.corr_gain_vs_base>=.10)
    if r.target=="attempt":
        return bool(r.residual_corr>=.20 and r.mae_gain_vs_base>=.25 and r.corr_gain_vs_base>=.05 and r.corrected_misses<=np.floor(r.base_misses*.95))
    return False


def add_control_attribution(allm: pd.DataFrame) -> pd.DataFrame:
    z=allm.copy()
    z["raw_gate"]=z.apply(raw_gate,axis=1)
    z["control_mae"]=np.nan
    z["control_corr"]=np.nan
    z["control_misses"]=np.nan
    z["mae_gain_vs_existing_control"]=np.nan
    z["corr_gain_vs_existing_control"]=np.nan
    z["miss_gain_vs_existing_control"]=np.nan
    z["incremental_control_gate"]=False
    for i,r in z[z.family.eq("existing_plus_m68_new")].iterrows():
        c=z[(z.family.eq("existing_only_control"))&(z.target.eq(r.target))&(z.model.eq(r.model))]
        if c.empty:
            continue
        c=c.iloc[0]
        z.loc[i,"control_mae"]=c.corrected_mae
        z.loc[i,"control_corr"]=c.corrected_corr
        z.loc[i,"control_misses"]=c.corrected_misses
        z.loc[i,"mae_gain_vs_existing_control"]=c.corrected_mae-r.corrected_mae
        z.loc[i,"corr_gain_vs_existing_control"]=r.corrected_corr-c.corrected_corr
        z.loc[i,"miss_gain_vs_existing_control"]=c.corrected_misses-r.corrected_misses
        if r.target=="pass":
            inc=(c.corrected_mae-r.corrected_mae>=.50 and r.corrected_corr-c.corrected_corr>=.02 and r.corrected_misses<=c.corrected_misses)
        elif r.target=="dbr":
            inc=(c.corrected_mae-r.corrected_mae>=.0025 and r.corrected_corr-c.corrected_corr>=.03)
        else:
            inc=(c.corrected_mae-r.corrected_mae>=.10 and r.corrected_corr-c.corrected_corr>=.03 and r.corrected_misses<=c.corrected_misses)
        z.loc[i,"incremental_control_gate"]=bool(inc)
    standalone=~z.family.isin(["existing_only_control","existing_plus_m68_new"])
    z["new_information_eligible"]=False
    z.loc[standalone,"new_information_eligible"]=z.loc[standalone,"raw_gate"]
    combo=z.family.eq("existing_plus_m68_new")
    z.loc[combo,"new_information_eligible"]=z.loc[combo,"raw_gate"] & z.loc[combo,"incremental_control_gate"]
    return z


def univariate_screen(x:pd.DataFrame,new_cols:list[str]) -> pd.DataFrame:
    raw_col="m64_pass_raw_reference" if "m64_pass_raw_reference" in x else "mc_proj_attempts_raw_only"
    ys={
        "pass_residual":num(x.actual)-num(x[raw_col]),
        "attempt_residual":num(x.actual_pass_att)-num(x.attempts_raw),
        "dbr_residual":num(x.m64_actual_dropback_rate)-num(x.m64_pred_dropback_rate_neutral),
    }
    rows=[]
    for c in new_cols:
        s=num(x[c])
        family="opening_script" if c.startswith("opening_") else "verified_playcaller" if c.startswith("playcaller_") else "playoff_leverage"
        for target,y in ys.items():
            vals={}
            for season in (2024,2025):
                m=num(x.season).eq(season)&s.notna()&y.notna()
                vals[season]=float(s[m].corr(y[m])) if m.sum()>=20 else np.nan
            m=s.notna()&y.notna()
            comb=float(s[m].corr(y[m])) if m.sum()>=40 else np.nan
            same=np.isfinite(vals[2024]) and np.isfinite(vals[2025]) and vals[2024]*vals[2025]>0
            strong=bool(same and abs(vals[2024])>=.10 and abs(vals[2025])>=.10 and abs(comb)>=.15)
            rows.append({"family":family,"feature":c,"target":target,"corr_2024":vals[2024],
                         "corr_2025":vals[2025],"corr_combined":comb,"strong_replicated":strong})
    return pd.DataFrame(rows)


def interpretation(scored:pd.DataFrame,uni:pd.DataFrame) -> pd.DataFrame:
    eligible=scored[scored.new_information_eligible]
    standalone=eligible[~eligible.family.eq("existing_plus_m68_new")]
    combo=eligible[eligible.family.eq("existing_plus_m68_new")]
    strong=uni[uni.strong_replicated]
    opening_caller=strong[strong.family.isin(["opening_script","verified_playcaller"])]
    leverage=strong[strong.family.eq("playoff_leverage")]
    if len(standalone) or len(combo):
        verdict="m68_new_information_breakthrough_followup"
    elif len(leverage):
        verdict="m68_leverage_partial_signal"
    elif len(opening_caller):
        verdict="m68_opening_playcaller_partial_signal"
    else:
        verdict="seek_deeper_week_specific_information_or_randomness_transition"
    return pd.DataFrame([{
        "new_information_actionable":bool(len(eligible)),
        "actionable_family_models":"|".join(f"{r.family}:{r.model}:{r.target}" for r in eligible.itertuples(index=False)),
        "standalone_actionable_n":int(len(standalone)),
        "existing_plus_new_incremental_actionable_n":int(len(combo)),
        "strong_replicated_new_feature_target_pairs":int(len(strong)),
        "strong_opening_playcaller_pairs":int(len(opening_caller)),
        "strong_leverage_pairs":int(len(leverage)),
        "m68_interpretation":verdict,
    }])


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--m65-game-level",type=Path,required=True)
    p.add_argument("--m65-state-features",type=Path,required=True)
    p.add_argument("--new-features",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    g,sf,nf=read(args.m65_game_level),read(args.m65_state_features),read(args.new_features)
    x,families,existing=prepare(g,sf,nf)
    frames=[fit_family_target(x,families,k) for k in ("pass","dbr","attempt")]
    allm=add_control_attribution(pd.concat(frames,ignore_index=True))
    new_cols=list(dict.fromkeys([c for name,cols in families.items() if name not in {"existing_only_control","existing_plus_m68_new"} for c in cols]))
    uni=univariate_screen(x,new_cols)
    interp=interpretation(allm,uni)

    allm.to_csv(args.out_dir/"m68_family_target_metrics.csv",index=False)
    allm[allm.target.eq("pass")].to_csv(args.out_dir/"m68_pass_metrics.csv",index=False)
    allm[allm.target.eq("dbr")].to_csv(args.out_dir/"m68_dropback_metrics.csv",index=False)
    allm[allm.target.eq("attempt")].to_csv(args.out_dir/"m68_attempt_metrics.csv",index=False)
    uni.to_csv(args.out_dir/"m68_univariate_new_information_screen.csv",index=False)
    interp.to_csv(args.out_dir/"m68_precommitted_interpretation.csv",index=False)
    pd.DataFrame([{"family":k,"features":"|".join(v),"feature_count":len(v)} for k,v in families.items()]).to_csv(args.out_dir/"m68_feature_families.csv",index=False)

    print("=== M68 PRECOMMITTED INTERPRETATION ===")
    print(interp.to_string(index=False))
    print("=== M68 ELIGIBLE RESULTS ===")
    print(allm[allm.new_information_eligible].to_string(index=False) if allm.new_information_eligible.any() else "none")
    print("=== M68 BEST BY TARGET (MAE) ===")
    print(allm.sort_values(["target","corrected_mae"]).groupby("target").head(5).to_string(index=False))
    print("=== M68 STRONG REPLICATED NEW FEATURES ===")
    print(uni[uni.strong_replicated].to_string(index=False) if uni.strong_replicated.any() else "none")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
