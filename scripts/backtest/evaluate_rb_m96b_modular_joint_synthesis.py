"""M96B: modular RB workload x efficiency synthesis.

Protocol is frozen in docs/migrations/M96B_PLAN.md.
No feature/weight search. Modules are deliberately isolated:
C=M94C center, E=M95C environment residual, W=M95F workload tail,
X=M95D incremental upside context, V=M95I vacancy-only diagnostic.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ALIASES = {"audricestime": "audricestim"}
KEYS = ["season", "week", "team", "player_join_key"]
TARGETS = {75: "actual_75", 100: "actual_100"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def canon_team(x):
    s = "" if pd.isna(x) else str(x).upper().strip()
    return {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAX": "JAC"}.get(s, s)


def clean_player_key(x):
    s = "" if pd.isna(x) else str(x)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return ALIASES.get(s, s)


def prep(x):
    z = x.copy()
    z.columns = [str(c).lower() for c in z.columns]
    z["season"] = num(z["season"]).astype(int)
    z["week"] = num(z["week"]).astype(int)
    z["team"] = z["team"].map(canon_team)
    if "player_clean_key" not in z.columns:
        raise RuntimeError("missing player_clean_key")
    z["player_join_key"] = z["player_clean_key"].map(clean_player_key)
    return z


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def point_metrics(actual, pred):
    q = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    if q.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan,
                "actual_mean": np.nan, "pred_mean": np.nan}
    err = q["pred"] - q["actual"]
    corr = q["actual"].corr(q["pred"]) if len(q) >= 3 and q["actual"].nunique() > 1 and q["pred"].nunique() > 1 else np.nan
    return {
        "n": int(len(q)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "corr": float(corr) if pd.notna(corr) else np.nan,
        "actual_mean": float(q["actual"].mean()),
        "pred_mean": float(q["pred"].mean()),
    }


def prob_metrics(y, p):
    q = pd.DataFrame({"y": num(y), "p": num(p)}).replace([np.inf, -np.inf], np.nan).dropna()
    if q.empty:
        return {"n": 0, "events": 0, "base_rate": np.nan, "mean_prob": np.nan,
                "cal_gap": np.nan, "auc": np.nan, "brier": np.nan, "logloss": np.nan}
    yy = q["y"].astype(int)
    pp = q["p"].clip(1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan
    return {
        "n": int(len(q)),
        "events": int(yy.sum()),
        "base_rate": float(yy.mean()),
        "mean_prob": float(pp.mean()),
        "cal_gap": float(pp.mean() - yy.mean()),
        "auc": auc,
        "brier": float(np.mean(np.square(pp - yy))),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])),
    }


def auc_score(y, score):
    q = pd.DataFrame({"y": num(y), "s": num(score)}).dropna()
    if q.empty or q["y"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(q["y"].astype(int), q["s"]))


def pct_rank(s):
    return num(s).rank(method="average", pct=True)


def load_m95d(root: Path):
    env = prep(pd.read_csv(find_one(root, "m95d_rb_environment_trace.csv"), low_memory=False))
    pred = pd.read_csv(find_one(root, "m95d_prediction_trace.csv"), low_memory=False)
    pred.columns = [str(c).lower() for c in pred.columns]
    pred = pred.loc[pred["target"].eq("rush_yards")].copy()
    needed = {"role_baseline", "role_plus_m95c_environment", "full_environment_matchup"}
    missing = needed - set(pred["family"].unique())
    if missing:
        raise RuntimeError(f"M96B M95D prediction families missing: {sorted(missing)}")
    wide = pred.pivot_table(index=["test_season", "row_index"], columns="family", values="prediction", aggfunc="first").reset_index()
    idcols = ["season", "week", "team", "player_join_key", "actual_rush_yards", "actual_carries",
              "role_is_workhorse", "rb_rb_share_avg1", "rb_rb_share_avg5"]
    ids = env[idcols].copy()
    ids["row_index"] = env.index
    return wide.merge(ids, left_on=["test_season", "row_index"], right_on=["season", "row_index"], how="left", validate="one_to_one")


def load_m95f(root: Path):
    f24 = prep(pd.read_csv(find_one(root, "m95f_2024_holdout_trace.csv"), low_memory=False))
    f25 = prep(pd.read_csv(find_one(root, "m95f_2025_rb_trace.csv"), low_memory=False))
    keep = KEYS + ["m94c_rush_att", "m95f_mix_mean", "m95f_p50", "m95f_p75", "m95f_p90", "m95f_p95",
                   "m95f_sim_prob_20", "m95f_sim_prob_25", "cal_prob_20", "cal_prob_25",
                   "actual_rush_yards", "actual_rush_att", "role_is_workhorse", "rb_rb_share_avg1", "rb_rb_share_avg5"]
    return f24[[c for c in keep if c in f24.columns]].copy(), f25[[c for c in keep if c in f25.columns]].copy()


def load_m94c(root: Path):
    x = prep(pd.read_csv(find_one(root, "m94c_2025_rb_trace.csv"), low_memory=False))
    return x[KEYS + ["candidate_rush_yards", "candidate_rush_att", "actual_rush_yards", "actual_rush_att"]].copy()


def load_m95i(root: Path):
    x = prep(pd.read_csv(find_one(root, "m95i_2025_trace.csv"), low_memory=False))
    return x[KEYS + ["prior_top1_unavailable", "p20_joint", "p25_joint", "cal_prob_20", "cal_prob_25"]].copy()


def build_tables(m94c_root, m95d_root, m95f_root, m95i_root):
    d = load_m95d(m95d_root)
    f24, f25 = load_m95f(m95f_root)
    c25 = load_m94c(m94c_root)
    i25 = load_m95i(m95i_root)
    d24 = d.loc[d["season"].eq(2024)].copy()
    d25 = d.loc[d["season"].eq(2025)].copy()
    j24 = d24.merge(f24, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_f"))
    j25 = d25.merge(f25, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_f"))
    j25 = j25.merge(c25, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_c"))
    j25 = j25.merge(i25, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_i"))
    audit = pd.DataFrame([
        {"universe": "m95d_2024_oos", "rows": len(d24), "joined_rows": len(j24), "coverage": len(j24)/max(len(d24),1)},
        {"universe": "m95d_2025_oos", "rows": len(d25), "joined_rows": len(j25), "coverage": len(j25)/max(len(d25),1)},
        {"universe": "m95f_2024_holdout", "rows": len(f24), "joined_rows": len(j24), "coverage": len(j24)/max(len(f24),1)},
        {"universe": "m95f_2025", "rows": len(f25), "joined_rows": len(j25), "coverage": len(j25)/max(len(f25),1)},
        {"universe": "m94c_2025", "rows": len(c25), "joined_rows": len(j25), "coverage": len(j25)/max(len(c25),1)},
    ])
    if len(j25) / max(len(d25), 1) < 0.97:
        raise RuntimeError(f"M96B primary join coverage below 97% of M95D OOS: {len(j25)}/{len(d25)}")
    if len(j24) < 300:
        raise RuntimeError(f"M96B 2024 calibration universe too small: {len(j24)}")
    yd25 = np.abs(num(j25["actual_rush_yards"]) - num(j25["actual_rush_yards_c"]))
    if yd25.max() > 1e-6:
        raise RuntimeError(f"M96B 2025 M95D/M94C yard truth mismatch max {yd25.max()}")
    if "actual_rush_yards_f" in j25.columns:
        shared = j25[["actual_rush_yards", "actual_rush_yards_f"]].dropna()
        if not shared.empty and np.abs(shared.iloc[:,0]-shared.iloc[:,1]).max() > 1e-6:
            raise RuntimeError("M96B M95F/M95D yard truth mismatch")
    return j24, j25, audit


def add_modules(j24, j25):
    for z in (j24, j25):
        z["e_delta"] = num(z["role_plus_m95c_environment"]) - num(z["role_baseline"])
        z["x_delta"] = num(z["full_environment_matchup"]) - num(z["role_plus_m95c_environment"])
        z["actual_75"] = num(z["actual_rush_yards"]).ge(75).astype(int)
        z["actual_100"] = num(z["actual_rush_yards"]).ge(100).astype(int)
        z["w_score_raw"] = (pct_rank(z["m95f_p90"]) + pct_rank(z["m95f_p95"])) / 2.0
        z["x_score_raw"] = pct_rank(z["x_delta"])
    j25["c_point"] = num(j25["candidate_rush_yards"])
    j25["ce_point"] = j25["c_point"] + num(j25["e_delta"])
    return j24, j25


def slice_masks(z):
    a = num(z["actual_carries"])
    vacancy = num(z["prior_top1_unavailable"]).fillna(0).eq(1) if "prior_top1_unavailable" in z.columns else pd.Series(False, index=z.index)
    stable = num(z["role_is_workhorse"]).fillna(0).eq(1) & (num(z["rb_rb_share_avg1"])-num(z["rb_rb_share_avg5"])).ge(-0.10)
    return {"all_rb": pd.Series(True, index=z.index), "actual_0_5": a.between(0,5), "actual_6_10": a.between(6,10),
            "actual_11_14": a.between(11,14), "actual_15_19": a.between(15,19), "actual_20_plus": a.ge(20),
            "actual_25_plus": a.ge(25), "incumbent": ~vacancy, "vacancy": vacancy, "stable_workhorse": stable}


def point_ablation(j25):
    rows=[]
    for scope_name, scope_mask in {"full_2025": pd.Series(True,index=j25.index), "late_2025_w13_18": num(j25["week"]).ge(13)}.items():
        q0=j25.loc[scope_mask]
        for sl, mask in slice_masks(q0).items():
            q=q0.loc[mask]
            for arm,col in [("C_m94c","c_point"),("C_plus_E","ce_point")]:
                rows.append({"scope":scope_name,"slice":sl,"arm":arm,**point_metrics(q["actual_rush_yards"],q[col])})
    out=pd.DataFrame(rows)
    def row(scope,sl,arm):
        return out.loc[(out.scope==scope)&(out.slice==sl)&(out.arm==arm)].iloc[0]
    base=row("full_2025","all_rb","C_m94c"); cand=row("full_2025","all_rb","C_plus_E")
    max_ord_reg=max(row("full_2025",sl,"C_plus_E")["mae"]-row("full_2025",sl,"C_m94c")["mae"] for sl in ["actual_0_5","actual_6_10","actual_11_14","actual_15_19"])
    e_pass = cand["mae"] < base["mae"] and abs(cand["bias"]) <= abs(base["bias"])+1.0 and max_ord_reg <= 1.0
    gate=pd.DataFrame([{"module":"E","base_mae":base["mae"],"candidate_mae":cand["mae"],"mae_gain":base["mae"]-cand["mae"],
                        "base_bias":base["bias"],"candidate_bias":cand["bias"],"max_ordinary_slice_mae_regression":max_ord_reg,
                        "point_gate_pass":int(e_pass),"selected_point_anchor":"CE" if e_pass else "C"}])
    return out,gate,e_pass


def build_tail_scores(z, point_col):
    q=z.copy(); q["b_score"]=pct_rank(q[point_col]); q["w_score"]=num(q["w_score_raw"]); q["x_score"]=num(q["x_score_raw"])
    q["score_B"]=q["b_score"]; q["score_BW"]=(q["b_score"]+q["w_score"])/2.0; q["score_BX"]=(q["b_score"]+q["x_score"])/2.0
    q["score_BWX"]=(q["b_score"]+q["w_score"]+q["x_score"])/3.0
    return q


def fit_platt(train_score, y):
    q=pd.DataFrame({"score":num(train_score),"y":num(y)}).dropna()
    if q["y"].nunique()<2: raise RuntimeError("M96B Platt train has one class")
    model=LogisticRegression(C=1.0,max_iter=2000,random_state=9602); model.fit(q[["score"]],q["y"].astype(int)); return model


def apply_platt(model, score):
    s=num(score); out=pd.Series(np.nan,index=s.index,dtype=float); mask=s.notna()
    out.loc[mask]=model.predict_proba(pd.DataFrame({"score":s.loc[mask]}))[:,1]
    return out.clip(1e-6,1-1e-6)


def tail_ablation(j24,j25,point_anchor):
    t24=j24.copy(); t24["b_score"]=pct_rank(t24["role_plus_m95c_environment"]); t24["w_score"]=num(t24["w_score_raw"]); t24["x_score"]=num(t24["x_score_raw"])
    t24["score_B"]=t24["b_score"]; t24["score_BW"]=(t24["b_score"]+t24["w_score"])/2.0; t24["score_BX"]=(t24["b_score"]+t24["x_score"])/2.0
    t24["score_BWX"]=(t24["b_score"]+t24["w_score"]+t24["x_score"])/3.0
    t25=build_tail_scores(j25, "ce_point" if point_anchor=="CE" else "c_point")
    arm_cols={"B":"score_B","B+W":"score_BW","B+X":"score_BX","B+W+X":"score_BWX"}; rows=[]; calrows=[]
    for threshold,target in TARGETS.items():
        for arm,col in arm_cols.items():
            model=fit_platt(t24[col],t24[target]); p24=apply_platt(model,t24[col]); p25=apply_platt(model,t25[col])
            calrows.append({"threshold":threshold,"arm":arm,"train_n":int(p24.notna().sum()),**{f"train_{k}":v for k,v in prob_metrics(t24[target],p24).items()}})
            for scope,mask in {"full_2025":pd.Series(True,index=t25.index),"late_2025_w13_18":num(t25["week"]).ge(13)}.items():
                rows.append({"threshold":threshold,"scope":scope,"arm":arm,**prob_metrics(t25.loc[mask,target],p25.loc[mask])})
            t25[f"p{threshold}_{arm.replace('+','_')}"]=p25
    return t24,t25,pd.DataFrame(calrows),pd.DataFrame(rows)


def module_retention(tail):
    rows=[]
    for module,arm in [("W","B+W"),("X","B+X")]:
        decisions=[]
        for th in [75,100]:
            fb=tail[(tail.threshold==th)&(tail.scope=="full_2025")&(tail.arm=="B")].iloc[0]; f=tail[(tail.threshold==th)&(tail.scope=="full_2025")&(tail.arm==arm)].iloc[0]
            lb=tail[(tail.threshold==th)&(tail.scope=="late_2025_w13_18")&(tail.arm=="B")].iloc[0]; l=tail[(tail.threshold==th)&(tail.scope=="late_2025_w13_18")&(tail.arm==arm)].iloc[0]
            material=((f.auc-fb.auc)>=0.005) or ((fb.brier-f.brier)>=0.001)
            no_damage=((f.auc-fb.auc)>=-0.005) and ((f.brier-fb.brier)<=0.001)
            late_ok=((l.auc-lb.auc)>=-0.02) and ((l.brier-lb.brier)<=0.003)
            decisions.append((th,material,no_damage,late_ok,f.auc-fb.auc,fb.brier-f.brier,l.auc-lb.auc,lb.brier-l.brier))
        retain=any(d[1] for d in decisions) and all(d[2] and d[3] for d in decisions)
        rows.append({"module":module,"status":"RETAIN" if retain else "REJECT","retention_pass":int(retain),
                     "detail":"; ".join(f"{d[0]}:full_auc_gain={d[4]:.6f},full_brier_gain={d[5]:.6f},late_auc_gain={d[6]:.6f},late_brier_gain={d[7]:.6f}" for d in decisions)})
    comp=[]
    for th in [75,100]:
        full=tail[(tail.threshold==th)&(tail.scope=="full_2025")]; late=tail[(tail.threshold==th)&(tail.scope=="late_2025_w13_18")]
        c=full[full.arm=="B+W+X"].iloc[0]; b=full[full.arm=="B"].iloc[0]; singles=full[full.arm.isin(["B+W","B+X"])]
        lc=late[late.arm=="B+W+X"].iloc[0]; lb=late[late.arm=="B"].iloc[0]
        noninfer=(c.auc>=float(singles.auc.max())-0.005) and (c.brier<=float(singles.brier.min())+0.001)
        lateok=(lc.auc>=lb.auc-0.02) and (lc.brier<=lb.brier+0.003); improve=((c.auc-b.auc)>=0.005) or ((b.brier-c.brier)>=0.001)
        comp.append((th,noninfer,lateok,improve,c.auc-b.auc,b.brier-c.brier))
    retain_combo=all(x[1] and x[2] for x in comp) and any(x[3] for x in comp)
    rows.append({"module":"W+X_COMBINATION","status":"PREFERRED" if retain_combo else "NOT_PREFERRED","retention_pass":int(retain_combo),
                 "detail":"; ".join(f"{x[0]}:auc_gain_vs_B={x[4]:.6f},brier_gain_vs_B={x[5]:.6f}" for x in comp)})
    return pd.DataFrame(rows)


def v_diagnostic(t25):
    vac=num(t25["prior_top1_unavailable"]).fillna(0).eq(1); q=t25.loc[vac].copy(); rows=[]
    for th,vcol in [(75,"p20_joint"),(100,"p25_joint")]:
        y=q[TARGETS[th]]; base=q["score_BWX"]; v=pct_rank(q[vcol]); fused=(pct_rank(base)+v)/2.0
        ba=auc_score(y,base); va=auc_score(y,fused)
        rows.append({"threshold":th,"n":len(q),"events":int(y.sum()),"base_auc":ba,"v_fused_auc":va,"auc_gain":va-ba,
                     "status":"RETAIN_VACANCY_DIAGNOSTIC_FOR_PROSPECTIVE_CONFIRMATION" if va>ba else "REJECT_V_INCREMENT_IN_M96B"})
    return pd.DataFrame(rows)


def casebook(t25):
    q=t25.copy(); q["w_minus_b"]=q["w_score"]-q["b_score"]; q["x_minus_b"]=q["x_score"]-q["b_score"]
    q["abs_module_disagreement"]=q[["w_minus_b","x_minus_b"]].abs().max(axis=1)
    cols=KEYS+["actual_rush_yards","actual_carries","c_point","ce_point","e_delta","w_score","x_score","score_B","score_BW","score_BX","score_BWX","w_minus_b","x_minus_b","abs_module_disagreement"]
    cols += [c for c in ["prior_top1_unavailable","p20_joint","p25_joint"] if c in q.columns]
    return q.sort_values("abs_module_disagreement",ascending=False)[cols].head(50)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m94c-root",type=Path,required=True); ap.add_argument("--m95d-root",type=Path,required=True)
    ap.add_argument("--m95f-root",type=Path,required=True); ap.add_argument("--m95i-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,default=Path("data/backtests/rb_m96b"))
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    j24,j25,audit=build_tables(args.m94c_root,args.m95d_root,args.m95f_root,args.m95i_root); j24,j25=add_modules(j24,j25)
    points,egate,e_pass=point_ablation(j25); point_anchor="CE" if e_pass else "C"; t24,t25,cal,tail=tail_ablation(j24,j25,point_anchor)
    modules=module_retention(tail); vdiag=v_diagnostic(t25); cb=casebook(t25)
    ledger=pd.concat([pd.DataFrame([{"module":"C","status":"RETAIN","retention_pass":1,"detail":"M94C frozen central opportunity / point anchor"},
                                   {"module":"E","status":"RETAIN" if e_pass else "REJECT","retention_pass":int(e_pass),"detail":f"direct frozen M95C environment residual; point anchor={point_anchor}"}]),modules],ignore_index=True)
    v_status="RETAIN_DIAGNOSTIC" if (vdiag["auc_gain"]>0).any() else "REJECT"
    ledger=pd.concat([ledger,pd.DataFrame([{"module":"V","status":v_status,"retention_pass":int((vdiag["auc_gain"]>0).any()),"detail":"2025 vacancy-only ranking diagnostic; never production-promoted by M96B"}])],ignore_index=True)
    combo=int(ledger.loc[ledger.module.eq("W+X_COMBINATION"),"retention_pass"].iloc[0]); wk=int(ledger.loc[ledger.module.eq("W"),"retention_pass"].iloc[0]); xk=int(ledger.loc[ledger.module.eq("X"),"retention_pass"].iloc[0])
    tail_arch="B+W+X" if combo else "TARGET_SPECIFIC_W_AND_X" if wk and xk else "B+W" if wk else "B+X" if xk else "B_ONLY"
    disposition=pd.DataFrame([{"point_anchor":point_anchor,"e_point_gate_pass":int(e_pass),"w_retained":wk,"x_retained":xk,"wx_combination_preferred":combo,
                               "tail_architecture":tail_arch,"v_status":v_status,"disposition":"M96B_MODULAR_SYNTHESIS_COMPLETE","model_fit":1,"feature_search":0,"weight_search":0,"sportsbook_inputs":0,"production_change":0}])
    drows=[]
    for yr,z in [(2024,j24),(2025,j25)]:
        for arm,col in [("role_baseline","role_baseline"),("E_environment","role_plus_m95c_environment"),("X_full_matchup","full_environment_matchup")]:
            drows.append({"season":yr,"arm":arm,**point_metrics(z["actual_rush_yards"],z[col]),"auc75":auc_score(z["actual_75"],z[col]),"auc100":auc_score(z["actual_100"],z[col])})
    support=pd.DataFrame(drows)
    audit.to_csv(args.out_dir/"m96b_source_audit.csv",index=False); egate.to_csv(args.out_dir/"m96b_e_point_gate.csv",index=False); points.to_csv(args.out_dir/"m96b_point_ablation.csv",index=False)
    cal.to_csv(args.out_dir/"m96b_2024_platt_calibration.csv",index=False); tail.to_csv(args.out_dir/"m96b_tail_ablation.csv",index=False); vdiag.to_csv(args.out_dir/"m96b_vacancy_diagnostic.csv",index=False)
    ledger.to_csv(args.out_dir/"m96b_module_ledger.csv",index=False); disposition.to_csv(args.out_dir/"m96b_disposition.csv",index=False); support.to_csv(args.out_dir/"m96b_frozen_module_support.csv",index=False); cb.to_csv(args.out_dir/"m96b_casebook.csv",index=False)
    print("=== M96B source audit ==="); print(audit.to_string(index=False)); print("=== M96B E point gate ==="); print(egate.to_string(index=False))
    print("=== M96B point all-RB ==="); print(points.loc[points["slice"].eq("all_rb")].to_string(index=False)); print("=== M96B tail full 2025 ==="); print(tail.loc[tail["scope"].eq("full_2025")].to_string(index=False))
    print("=== M96B module ledger ==="); print(ledger.to_string(index=False)); print("=== M96B V diagnostic ==="); print(vdiag.to_string(index=False)); print("=== M96B disposition ==="); print(disposition.to_string(index=False)); return 0


if __name__=="__main__":
    raise SystemExit(main())
