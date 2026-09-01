"""M96D: pregame conditional efficiency routing audit.

Protocol frozen in docs/migrations/M96D_RB_PREGAME_CONDITIONAL_EFFICIENCY_ROUTING_PLAN.md.
No new outcome-trained router. M94C carries/center remain fixed; target-week actuals
are evaluation-only. Primary D gate is the only retention-eligible arm.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ALIASES = {"audricestime": "audricestim"}
KEYS = ["season", "week", "team", "player_join_key"]
PRIMARY_CARRY_THRESHOLD = 15.0
ENTRENCHED_SHARE_THRESHOLD = 0.65


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


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def prep_d(x):
    z = x.copy(); z.columns = [str(c).lower() for c in z.columns]
    z["season"] = num(z["season"]).astype(int); z["week"] = num(z["week"]).astype(int)
    z["team"] = z["team"].map(canon_team)
    if "player_clean_key" not in z.columns:
        raise RuntimeError("M96D M95D source missing player_clean_key")
    z["player_join_key"] = z["player_clean_key"].map(clean_player_key)
    return z


def point_metrics(actual, pred):
    q = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    if q.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = q["pred"] - q["actual"]
    corr = q["actual"].corr(q["pred"]) if len(q) >= 3 and q["actual"].nunique() > 1 and q["pred"].nunique() > 1 else np.nan
    return {"n": int(len(q)), "mae": float(e.abs().mean()), "rmse": float(np.sqrt(np.mean(np.square(e)))),
            "bias": float(e.mean()), "corr": float(corr) if pd.notna(corr) else np.nan}


def auc_safe(y, score):
    q = pd.DataFrame({"y": num(y), "s": num(score)}).dropna()
    if q.empty or q["y"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(q["y"].astype(int), q["s"]))


def load_join(m96c_root: Path, m95d_root: Path):
    c = pd.read_csv(find_one(m96c_root, "m96c_oof_trace.csv"), low_memory=False)
    c.columns = [str(x).lower() for x in c.columns]
    c["season"] = num(c["season"]).astype(int); c["week"] = num(c["week"]).astype(int); c["team"] = c["team"].map(canon_team)
    d = prep_d(pd.read_csv(find_one(m95d_root, "m95d_rb_environment_trace.csv"), low_memory=False))
    d = d.loc[d["season"].eq(2025)].copy()
    need_c = KEYS + ["player", "actual_rush_att", "actual_rush_yards_m94c", "candidate_rush_att", "candidate_rush_yards",
                     "pred_d", "pred_e", "pred_p"]
    missing = [x for x in need_c if x not in c.columns]
    if missing: raise RuntimeError(f"M96D M96C trace missing {missing}")
    dkeep = KEYS + [c for c in ["role_is_workhorse", "role_is_starter_plus", "rb_rb_share_avg1", "rb_rb_share_avg3", "rb_rb_share_avg5",
                                      "team_top1_share_avg3", "team_top1_share_avg5", "team_rb_used_avg3", "team_rb_used_avg5",
                                      "team_rb_pool_avg3", "team_rb_pool_avg5", "off_role_opportunity_score", "pregame_role"] if c in d.columns]
    j = c[need_c].merge(d[dkeep], on=KEYS, how="left", validate="one_to_one")
    cov = float(j["role_is_workhorse"].notna().mean()) if "role_is_workhorse" in j else 0.0
    if cov < 0.97: raise RuntimeError(f"M96D workload-state join coverage below 97%: {cov}")
    audit = pd.DataFrame([{"m96c_rows": len(c), "joined_rows": len(j), "role_state_coverage": cov,
                           "m96c_eval_rows_w6_18": int(num(c["week"]).ge(6).sum())}])
    return j, audit


def add_routes(z):
    q = z.copy()
    q["role_is_workhorse"] = num(q["role_is_workhorse"]).fillna(0)
    q["rb_rb_share_avg5"] = num(q["rb_rb_share_avg5"])
    q["entrenched_workhorse"] = q["role_is_workhorse"].eq(1) & q["rb_rb_share_avg5"].ge(ENTRENCHED_SHARE_THRESHOLD)
    q["low_projected_workload"] = num(q["candidate_rush_att"]).lt(PRIMARY_CARRY_THRESHOLD)
    q["gate_primary"] = q["low_projected_workload"] & ~q["entrenched_workhorse"]
    q["gate_carry_only"] = q["low_projected_workload"]
    q["gate_role_only"] = ~q["entrenched_workhorse"]
    q["pred_C"] = num(q["candidate_rush_yards"])
    q["pred_R_D_PRIMARY"] = np.where(q["gate_primary"], num(q["pred_d"]), q["pred_C"])
    q["pred_R_D_CARRY_ONLY"] = np.where(q["gate_carry_only"], num(q["pred_d"]), q["pred_C"])
    q["pred_R_D_ROLE_ONLY"] = np.where(q["gate_role_only"], num(q["pred_d"]), q["pred_C"])
    q["pred_R_E_PRIMARY"] = np.where(q["gate_primary"], num(q["pred_e"]), q["pred_C"])
    q["pred_R_P_PRIMARY"] = np.where(q["gate_primary"], num(q["pred_p"]), q["pred_C"])
    q["actual_75"] = num(q["actual_rush_yards_m94c"]).ge(75).astype(int)
    q["actual_100"] = num(q["actual_rush_yards_m94c"]).ge(100).astype(int)
    return q


def slices(q):
    a = num(q["actual_rush_att"])
    return {"all_rb": pd.Series(True, index=q.index), "actual_0_5": a.between(0,5), "actual_6_10": a.between(6,10),
            "actual_11_14": a.between(11,14), "actual_15_19": a.between(15,19), "actual_20_plus": a.ge(20), "actual_25_plus": a.ge(25)}


def evaluate(q):
    arms = ["C", "R_D_PRIMARY", "R_D_CARRY_ONLY", "R_D_ROLE_ONLY", "R_E_PRIMARY", "R_P_PRIMARY"]
    rows=[]; tail=[]
    evalq=q.loc[num(q["week"]).ge(6)].copy()
    for scope,sm in {"weeks6_18":pd.Series(True,index=evalq.index), "weeks13_18":num(evalq["week"]).ge(13)}.items():
        s=evalq.loc[sm].copy()
        for sl,mask in slices(s).items():
            g=s.loc[mask]
            for arm in arms:
                rows.append({"scope":scope,"slice":sl,"arm":arm,**point_metrics(g["actual_rush_yards_m94c"],g[f"pred_{arm}"])})
        for th in [75,100]:
            y=s[f"actual_{th}"]
            for arm in arms:
                tail.append({"scope":scope,"threshold":th,"arm":arm,"n":len(s),"events":int(y.sum()),"auc":auc_safe(y,s[f"pred_{arm}"])})
    return pd.DataFrame(rows), pd.DataFrame(tail)


def activation(q):
    x=q.loc[num(q["week"]).ge(6)].copy(); rows=[]
    strata={"all":pd.Series(True,index=x.index), "proj_lt10":num(x["candidate_rush_att"]).lt(10),
            "proj_10_14_99":num(x["candidate_rush_att"]).ge(10)&num(x["candidate_rush_att"]).lt(15),
            "proj_15_plus":num(x["candidate_rush_att"]).ge(15), "entrenched_workhorse":x["entrenched_workhorse"],
            "not_entrenched":~x["entrenched_workhorse"]}
    for name,m in strata.items():
        g=x.loc[m]
        rows.append({"stratum":name,"n":len(g),"primary_activation_rate":float(g["gate_primary"].mean()) if len(g) else np.nan,
                     "mean_projected_carries":float(num(g["candidate_rush_att"]).mean()) if len(g) else np.nan,
                     "actual_20plus_rate_eval_only":float(num(g["actual_rush_att"]).ge(20).mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def gate(point, tail):
    def pr(scope,sl,arm): return point.loc[(point.scope==scope)&(point.slice==sl)&(point.arm==arm)].iloc[0]
    b=pr("weeks6_18","all_rb","C"); r=pr("weeks6_18","all_rb","R_D_PRIMARY"); lateb=pr("weeks13_18","all_rb","C"); later=pr("weeks13_18","all_rb","R_D_PRIMARY")
    slice_regs={sl:pr("weeks6_18",sl,"R_D_PRIMARY").mae-pr("weeks6_18",sl,"C").mae for sl in ["actual_15_19","actual_20_plus","actual_25_plus"]}
    auc_regs={}
    for th in [75,100]:
        tb=tail.loc[(tail.scope=="weeks6_18")&(tail.threshold==th)&(tail.arm=="C")].iloc[0]
        tr=tail.loc[(tail.scope=="weeks6_18")&(tail.threshold==th)&(tail.arm=="R_D_PRIMARY")].iloc[0]
        auc_regs[th]=tr.auc-tb.auc
    checks={"mae_gain_ge_0p10":b.mae-r.mae>=0.10,"rmse_reg_le_0p15":r.rmse-b.rmse<=0.15,
            "abs_bias_worsen_le_1p0":abs(r.bias)-abs(b.bias)<=1.0,"actual_15_19_reg_le_0p50":slice_regs["actual_15_19"]<=0.50,
            "actual_20_plus_reg_le_0p50":slice_regs["actual_20_plus"]<=0.50,"actual_25_plus_reg_le_0p50":slice_regs["actual_25_plus"]<=0.50,
            "auc75_reg_ge_neg0p005":auc_regs[75]>=-0.005,"auc100_reg_ge_neg0p005":auc_regs[100]>=-0.005,
            "late_mae_reg_le_0p10":later.mae-lateb.mae<=0.10}
    passed=all(checks.values())
    row={"retention_pass":int(passed),"base_mae":b.mae,"routed_mae":r.mae,"mae_gain":b.mae-r.mae,"base_rmse":b.rmse,"routed_rmse":r.rmse,
         "rmse_gain":b.rmse-r.rmse,"base_bias":b.bias,"routed_bias":r.bias,"late_mae_gain":lateb.mae-later.mae,
         "actual_15_19_mae_regression":slice_regs["actual_15_19"],"actual_20_plus_mae_regression":slice_regs["actual_20_plus"],
         "actual_25_plus_mae_regression":slice_regs["actual_25_plus"],"auc75_gain":auc_regs[75],"auc100_gain":auc_regs[100],
         **{f"check_{k}":int(v) for k,v in checks.items()}}
    return pd.DataFrame([row])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m96c-root",type=Path,required=True); ap.add_argument("--m95d-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    q,audit=load_join(args.m96c_root,args.m95d_root); q=add_routes(q); point,tail=evaluate(q); act=activation(q); gt=gate(point,tail)
    passed=bool(gt.iloc[0].retention_pass)
    # Diagnostic comparisons are recorded but are never eligible to override the primary frozen arm.
    allrows=point.loc[(point.scope=="weeks6_18")&(point.slice=="all_rb")].copy()
    base=float(allrows.loc[allrows.arm=="C","mae"].iloc[0]); allrows["mae_gain_vs_C"]=base-allrows["mae"]
    disp=pd.DataFrame([{"selected_arm":"R_D_PRIMARY" if passed else "C_M94C","primary_retention_pass":int(passed),
                        "disposition":"M96D_ROUTE_RETAIN_RESEARCH_ONLY" if passed else "M96D_PRIMARY_ROUTER_FAILED",
                        "next_step":"PROSPECTIVE_2026_CONFIRMATION" if passed else "STOP_OR_ONE_PRECOMMITTED_ROUTER_TYPE_IF_JUSTIFIED",
                        "model_fit":0,"threshold_search":0,"feature_search":0,"sportsbook_inputs":0,"production_change":0}])
    ledger=pd.DataFrame([
        {"module":"C","status":"RETAIN","detail":"Frozen M94C center"},
        {"module":"D","status":"ROUTED_RESEARCH_ONLY" if passed else "CONDITIONAL_CLUE","detail":"M96C opponent-defense efficiency expert under frozen pregame router"},
        {"module":"E","status":"CONTROL_ONLY","detail":"Primary-router controlled alternative; not selection eligible"},
        {"module":"P","status":"CONTROL_ONLY","detail":"Primary-router controlled alternative; not selection eligible"},
        {"module":"X","status":"REJECT_ISOLATED","detail":"Frozen from M96C"},
    ])
    audit.to_csv(args.out_dir/"m96d_source_audit.csv",index=False); q.to_csv(args.out_dir/"m96d_router_trace.csv",index=False); point.to_csv(args.out_dir/"m96d_point_metrics.csv",index=False)
    tail.to_csv(args.out_dir/"m96d_tail_auc.csv",index=False); act.to_csv(args.out_dir/"m96d_activation.csv",index=False); gt.to_csv(args.out_dir/"m96d_retention_gate.csv",index=False)
    allrows.to_csv(args.out_dir/"m96d_arm_summary.csv",index=False); disp.to_csv(args.out_dir/"m96d_disposition.csv",index=False); ledger.to_csv(args.out_dir/"m96d_capability_ledger.csv",index=False)
    print("=== M96D source audit ==="); print(audit.to_string(index=False)); print("=== M96D all-RB arm summary ==="); print(allrows.to_string(index=False)); print("=== M96D activation ==="); print(act.to_string(index=False)); print("=== M96D gate ==="); print(gt.to_string(index=False)); print("=== M96D disposition ==="); print(disp.to_string(index=False))

if __name__=="__main__": main()
