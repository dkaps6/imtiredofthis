#!/usr/bin/env python3
"""M88: untouched-2023 replication of M87 low-chaos QB failure regimes.

No passing-yard correction is fit. 2023 target-game outcomes/PBP are used only
for confirmation labels after the M87-derived regime definitions are frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.ensemble_v2 import apply_ensemble, fit_market_weights

SEASON = 2023
PRIOR_SEASON = 2022
HISTORY_GAMES = 8
MIN_HISTORY_GAMES = 4
MIN_ENSEMBLE_ROWS = 40
BOOT_N = 2000
BOOT_SEED = 88

# Frozen before 2023 is opened: exact midpoint of M87 target/control means.
VOL_DEF_PASS_RATE_MIN = 0.6062143950065955
VOL_OFF_DEEP_RATE_MAX = 0.18505508535326465
EFF_DEF_SUCCESS_MAX = 0.42486131276490247
EFF_DEF_YPA_MAX = 6.426889901131469

TEAM_REMAP = {"JAC": "JAX", "LA": "LAR", "OAK": "LV", "SD": "LAC", "STL": "LAR"}
MARKET_TOKENS = ("sportsbook", "prop_line", "moneyline", "game_total", "vegas", "spread_line", "total_line")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def canon_team_value(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    return TEAM_REMAP.get(s, s)


def canon_team(s: pd.Series) -> pd.Series:
    return s.map(canon_team_value)


def player_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def n(x: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in x.columns:
        return pd.Series(default, index=x.index, dtype=float)
    return pd.to_numeric(x[col], errors="coerce").fillna(default)


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    return aa.div(bb.where(bb.ne(0)))


def load_predictions(path: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(path, low_memory=False))
    required = {"season", "week", "team", "opponent", "player_clean_key", "market", "actual", "mc_proj", "ml_proj", "state_proj"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"M88 component predictions missing columns: {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(SEASON) & x["week"].between(1, 18)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["team"] = canon_team(x["team"])
    x["opponent"] = canon_team(x["opponent"])
    x["player_clean_key"] = player_key(x["player_clean_key"])
    bad = [c for c in x.columns if any(tok in c for tok in MARKET_TOKENS)]
    selected = {"season","week","team","opponent","player_clean_key","market","actual","mc_proj","ml_proj","state_proj",
                "mc_expected_pass_attempts","mc_rules_ypa"}
    bad_selected = sorted(set(bad) & selected)
    if bad_selected:
        raise RuntimeError(f"M88 selected prediction field crosses sportsbook boundary: {bad_selected}")
    q = x.loc[x["market"].astype(str).str.lower().eq("pass_yards")].copy()
    if q.empty:
        raise RuntimeError("M88 found no 2023 pass_yards component predictions")
    if q.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("M88 duplicate 2023 pass-yards prediction identities")
    if "mc_expected_pass_attempts" not in q.columns:
        raise RuntimeError("M88 current-stack trace lacks mc_expected_pass_attempts")
    return q


def build_oos_ensemble(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, weights_rows = [], []
    for week in sorted(q["week"].dropna().astype(int).unique()):
        target = q.loc[q["week"].eq(week)].copy()
        hist = q.loc[q["week"].lt(week)].copy()
        usable = hist.dropna(subset=["actual","mc_proj","ml_proj","state_proj"])
        if len(usable) >= MIN_ENSEMBLE_ROWS:
            w = fit_market_weights(usable, min_rows=MIN_ENSEMBLE_ROWS)
            applied = apply_ensemble(target, weights=w)
            if not w.empty:
                wr = w.loc[w["market"].astype(str).str.lower().eq("pass_yards")]
                if not wr.empty:
                    d = wr.iloc[0].to_dict()
                    d.update({"target_week": int(week), "fit_scope": "earlier_2023_oos"})
                    weights_rows.append(d)
        else:
            applied = apply_ensemble(target, weights=pd.DataFrame())
            weights_rows.append({
                "market":"pass_yards","mc_weight":1.0,"ml_weight":0.0,"state_weight":0.0,
                "calibration_rows":int(len(usable)),"method":"mc_fallback_insufficient_prior_oos",
                "target_week":int(week),"fit_scope":"earlier_2023_oos",
            })
        out.append(applied)
    z = pd.concat(out, ignore_index=True)
    if z["ensemble_proj"].isna().any():
        raise RuntimeError(f"M88 OOS ensemble missing predictions: {int(z['ensemble_proj'].isna().sum())}")
    return z, pd.DataFrame(weights_rows)


def load_logs(path: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(path, low_memory=False))
    required = {"season","week","team","player","pass_att","pass_yards"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"M88 player logs missing columns: {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(SEASON) & x["week"].between(1,18)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["team"] = canon_team(x["team"])
    if "player_clean_key" in x.columns:
        x["player_clean_key"] = player_key(x["player_clean_key"])
    else:
        x["player_clean_key"] = player_key(x["player"])
    x["pass_att_num"] = pd.to_numeric(x["pass_att"], errors="coerce").fillna(0)
    x["pass_yards_num"] = pd.to_numeric(x["pass_yards"], errors="coerce")
    return x


def build_stable_cohort(ensemble_rows: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    totals = logs.groupby(["week","team"], as_index=False)["pass_att_num"].sum().rename(columns={"pass_att_num":"team_qb_attempts"})
    lp = logs.merge(totals, on=["week","team"], how="left")
    lp["qb_attempt_share"] = lp["pass_att_num"] / lp["team_qb_attempts"].replace(0, np.nan)
    prim = (lp.sort_values(["week","team","pass_att_num"], ascending=[True,True,False])
              .drop_duplicates(["week","team"])
              [["week","team","player_clean_key","pass_att_num","team_qb_attempts","qb_attempt_share"]]
              .rename(columns={"player_clean_key":"actual_primary_key","pass_att_num":"actual_primary_attempts",
                               "qb_attempt_share":"actual_qb_attempt_share"}))
    actual = (logs[["week","team","player_clean_key","pass_att_num","pass_yards_num"]]
              .drop_duplicates(["week","team","player_clean_key"])
              .rename(columns={"pass_att_num":"actual_attempts","pass_yards_num":"actual_pass_yards"}))

    q = ensemble_rows.merge(prim, on=["week","team"], how="left", validate="many_to_one")
    q = q.loc[q["player_clean_key"].eq(q["actual_primary_key"]) & q["actual_qb_attempt_share"].ge(0.80)].copy()
    q = q.merge(actual, on=["week","team","player_clean_key"], how="left", validate="one_to_one")
    if len(q) < 250:
        raise RuntimeError(f"M88 stable-primary 2023 cohort unexpectedly small: {len(q)}")
    if q.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("M88 stable-primary identity duplication")

    q["pred_attempts"] = pd.to_numeric(q["mc_expected_pass_attempts"], errors="coerce")
    q["pred_pass_yards"] = pd.to_numeric(q["mc_proj"], errors="coerce")
    q["implied_pred_ypa"] = q["pred_pass_yards"] / q["pred_attempts"].replace(0, np.nan)
    q["actual_attempts"] = pd.to_numeric(q["actual_attempts"], errors="coerce")
    q["actual_pass_yards"] = pd.to_numeric(q["actual_pass_yards"], errors="coerce")
    q["actual_ypa"] = q["actual_pass_yards"] / q["actual_attempts"].replace(0, np.nan)
    q["ensemble_proj"] = pd.to_numeric(q["ensemble_proj"], errors="coerce")
    core = ["actual_pass_yards","actual_attempts","pred_attempts","implied_pred_ypa","ensemble_proj"]
    if q[core].isna().any().any():
        counts = q[core].isna().sum().to_dict()
        raise RuntimeError(f"M88 stable-primary core value missing: {counts}")
    return q.sort_values(["week","team","player_clean_key"]).reset_index(drop=True)


def load_pbp() -> pd.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_pbp(seasons=[PRIOR_SEASON, SEASON])
    x = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    x = lower(x)
    if "season_type" in x.columns:
        x = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].isin([PRIOR_SEASON,SEASON]) & x["week"].between(1,18)].copy()
    x["team"] = canon_team(x["posteam"])
    x["opponent"] = canon_team(x["defteam"])
    return x


def aggregate_team_games(pbp: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    x = pbp.copy()
    pass_attempt = n(x,"pass_attempt").eq(1)
    sack = n(x,"sack").eq(1)
    scramble = n(x,"qb_scramble").eq(1)
    dropback = n(x,"qb_dropback").eq(1) if "qb_dropback" in x.columns else (pass_attempt | sack | scramble)
    rush = n(x,"rush_attempt").eq(1) & ~scramble
    valid = dropback | rush
    pass_yards = pd.to_numeric(x["passing_yards"],errors="coerce").fillna(0.0) if "passing_yards" in x.columns else pd.Series(0.0,index=x.index)
    air = pd.to_numeric(x["air_yards"],errors="coerce") if "air_yards" in x.columns else pd.Series(np.nan,index=x.index)
    success = pd.to_numeric(x["success"],errors="coerce") if "success" in x.columns else pd.Series(np.nan,index=x.index)
    x["m88_valid"] = valid.astype(int)
    x["m88_pass_call"] = dropback.astype(int)
    x["m88_pass_attempt"] = pass_attempt.astype(int)
    x["m88_pass_yards"] = np.where(pass_attempt, pass_yards, 0.0)
    x["m88_deep"] = (pass_attempt & air.ge(15)).astype(int)
    x["m88_success"] = success.where(valid)
    off = x.groupby(["season","week","team","opponent"],as_index=False).agg(
        plays=("m88_valid","sum"), pass_calls=("m88_pass_call","sum"),
        pass_attempts=("m88_pass_attempt","sum"), pass_yards=("m88_pass_yards","sum"),
        deep_attempts=("m88_deep","sum"), success_rate=("m88_success","mean"),
    )
    off["pass_rate"] = safe_div(off["pass_calls"],off["plays"])
    off["deep_attempt_rate"] = safe_div(off["deep_attempts"],off["pass_attempts"])
    off["ypa"] = safe_div(off["pass_yards"],off["pass_attempts"])
    defense = pd.DataFrame({
        "season":off["season"],"week":off["week"],"team":off["opponent"],"opponent":off["team"],
        "def_pass_rate_faced":off["pass_rate"],
        "def_success_rate_allowed":off["success_rate"],
        "def_ypa_allowed":off["ypa"],
    })
    return off.sort_values(["season","week","team"]).reset_index(drop=True), defense.sort_values(["season","week","team"]).reset_index(drop=True)


def prior_window(df: pd.DataFrame, team: str, week: int) -> pd.DataFrame:
    h = df.loc[
        df["team"].eq(team)
        & df["season"].ge(PRIOR_SEASON)
        & ((df["season"].lt(SEASON)) | (df["season"].eq(SEASON) & df["week"].lt(week)))
    ].sort_values(["season","week"])
    return h.tail(HISTORY_GAMES)


def add_regime_history(q: pd.DataFrame, off: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in q.iterrows():
        team = canon_team_value(r["team"])
        opp = canon_team_value(r["opponent"])
        week = int(r["week"])
        oh = prior_window(off, team, week)
        dh = prior_window(defense, opp, week)
        rows.append({
            "off_history_games":len(oh),
            "def_history_games":len(dh),
            "off_deep_attempt_rate":float(pd.to_numeric(oh["deep_attempt_rate"],errors="coerce").mean()) if len(oh)>=MIN_HISTORY_GAMES else np.nan,
            "def_pass_rate_faced":float(pd.to_numeric(dh["def_pass_rate_faced"],errors="coerce").mean()) if len(dh)>=MIN_HISTORY_GAMES else np.nan,
            "def_success_rate_allowed":float(pd.to_numeric(dh["def_success_rate_allowed"],errors="coerce").mean()) if len(dh)>=MIN_HISTORY_GAMES else np.nan,
            "def_ypa_allowed":float(pd.to_numeric(dh["def_ypa_allowed"],errors="coerce").mean()) if len(dh)>=MIN_HISTORY_GAMES else np.nan,
        })
    h = pd.DataFrame(rows,index=q.index)
    out = pd.concat([q.reset_index(drop=True),h.reset_index(drop=True)],axis=1)
    feats=["off_deep_attempt_rate","def_pass_rate_faced","def_success_rate_allowed","def_ypa_allowed"]
    out["regime_history_complete"] = out[feats].notna().all(axis=1)
    return out


def aggregate_chaos_events(pbp_2023: pd.DataFrame) -> pd.DataFrame:
    x = pbp_2023.copy()
    pa = n(x,"pass_attempt").eq(1)
    cp = n(x,"complete_pass").eq(1)
    py = n(x,"passing_yards")
    yac = pd.to_numeric(x["yards_after_catch"],errors="coerce") if "yards_after_catch" in x.columns else pd.Series(np.nan,index=x.index)
    x["pass40"]=(pa&cp&py.ge(40)).astype(int)
    x["pass60"]=(pa&cp&py.ge(60)).astype(int)
    x["yac30"]=(pa&cp&yac.ge(30)).astype(int)
    x["ints"]=n(x,"interception").eq(1).astype(int)
    x["sacks"]=n(x,"sack").eq(1).astype(int)
    x["scrambles"]=n(x,"qb_scramble").eq(1).astype(int)
    x["ot"]=n(x,"qtr").ge(5).astype(int)
    x["fourth"]=((n(x,"down").eq(4)) & (pa | n(x,"rush_attempt").eq(1))).astype(int)
    g=x.groupby(["season","week","team"],as_index=False).agg(
        pass_40_plus=("pass40","sum"),pass_60_plus=("pass60","sum"),yac_30_plus=("yac30","sum"),
        interceptions=("ints","sum"),sacks=("sacks","sum"),qb_scrambles=("scrambles","sum"),
        overtime=("ot","max"),fourth_down_attempts=("fourth","sum"),
    )
    g["high_event_chaos"]=(
        g["pass_60_plus"].ge(1)|g["pass_40_plus"].ge(2)|g["yac_30_plus"].ge(1)|
        g["overtime"].ge(1)|g["sacks"].ge(4)|g["interceptions"].ge(2)|
        g["qb_scrambles"].ge(5)|g["fourth_down_attempts"].ge(4)
    )
    return g


def add_confirmation_labels(q: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    x=q.merge(events,on=["season","week","team"],how="left",validate="many_to_one")
    if x["high_event_chaos"].isna().any():
        raise RuntimeError(f"M88 target-game event join missing {int(x['high_event_chaos'].isna().sum())} rows")
    x["ensemble_error"]=x["ensemble_proj"]-x["actual_pass_yards"]
    x["ensemble_abs_error"]=x["ensemble_error"].abs()
    x["tail100"]=x["ensemble_abs_error"].ge(100)
    x["tail_direction"]=np.where(x["ensemble_error"].le(-100),"UNDERPROJECTED",
                         np.where(x["ensemble_error"].ge(100),"OVERPROJECTED","NONTAIL"))
    x["attempt_resid"]=x["actual_attempts"]-x["pred_attempts"]
    x["ypa_resid"]=x["actual_ypa"]-x["implied_pred_ypa"]
    ac=(x["attempt_resid"]*x["implied_pred_ypa"]).abs()
    yc=(x["ypa_resid"]*x["pred_attempts"]).abs()
    x["component_class"]="MIXED"
    x.loc[ac.ge(1.25*yc),"component_class"]="VOLUME_DOMINANT"
    x.loc[yc.ge(1.25*ac),"component_class"]="EFFICIENCY_DOMINANT"
    x["low_event_chaos"]=~x["high_event_chaos"].astype(bool)
    x["volume_expected_event"]=(
        x["tail100"] & x["low_event_chaos"] & x["component_class"].eq("VOLUME_DOMINANT") &
        x["tail_direction"].eq("UNDERPROJECTED")
    )
    x["efficiency_expected_event"]=(
        x["tail100"] & x["low_event_chaos"] & x["component_class"].eq("EFFICIENCY_DOMINANT") &
        x["tail_direction"].eq("OVERPROJECTED")
    )
    x["volume_regime"]=(
        x["regime_history_complete"] &
        x["def_pass_rate_faced"].ge(VOL_DEF_PASS_RATE_MIN) &
        x["off_deep_attempt_rate"].le(VOL_OFF_DEEP_RATE_MAX)
    )
    x["efficiency_regime"]=(
        x["regime_history_complete"] &
        x["def_success_rate_allowed"].le(EFF_DEF_SUCCESS_MAX) &
        x["def_ypa_allowed"].le(EFF_DEF_YPA_MAX)
    )
    return x


def bootstrap_support(reg: pd.Series, non: pd.Series, direction: str, seed: int) -> float:
    a=pd.to_numeric(reg,errors="coerce").dropna().to_numpy(float)
    b=pd.to_numeric(non,errors="coerce").dropna().to_numpy(float)
    if len(a)<2 or len(b)<2:
        return float("nan")
    rng=np.random.default_rng(seed)
    diffs=np.empty(BOOT_N)
    for i in range(BOOT_N):
        diffs[i]=rng.choice(a,size=len(a),replace=True).mean()-rng.choice(b,size=len(b),replace=True).mean()
    return float(np.mean(diffs>0)) if direction=="positive" else float(np.mean(diffs<0))


def metric(pred: pd.Series, actual: pd.Series) -> dict:
    z=pd.DataFrame({"p":pd.to_numeric(pred,errors="coerce"),"a":pd.to_numeric(actual,errors="coerce")}).dropna()
    e=z.p-z.a
    return {"n":len(z),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),
            "bias":float(e.mean()),"corr":float(z.p.corr(z.a)) if len(z)>2 else np.nan,
            "tail100":int(e.abs().ge(100).sum())}


def evaluate_regime(x: pd.DataFrame, name: str) -> dict:
    eligible=x.loc[x["regime_history_complete"]].copy()
    if name=="PASS_FUNNEL_SHORT_INTERMEDIATE_VOLUME":
        flag="volume_regime"; event="volume_expected_event"; comp="attempt_resid"; comp_dir="positive"; ensemble_dir="negative"
    else:
        flag="efficiency_regime"; event="efficiency_expected_event"; comp="ypa_resid"; comp_dir="negative"; ensemble_dir="positive"
    r=eligible.loc[eligible[flag]].copy()
    c=eligible.loc[~eligible[flag]].copy()
    coverage=float(len(eligible)/len(x)) if len(x) else 0.0
    rr=float(r[event].mean()) if len(r) else np.nan
    cr=float(c[event].mean()) if len(c) else np.nan
    lift=rr-cr if np.isfinite(rr) and np.isfinite(cr) else np.nan
    ratio=(rr/cr) if np.isfinite(rr) and np.isfinite(cr) and cr>0 else (np.inf if np.isfinite(rr) and rr>0 and cr==0 else np.nan)
    rcomp=float(pd.to_numeric(r[comp],errors="coerce").mean()) if len(r) else np.nan
    ccomp=float(pd.to_numeric(c[comp],errors="coerce").mean()) if len(c) else np.nan
    rerr=float(r["ensemble_error"].mean()) if len(r) else np.nan
    cerr=float(c["ensemble_error"].mean()) if len(c) else np.nan
    boot=bootstrap_support(r[comp],c[comp],comp_dir,BOOT_SEED+(1 if name.startswith("PASS_") else 2))
    gates={
        "coverage_ge_90pct": coverage>=0.90,
        "regime_n_ge_15": len(r)>=15,
        "expected_events_ge_3": int(r[event].sum())>=3,
        "event_enrichment": bool(np.isfinite(lift) and lift>=0.05 and np.isfinite(ratio) and ratio>=1.50),
        "component_residual_direction": bool((rcomp>ccomp) if comp_dir=="positive" else (rcomp<ccomp)),
        "ensemble_error_direction": bool((rerr<cerr) if ensemble_dir=="negative" else (rerr>cerr)),
        "bootstrap_support_ge_70pct": bool(np.isfinite(boot) and boot>=0.70),
    }
    return {
        "regime":name,"stable_primary_n":len(x),"eligible_n":len(eligible),"feature_coverage":coverage,
        "regime_n":len(r),"nonregime_n":len(c),"expected_event_count":int(r[event].sum()),
        "nonregime_expected_event_count":int(c[event].sum()),"regime_expected_event_rate":rr,
        "nonregime_expected_event_rate":cr,"event_rate_lift":lift,"event_rate_ratio":ratio,
        "regime_component_residual_mean":rcomp,"nonregime_component_residual_mean":ccomp,
        "regime_ensemble_error_mean":rerr,"nonregime_ensemble_error_mean":cerr,
        "bootstrap_expected_component_direction":boot,**gates,
        "final_status":"REPLICATED_2023" if all(gates.values()) else ("INSUFFICIENT_2023_COVERAGE" if coverage<0.90 or len(r)<15 else "NOT_REPLICATED_2023"),
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    q=load_predictions(args.predictions)
    ens,weights=build_oos_ensemble(q)
    logs=load_logs(args.player_logs)
    stable=build_stable_cohort(ens,logs)
    pbp=load_pbp()
    off,defense=aggregate_team_games(pbp)
    stable=add_regime_history(stable,off,defense)
    events=aggregate_chaos_events(pbp.loc[pbp["season"].eq(SEASON)].copy())
    trace=add_confirmation_labels(stable,events)

    thresholds={
        "volume_def_pass_rate_faced_min":VOL_DEF_PASS_RATE_MIN,
        "volume_off_deep_attempt_rate_max":VOL_OFF_DEEP_RATE_MAX,
        "efficiency_def_success_rate_allowed_max":EFF_DEF_SUCCESS_MAX,
        "efficiency_def_ypa_allowed_max":EFF_DEF_YPA_MAX,
    }
    rows=[
        evaluate_regime(trace,"PASS_FUNNEL_SHORT_INTERMEDIATE_VOLUME"),
        evaluate_regime(trace,"EFFICIENCY_SUPPRESSION"),
    ]
    summary=pd.DataFrame(rows)

    scores=[]
    for name,col in [("current_mc","mc_proj"),("current_ml","ml_proj"),("current_state","state_proj"),("oos_ensemble","ensemble_proj")]:
        scores.append({"model":name,**metric(trace[col],trace["actual_pass_yards"])})
    scoreboard=pd.DataFrame(scores)

    statuses=summary.set_index("regime")["final_status"].to_dict()
    if any(v=="REPLICATED_2023" for v in statuses.values()):
        disposition="M87_REGIME_REPLICATION_CONFIRMED"
    elif any(v=="INSUFFICIENT_2023_COVERAGE" for v in statuses.values()):
        disposition="M88_SOURCE_OR_COHORT_FAILURE"
    else:
        disposition="M87_REGIMES_NOT_REPLICATED"
    decision={
        "migration":"M88","confirmation_season":2023,"prior_history_season":2022,
        "stable_primary_rows":len(trace),"thresholds":thresholds,
        "regime_statuses":statuses,"final_disposition":disposition,
        "mc_iterations":2000,"bootstrap_draws":BOOT_N,"bootstrap_seed":BOOT_SEED,
        "sportsbook_features_used":False,"pass_yard_correction_fit":False,
        "target_game_pbp_used_as_pregame_feature":False,"production_actionable":False,
        "next_predictive_migration_allowed":bool(disposition=="M87_REGIME_REPLICATION_CONFIRMED"),
    }

    trace.to_csv(args.out_dir/"m88_2023_stable_qb_regime_trace.csv",index=False)
    weights.to_csv(args.out_dir/"m88_2023_oos_ensemble_weights.csv",index=False)
    summary.to_csv(args.out_dir/"m88_2023_regime_replication.csv",index=False)
    scoreboard.to_csv(args.out_dir/"m88_2023_full_stack_scoreboard.csv",index=False)
    (args.out_dir/"m88_decision.json").write_text(json.dumps(decision,indent=2,default=str)+"\n")

    print("[m88_scoreboard]")
    print(scoreboard.to_string(index=False))
    print("[m88_replication]")
    print(summary.to_string(index=False))
    print("[m88_decision]")
    print(json.dumps(decision,indent=2,default=str))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
