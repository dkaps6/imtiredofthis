#!/usr/bin/env python3
"""Migration 70 — extreme QB YPA mechanism autopsy.

M70 is forensic discovery only. It starts from immutable
`qb_frontier_canonical_v1`; it MUST NOT rebuild M59-M65 and it does not fit a
new predictive model.

M69 showed that 47/96 Raw 100+ passing-yard catastrophes were efficiency/YPA
predominant (27 collapses, 20 explosions). M70 asks a narrower question we have
not yet answered: when the model misses YPA by enough to create a catastrophic
passing-yard miss, what physically produced that YPA outcome?

The target is decomposed backward from postgame PBP. Postgame fields are never
eligible as production inputs. Pregame QB baselines are built strictly from
prior regular-season games and are used only to distinguish an actual
performance shift from a contextual-model adjustment miss.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

TARGET_MECHANISMS = {"ypa_explosion", "ypa_collapse"}
MIN_QB_PRIOR_GAMES = 3
PRIOR_WINDOW = 8
PHYSICAL_DOMINANCE_SHARE = 0.45
CONTEXT_NEAR_BASELINE = 0.75
CONTEXT_MATERIAL_SHIFT = 1.25
REAL_SHIFT = 1.50
EXPLOSIVE_YARD_SHARE = 0.45
EXPLOSIVE_YARD_SHARE_DELTA = 0.15


def num(v):
    return pd.to_numeric(v, errors="coerce")


def canon(v):
    t = canon_team(v)
    return "WAS" if t == "WSH" else t


def to_pd(o):
    if isinstance(o, pd.DataFrame):
        return o.copy()
    if hasattr(o, "to_pandas"):
        return o.to_pandas()
    return pd.DataFrame(o)


def lower(df):
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def regular(x):
    q = x.copy()
    c = "season_type" if "season_type" in q else "game_type" if "game_type" in q else None
    if c:
        s = q[c].astype(str).str.upper()
        r = q[s.isin(["REG", "REGULAR", "RS", ""])].copy()
        if len(r):
            q = r
    return q


def safe_div(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def load_pbp(seasons):
    import nflreadpy as nfl
    frames, manifest = [], []
    for season in seasons:
        try:
            q = lower(to_pd(nfl.load_pbp(seasons=[int(season)])))
            q = regular(q)
            if len(q):
                frames.append(q)
            manifest.append({"season": int(season), "family": "pbp", "status": "recovered" if len(q) else "empty", "rows": int(len(q))})
        except Exception as exc:
            manifest.append({"season": int(season), "family": "pbp", "status": f"failed:{type(exc).__name__}", "rows": 0})
    if not frames:
        raise RuntimeError("M70 requires historical PBP")
    return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(manifest)


def ensure(x, cols):
    for c in cols:
        if c not in x.columns:
            x[c] = np.nan
    return x


def build_passer_games(pbp):
    x = pbp.copy()
    x = ensure(x, [
        "season","week","game_id","posteam","defteam","passer_player_id","passer_player_name",
        "pass_attempt","complete_pass","interception","passing_yards","air_yards","yards_after_catch",
        "cpoe","epa","qb_hit","was_pressure","shotgun","no_huddle","play_action",
    ])
    x["season"] = num(x.season)
    x["week"] = num(x.week)
    x["team"] = x.posteam.map(canon)
    x["opponent"] = x.defteam.map(canon)
    x["passer_id"] = x.passer_player_id.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    x["passer_name"] = x.passer_player_name.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    x["_pa"] = num(x.pass_attempt).fillna(0).eq(1)
    x = x[x._pa & x.team.ne("") & x.passer_id.ne("")].copy()
    if x.empty:
        raise RuntimeError("M70 PBP produced no passer attempts")

    for c in ["complete_pass","interception","passing_yards","air_yards","yards_after_catch","cpoe","epa","qb_hit","was_pressure","shotgun","no_huddle","play_action"]:
        x[f"_{c}"] = num(x[c])
    x["_complete"] = x._complete_pass.fillna(0).eq(1)
    x["_pass_yards"] = x._passing_yards.fillna(0)
    x["_completed_air"] = np.where(x._complete, x._air_yards.fillna(0), 0.0)
    x["_yac"] = np.where(x._complete, x._yards_after_catch.fillna(0), 0.0)
    x["_exp10"] = (x._complete & x._pass_yards.ge(10)).astype(int)
    x["_exp20"] = (x._complete & x._pass_yards.ge(20)).astype(int)
    x["_exp30"] = (x._complete & x._pass_yards.ge(30)).astype(int)
    x["_exp40"] = (x._complete & x._pass_yards.ge(40)).astype(int)
    x["_exp20_yards"] = np.where(x._complete & x._pass_yards.ge(20), x._pass_yards, 0.0)
    x["_deep15"] = x._air_yards.ge(15).astype(float)
    x["_deep20"] = x._air_yards.ge(20).astype(float)

    rows = []
    keys = ["season","week","game_id","team","opponent","passer_id"]
    for key, g in x.groupby(keys, sort=True, dropna=False):
        season, week, game_id, team, opponent, passer_id = key
        attempts = float(len(g))
        completions = float(g._complete.sum())
        pass_yards = float(g._pass_yards.sum())
        completed_air = float(g._completed_air.sum())
        yac = float(g._yac.sum())
        exp20_yards = float(g._exp20_yards.sum())
        rec = {
            "season": int(season), "week": int(week), "game_id": str(game_id), "team": canon(team), "opponent": canon(opponent),
            "passer_id": str(passer_id), "passer_name": next((v for v in g.passer_name if str(v)), ""),
            "pbp_attempts": attempts, "pbp_completions": completions, "pbp_pass_yards": pass_yards,
            "pbp_ypa": safe_div(pass_yards, attempts), "completion_rate": safe_div(completions, attempts),
            "yards_per_completion": safe_div(pass_yards, completions),
            "completed_air_yards": completed_air, "yac_yards": yac,
            "completed_air_per_att": safe_div(completed_air, attempts), "yac_per_att": safe_div(yac, attempts),
            "air_per_completion": safe_div(completed_air, completions), "yac_per_completion": safe_div(yac, completions),
            "adot": float(g._air_yards.mean()) if g._air_yards.notna().any() else np.nan,
            "deep15_attempt_rate": float(g._deep15.mean()) if g._air_yards.notna().any() else np.nan,
            "deep20_attempt_rate": float(g._deep20.mean()) if g._air_yards.notna().any() else np.nan,
            "explosive10_rate": float(g._exp10.mean()), "explosive20_rate": float(g._exp20.mean()),
            "explosive30_rate": float(g._exp30.mean()), "explosive40_rate": float(g._exp40.mean()),
            "explosive20_yard_share": safe_div(exp20_yards, pass_yards) if pass_yards > 0 else np.nan,
            "mean_cpoe": float(g._cpoe.mean()) if g._cpoe.notna().any() else np.nan,
            "pass_epa_per_att": float(g._epa.mean()) if g._epa.notna().any() else np.nan,
            "qb_hit_rate": float(g._qb_hit.mean()) if g._qb_hit.notna().any() else np.nan,
            "pressure_rate": float(g._was_pressure.mean()) if g._was_pressure.notna().any() else np.nan,
            "shotgun_rate": float(g._shotgun.mean()) if g._shotgun.notna().any() else np.nan,
            "no_huddle_rate": float(g._no_huddle.mean()) if g._no_huddle.notna().any() else np.nan,
            "play_action_rate": float(g._play_action.mean()) if g._play_action.notna().any() else np.nan,
            "interception_rate": float(g._interception.fillna(0).mean()),
        }
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["season","week","team","pbp_attempts"], ascending=[True,True,True,False]).reset_index(drop=True)
    return out


def match_targets(canonical, passer_games):
    rows = []
    for _, r in canonical.iterrows():
        candidates = passer_games[(passer_games.season.eq(int(r.season))) & (passer_games.week.eq(int(r.week))) & (passer_games.team.eq(canon(r.team)))].copy()
        if "game_id" in r and pd.notna(r.game_id):
            exact_game = candidates[candidates.game_id.astype(str).eq(str(r.game_id))]
            if len(exact_game):
                candidates = exact_game
        if candidates.empty:
            rec = r.to_dict(); rec.update({"passer_match_status": "missing", "passer_id": "", "passer_name": "", "passer_attempt_diff": np.nan}); rows.append(rec); continue
        candidates["_diff"] = (num(candidates.pbp_attempts) - float(r.actual_attempts)).abs()
        q = candidates.sort_values(["_diff","pbp_attempts"], ascending=[True,False]).iloc[0]
        rec = r.to_dict()
        for c in passer_games.columns:
            rec[c] = q[c]
        rec["passer_attempt_diff"] = float(q._diff)
        rec["passer_match_status"] = "exact" if q._diff == 0 else "within2" if q._diff <= 2 else "mismatch"
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_prior(games):
    attempts = float(games.pbp_attempts.sum())
    completions = float(games.pbp_completions.sum())
    pass_yards = float(games.pbp_pass_yards.sum())
    completed_air = float(games.completed_air_yards.sum())
    yac = float(games.yac_yards.sum())
    def wavg(col, weight="pbp_attempts"):
        z = games[[col, weight]].copy(); z[col]=num(z[col]); z[weight]=num(z[weight]); z=z.dropna()
        return float(np.average(z[col], weights=z[weight])) if len(z) and z[weight].sum() > 0 else np.nan
    return {
        "prior_games": int(len(games)), "prior_attempts": attempts,
        "prior_ypa": safe_div(pass_yards, attempts), "prior_completion_rate": safe_div(completions, attempts),
        "prior_yards_per_completion": safe_div(pass_yards, completions),
        "prior_air_per_completion": safe_div(completed_air, completions), "prior_yac_per_completion": safe_div(yac, completions),
        "prior_completed_air_per_att": safe_div(completed_air, attempts), "prior_yac_per_att": safe_div(yac, attempts),
        "prior_adot": wavg("adot"), "prior_deep15_attempt_rate": wavg("deep15_attempt_rate"), "prior_deep20_attempt_rate": wavg("deep20_attempt_rate"),
        "prior_explosive20_rate": wavg("explosive20_rate"), "prior_explosive20_yard_share": wavg("explosive20_yard_share"),
        "prior_mean_cpoe": wavg("mean_cpoe"), "prior_pressure_rate": wavg("pressure_rate"),
    }


def attach_prior_baselines(targets, all_games):
    chronology = all_games.sort_values(["season","week"]).copy()
    rows=[]
    for _, r in targets.iterrows():
        rec = r.to_dict()
        pid = str(r.get("passer_id", ""))
        prior = chronology[(chronology.passer_id.astype(str).eq(pid)) & ((chronology.season < int(r.season)) | ((chronology.season == int(r.season)) & (chronology.week < int(r.week))))].tail(PRIOR_WINDOW)
        rec.update(aggregate_prior(prior) if len(prior) else {"prior_games": 0})
        rows.append(rec)
    return pd.DataFrame(rows)


def classify_forecast_failure(r):
    if int(r.get("prior_games", 0) or 0) < MIN_QB_PRIOR_GAMES or not np.isfinite(r.get("prior_ypa", np.nan)):
        return "insufficient_qb_history"
    actual_shift = r.actual_ypa - r.prior_ypa
    pred_shift = r.pred_ypa - r.prior_ypa
    if abs(actual_shift) < CONTEXT_NEAR_BASELINE and abs(pred_shift) >= CONTEXT_MATERIAL_SHIFT:
        return "model_context_overadjustment"
    if abs(actual_shift) >= CONTEXT_NEAR_BASELINE and abs(pred_shift) >= CONTEXT_NEAR_BASELINE and np.sign(actual_shift) != np.sign(pred_shift):
        return "model_context_wrong_direction"
    if abs(actual_shift) >= REAL_SHIFT and abs(pred_shift) < CONTEXT_NEAR_BASELINE:
        return "model_underreacted_to_real_shift"
    if abs(actual_shift) >= REAL_SHIFT:
        return "model_partially_missed_real_shift"
    return "mixed_model_and_performance_shift"


def decompose_physical(r):
    needed = ["completion_rate","yards_per_completion","prior_completion_rate","prior_yards_per_completion","air_per_completion","yac_per_completion","prior_air_per_completion","prior_yac_per_completion"]
    if int(r.get("prior_games", 0) or 0) < MIN_QB_PRIOR_GAMES or any(not np.isfinite(r.get(c, np.nan)) for c in needed):
        return {"physical_driver": "insufficient_qb_history", "physical_dominance_share": np.nan}
    cr, cr0 = r.completion_rate, r.prior_completion_rate
    ypc, ypc0 = r.yards_per_completion, r.prior_yards_per_completion
    air, air0 = r.air_per_completion, r.prior_air_per_completion
    yac, yac0 = r.yac_per_completion, r.prior_yac_per_completion
    completion = (cr-cr0) * ypc0
    airc = cr0 * (air-air0)
    yacc = cr0 * (yac-yac0)
    interaction = (cr-cr0) * (ypc-ypc0)
    split_rem = cr0 * (ypc-ypc0) - airc - yacc
    total = completion + airc + yacc + interaction + split_rem
    expected = r.pbp_ypa - r.prior_ypa
    roundoff = expected-total
    comps = {
        "completion_rate": completion,
        "completed_air_depth": airc,
        "yac": yacc,
        "completion_x_ypc_interaction": interaction,
        "pbp_split_remainder": split_rem,
    }
    denom = sum(abs(v) for v in comps.values() if np.isfinite(v))
    driver, maxv = max(comps.items(), key=lambda kv: abs(kv[1])) if denom else ("unresolved", np.nan)
    dom = abs(maxv)/denom if denom else np.nan
    if np.isfinite(dom) and dom < PHYSICAL_DOMINANCE_SHARE:
        driver = "mixed_physical"
    return {
        "completion_rate_ypa_contribution": completion,
        "completed_air_ypa_contribution": airc,
        "yac_ypa_contribution": yacc,
        "completion_x_ypc_interaction": interaction,
        "pbp_split_remainder": split_rem,
        "historical_ypa_delta_decomposition": total,
        "physical_decomposition_roundoff": roundoff,
        "physical_driver": driver,
        "physical_dominance_share": dom,
    }


def add_deltas(x):
    pairs = [
        ("adot","prior_adot"),("deep15_attempt_rate","prior_deep15_attempt_rate"),("deep20_attempt_rate","prior_deep20_attempt_rate"),
        ("explosive20_rate","prior_explosive20_rate"),("explosive20_yard_share","prior_explosive20_yard_share"),
        ("mean_cpoe","prior_mean_cpoe"),("pressure_rate","prior_pressure_rate"),
        ("completion_rate","prior_completion_rate"),("yards_per_completion","prior_yards_per_completion"),
        ("air_per_completion","prior_air_per_completion"),("yac_per_completion","prior_yac_per_completion"),
    ]
    for a,b in pairs:
        x[f"{a}_delta_vs_prior8"] = num(x.get(a,np.nan)) - num(x.get(b,np.nan))
    x["actual_shift_vs_prior8"] = num(x.actual_ypa) - num(x.prior_ypa)
    x["pred_shift_vs_prior8"] = num(x.pred_ypa) - num(x.prior_ypa)
    x["explosive_concentrated"] = (
        num(x.explosive20_yard_share).ge(EXPLOSIVE_YARD_SHARE)
        & num(x.explosive20_yard_share_delta_vs_prior8).ge(EXPLOSIVE_YARD_SHARE_DELTA)
    )
    return x


def summaries(targets):
    rows=[]
    for slice_name, q in [("100plus_ypa", targets), ("75plus_ypa", None)]:
        if q is None: continue
        for season_label, g in [("combined",q), *[(str(s),q[q.season.eq(s)]) for s in sorted(q.season.unique())]]:
            for c in ["forecast_failure_type","physical_driver"]:
                for v,n in g[c].value_counts(dropna=False).items():
                    rows.append({"slice":slice_name,"season":season_label,"dimension":c,"value":v,"n":int(n),"share":float(n/len(g)) if len(g) else np.nan})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--canonical",type=Path,required=True)
    ap.add_argument("--history-seasons",default="2023,2024,2025")
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    history=[int(v) for v in a.history_seasons.split(",") if v.strip()]

    base=lower(pd.read_csv(a.canonical,low_memory=False))
    if len(base)!=643 or int(base.cat100.sum())!=96:
        raise RuntimeError("M70 canonical v1 invariant failed")
    base["team"]=base.team.map(canon); base["opponent"]=base.opponent.map(canon)
    target=base[base.cat100.astype(bool) & base.mechanism.isin(TARGET_MECHANISMS)].copy()
    if len(target)!=47:
        raise RuntimeError(f"M70 frozen target must be 47 100+ YPA catastrophes; got {len(target)}")

    pbp,manifest=load_pbp(history)
    games=build_passer_games(pbp)
    matched=match_targets(base,games)
    match_cov=float(matched.passer_match_status.isin(["exact","within2"]).mean())
    if match_cov < .98:
        raise RuntimeError(f"M70 passer PBP match coverage too low: {match_cov:.3f}")
    full=attach_prior_baselines(matched,games)
    full=add_deltas(full)
    physical=pd.DataFrame([decompose_physical(r) for _,r in full.iterrows()],index=full.index)
    full=pd.concat([full.reset_index(drop=True),physical.reset_index(drop=True)],axis=1)
    full["forecast_failure_type"]=[classify_forecast_failure(r) for _,r in full.iterrows()]

    target=full[full.cat100.astype(bool) & full.mechanism.isin(TARGET_MECHANISMS)].copy()
    tail75=full[full.cat75.astype(bool) & full.mechanism.isin(TARGET_MECHANISMS)].copy()
    max_round=float(num(target.physical_decomposition_roundoff).abs().max())
    if max_round > 1e-6:
        raise RuntimeError(f"M70 physical decomposition failed: {max_round}")

    fail_counts=target.forecast_failure_type.value_counts()
    context_fail=int(fail_counts.get("model_context_overadjustment",0)+fail_counts.get("model_context_wrong_direction",0))
    context_share=context_fail/len(target)
    driver_counts=target.physical_driver.value_counts()
    top_driver=str(driver_counts.index[0]) if len(driver_counts) else ""
    top_driver_share=float(driver_counts.iloc[0]/len(target)) if len(driver_counts) else np.nan
    replicated=False
    if top_driver:
        shares=[]
        for s in [2024,2025]:
            g=target[target.season.eq(s)]
            shares.append(float(g.physical_driver.eq(top_driver).mean()) if len(g) else 0.0)
        replicated=top_driver_share>=.50 and min(shares)>=.35
    if context_share>=.50 and min(float(target[target.season.eq(s)].forecast_failure_type.isin(["model_context_overadjustment","model_context_wrong_direction"]).mean()) for s in [2024,2025])>=.40:
        interpretation="m70_contextual_ypa_adjustment_primary_extreme_failure"
    elif replicated:
        interpretation="m70_single_recurrent_physical_ypa_mechanism"
    else:
        interpretation="m70_extreme_ypa_multi_mechanism_seek_specific_new_information"

    out=a.out_dir; out.mkdir(parents=True,exist_ok=True)
    target.to_csv(out/"m70_extreme_ypa_game_autopsy.csv",index=False)
    tail75.to_csv(out/"m70_75plus_ypa_secondary_autopsy.csv",index=False)
    summaries(target).to_csv(out/"m70_mechanism_summary.csv",index=False)
    pd.DataFrame([{
        "target_games":len(target),"ypa_collapse_games":int(target.mechanism.eq("ypa_collapse").sum()),"ypa_explosion_games":int(target.mechanism.eq("ypa_explosion").sum()),
        "prior_history_eligible_games":int(target.prior_games.ge(MIN_QB_PRIOR_GAMES).sum()),"context_adjustment_failure_games":context_fail,"context_adjustment_failure_share":context_share,
        "top_physical_driver":top_driver,"top_physical_driver_share":top_driver_share,"top_driver_replicated":replicated,
        "explosive_concentrated_games":int(target.explosive_concentrated.sum()),"explosive_concentrated_share":float(target.explosive_concentrated.mean()),
        "max_physical_decomposition_roundoff":max_round,"m70_interpretation":interpretation,"production_actionable":False,
    }]).to_csv(out/"m70_precommitted_interpretation.csv",index=False)
    manifest.to_csv(out/"m70_source_manifest.csv",index=False)

    print("=== M70 INTERPRETATION ===")
    print(pd.read_csv(out/"m70_precommitted_interpretation.csv").to_string(index=False))
    print("=== M70 FORECAST FAILURE TYPES ===")
    print(target.forecast_failure_type.value_counts().to_string())
    print("=== M70 PHYSICAL DRIVERS ===")
    print(target.physical_driver.value_counts().to_string())
    print("=== M70 BY YEAR ===")
    print(pd.crosstab(target.season,target.physical_driver,normalize="index").round(3).to_string())
    return 0


if __name__=="__main__":
    raise SystemExit(main())
