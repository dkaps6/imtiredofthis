#!/usr/bin/env python3
"""PRISM Injury-Partial Receiving Exposure Decomposition V1.

Named-player receiving-role test for recent resolved injury-partial history.
Separates expected ordinary offensive participation from target earning during
actual participation. No fitted coefficients, sportsbook, 2026 outcomes, or
production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_INJURY_PARTIAL_RECEIVING_EXPOSURE_DECOMPOSITION_V1"
FREEZE_COMMIT = "653eb5e4e46a49f0b575c119259d2b11eb845380"
TARGET_SEASONS = (2024, 2025)
FAMILIES = {"RB", "WR", "TE"}
STATE_OK = "PARTICIPATION_NOT_COLLAPSED"
STATE_INJURY = "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED"
STATE_UNRESOLVED = "PARTICIPATION_COLLAPSE_UNRESOLVED"
STATE_INSUFFICIENT = "PARTICIPATION_BASELINE_INSUFFICIENT"
STATE_UNAVAILABLE = "PARTICIPATION_STATE_UNAVAILABLE"
WINDOW = 4
MIN_COMPARABLE = 3
N_BOOT = 5000
BOOT_SEED = 20261018
TOL = 1e-12

REQ = {
    "season","week","team","stable_id","family","offense_pct",
    "targets","team_targets","target_share_game",
    "participation_anomaly_state",
}


def read_source(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty source: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    miss=REQ-set(x.columns)
    if miss:
        raise RuntimeError(f"source missing {sorted(miss)}")

    for c in ("season","week","offense_pct","targets","team_targets","target_share_game"):
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x["team"]=x["team"].astype("string").fillna("").str.strip()
    x["stable_id"]=x["stable_id"].astype("string").fillna("").str.strip()
    x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x["participation_anomaly_state"]=(
        x["participation_anomaly_state"].astype("string").fillna("").str.strip()
    )

    x=x.loc[
        x["season"].isin(TARGET_SEASONS)
        & x["family"].isin(FAMILIES)
        & x["stable_id"].ne("")
        & x["team"].ne("")
    ].copy()
    x["season"]=x["season"].astype(int)
    x["week"]=x["week"].astype(int)

    keys=["season","week","team","stable_id"]
    if x.duplicated(keys).any():
        bad=x.loc[x.duplicated(keys,keep=False),keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate source identities: {bad}")

    return x.sort_values(
        ["season","team","stable_id","week"],kind="mergesort"
    ).reset_index(drop=True)


def build_scored_rows(x: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows=[]
    audit={
        "chronology_violations":0,
        "target_major_collapse_excluded":0,
        "target_noneligible_state_excluded":0,
        "prior_unresolved_excluded":0,
        "prior_state_unavailable_excluded":0,
        "prior_window_short_excluded":0,
        "no_injury_partial_history_excluded":0,
        "comparable_support_excluded":0,
        "nonfinite_source_excluded":0,
    }

    for (season,team,pid),g in x.groupby(
        ["season","team","stable_id"],sort=False
    ):
        g=g.sort_values("week",kind="mergesort").reset_index(drop=True)
        recs=g.to_dict("records")

        for i,target in enumerate(recs):
            tstate=str(target["participation_anomaly_state"])
            if tstate in {STATE_INJURY,STATE_UNRESOLVED}:
                audit["target_major_collapse_excluded"]+=1
                continue
            if tstate != STATE_OK:
                audit["target_noneligible_state_excluded"]+=1
                continue

            prior=recs[max(0,i-WINDOW):i]
            if len(prior)!=WINDOW:
                audit["prior_window_short_excluded"]+=1
                continue
            if not all(int(r["week"]) < int(target["week"]) for r in prior):
                audit["chronology_violations"]+=1
                continue

            states=[str(r["participation_anomaly_state"]) for r in prior]
            if STATE_UNRESOLVED in states:
                audit["prior_unresolved_excluded"]+=1
                continue
            if STATE_UNAVAILABLE in states or any(not s for s in states):
                audit["prior_state_unavailable_excluded"]+=1
                continue

            n_injury=states.count(STATE_INJURY)
            if n_injury < 1:
                audit["no_injury_partial_history_excluded"]+=1
                continue

            comparable=[
                r for r in prior
                if str(r["participation_anomaly_state"]) != STATE_INJURY
            ]
            if len(comparable) < MIN_COMPARABLE:
                audit["comparable_support_excluded"]+=1
                continue

            vals=[]
            finite=True
            for r in prior:
                tgt=float(r["targets"]) if pd.notna(r["targets"]) else np.nan
                team_t=float(r["team_targets"]) if pd.notna(r["team_targets"]) else np.nan
                pct=float(r["offense_pct"]) if pd.notna(r["offense_pct"]) else np.nan
                if not (np.isfinite(tgt) and np.isfinite(team_t) and team_t>0 and np.isfinite(pct) and pct>0):
                    finite=False
                    break
                vals.append((tgt,team_t,pct))
            t_actual=float(target["target_share_game"]) if pd.notna(target["target_share_game"]) else np.nan
            if not finite or not np.isfinite(t_actual):
                audit["nonfinite_source_excluded"]+=1
                continue

            total_targets=float(sum(v[0] for v in vals))
            total_team_targets=float(sum(v[1] for v in vals))
            exposure_den=float(sum(v[1]*v[2] for v in vals))
            if total_team_targets<=0 or exposure_den<=0:
                audit["nonfinite_source_excluded"]+=1
                continue

            baseline=float(total_targets/total_team_targets)
            comp_pcts=np.asarray([float(r["offense_pct"]) for r in comparable],dtype=float)
            expected_pct=float(np.median(comp_pcts))
            earning=float(total_targets/exposure_den)
            raw_candidate=float(expected_pct*earning)
            candidate=float(np.clip(raw_candidate,0.0,1.0))
            gain=float(abs(baseline-t_actual)-abs(candidate-t_actual))

            rows.append({
                "season":int(season),
                "week":int(target["week"]),
                "team":str(team),
                "stable_id":str(pid),
                "family":str(target["family"]),
                "actual_target_share":t_actual,
                "baseline_target_share":baseline,
                "expected_offense_pct":expected_pct,
                "earning_intensity":earning,
                "candidate_target_share_raw":raw_candidate,
                "candidate_target_share":candidate,
                "candidate_clipped":bool(abs(candidate-raw_candidate)>TOL),
                "player_game_closeness_gain":gain,
                "prior_window_rows":int(len(prior)),
                "prior_injury_partial_rows":int(n_injury),
                "prior_comparable_rows":int(len(comparable)),
                "prior_min_week":int(min(int(r["week"]) for r in prior)),
                "prior_max_week":int(max(int(r["week"]) for r in prior)),
            })

    out=pd.DataFrame(rows)
    if out.empty:
        return out,audit
    return out.sort_values(
        ["season","week","team","stable_id"],kind="mergesort"
    ).reset_index(drop=True),audit


def player_summary(s: pd.DataFrame) -> pd.Series:
    return s.groupby("stable_id")["player_game_closeness_gain"].mean().astype(float)


def bootstrap_players(s: pd.DataFrame) -> dict:
    p=player_summary(s)
    vals=p.to_numpy(float)
    if len(vals)<2:
        return {"reps":0,"ci_low":np.nan,"ci_high":np.nan}
    rng=np.random.default_rng(BOOT_SEED)
    boots=np.empty(N_BOOT,dtype=float)
    for i in range(N_BOOT):
        boots[i]=float(rng.choice(vals,size=len(vals),replace=True).mean())
    return {
        "reps":N_BOOT,
        "ci_low":float(np.quantile(boots,0.025)),
        "ci_high":float(np.quantile(boots,0.975)),
    }


def summarize(s: pd.DataFrame, season: int, bootstrap: bool) -> dict:
    z=s.loc[s["season"].eq(season)].copy()
    if z.empty:
        raise RuntimeError(f"zero rows for {season}")
    p=player_summary(z)
    fam=[]
    for family,g in z.groupby("family",sort=True):
        pg=player_summary(g)
        fam.append({
            "family":str(family),
            "target_games":int(len(g)),
            "named_players":int(len(pg)),
            "equal_player_average_closeness_gain":float(pg.mean()),
            "median_player_gain":float(pg.median()),
            "improved_player_share":float((pg>0).mean()),
            "tied_player_share":float(np.isclose(pg,0.0,atol=1e-15).mean()),
            "worsened_player_share":float((pg<0).mean()),
        })

    out={
        "season":int(season),
        "target_games":int(len(z)),
        "named_players":int(len(p)),
        "equal_player_average_closeness_gain":float(p.mean()),
        "median_player_gain":float(p.median()),
        "improved_player_share":float((p>0).mean()),
        "tied_player_share":float(np.isclose(p,0.0,atol=1e-15).mean()),
        "worsened_player_share":float((p<0).mean()),
        "p10_player_gain":float(p.quantile(0.10)),
        "p25_player_gain":float(p.quantile(0.25)),
        "p75_player_gain":float(p.quantile(0.75)),
        "p90_player_gain":float(p.quantile(0.90)),
        "candidate_clipped_rows":int(z["candidate_clipped"].sum()),
        "family_diagnostics":fam,
    }
    if bootstrap:
        out["player_bootstrap"]=bootstrap_players(z)
    return out


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    source=read_source(a.source)
    rows,audit=build_scored_rows(source)
    if rows.empty:
        raise RuntimeError("zero scored rows")
    dev=summarize(rows,2024,bootstrap=False)
    conf=summarize(rows,2025,bootstrap=True)

    gates={
        "chronology_violations_zero":audit["chronology_violations"]==0,
        "target_major_collapse_rows_zero":True,
        "prior_unresolved_rows_zero":True,
        "prior_window_exact_four_all":bool(rows["prior_window_rows"].eq(4).all()),
        "prior_comparable_ge_three_all":bool(rows["prior_comparable_rows"].ge(3).all()),
        "expected_offense_pct_finite_positive":bool(
            np.isfinite(rows["expected_offense_pct"]).all()
            and rows["expected_offense_pct"].gt(0).all()
        ),
        "earning_intensity_finite_nonnegative":bool(
            np.isfinite(rows["earning_intensity"]).all()
            and rows["earning_intensity"].ge(0).all()
        ),
        "candidate_share_finite_bounded":bool(
            np.isfinite(rows["candidate_target_share"]).all()
            and rows["candidate_target_share"].between(0,1).all()
        ),
        "development_target_games_ge_100":dev["target_games"]>=100,
        "development_named_players_ge_40":dev["named_players"]>=40,
        "development_equal_player_positive":dev["equal_player_average_closeness_gain"]>0,
        "development_median_player_nonnegative":dev["median_player_gain"]>=0,
        "development_improved_share_gt_worsened":dev["improved_player_share"]>dev["worsened_player_share"],
        "confirmation_target_games_ge_100":conf["target_games"]>=100,
        "confirmation_named_players_ge_40":conf["named_players"]>=40,
        "confirmation_equal_player_positive":conf["equal_player_average_closeness_gain"]>0,
        "confirmation_median_player_nonnegative":conf["median_player_gain"]>=0,
        "confirmation_improved_share_gt_worsened":conf["improved_player_share"]>conf["worsened_player_share"],
        "confirmation_player_bootstrap_ci_low_positive":conf["player_bootstrap"]["ci_low"]>0,
        "coefficients_fit_zero":True,
        "thresholds_searched_zero":True,
        "outcomes_2026_zero":True,
        "sportsbook_oddsapi_zero":True,
    }
    qualified=bool(all(gates.values()))

    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "source_run":39161711741,
        "source_artifact":12149181200,
        "source_digest":"sha256:6cae427b667db4a120a1308747d00b82b6770fe0adbc46d3efc68fce3e1db240",
        "disposition":(
            "PRISM_INJURY_PARTIAL_RECEIVING_EXPOSURE_DECOMPOSITION_V1_QUALIFIED"
            if qualified else
            "PRISM_INJURY_PARTIAL_RECEIVING_EXPOSURE_DECOMPOSITION_V1_CLOSED"
        ),
        "qualified":qualified,
        "audit":audit,
        "development_2024":dev,
        "confirmation_2025":conf,
        "gates":gates,
        "prior_window":WINDOW,
        "minimum_comparable_rows":MIN_COMPARABLE,
        "bootstrap_replicates":N_BOOT,
        "bootstrap_seed":BOOT_SEED,
        "routes_inferred_from_snaps":False,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }

    rows.to_csv(
        a.out_dir/"injury_partial_receiving_exposure_decomposition_rows.csv",
        index=False,float_format="%.12f"
    )
    (a.out_dir/"injury_partial_receiving_exposure_decomposition_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
