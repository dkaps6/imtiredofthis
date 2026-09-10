#!/usr/bin/env python3
"""R27C2 retrospective target-quality decomposition.

Diagnostic only. Target-game PBP is used strictly as postgame explanatory labels.
No predictive model is fit and no candidate projection is created.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.pbp import get_pbp

RB_POS = {"RB", "FB", "HB", "TB"}
PHYSICAL = [
    "pbp_targets",
    "pbp_receptions",
    "pbp_rec_yards",
    "actual_catch_rate_pbp",
    "actual_air_yards_per_target",
    "actual_yac_per_reception",
    "actual_screen_target_rate",
    "actual_explosive20_target_rate",
    "actual_ypr_pbp",
    "actual_ypt_pbp",
    "explosive20_yard_share",
    "max_receiving_gain",
]


def _key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def _nonblank(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().ne("")


def _load_weekly(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    return _normalize_weekly(_to_pandas(raw), int(season))


def load_rb_target_games(season: int) -> tuple[pd.DataFrame, dict]:
    p = get_pbp(int(season), min_rows=1).copy()
    p.columns = [str(c).strip().lower() for c in p.columns]
    if "season_type" in p.columns:
        reg = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            p = reg

    receiver_name_col = _col(p, ("receiver_player_name", "receiver_name", "receiver"))
    receiver_id_col = _col(p, ("receiver_player_id", "receiver_id"))
    required = ["week", "posteam", "complete_pass", "yards_gained", "air_yards", "yards_after_catch"]
    missing = [c for c in required if c not in p.columns]
    if receiver_name_col is None and receiver_id_col is None:
        missing.append("receiver_identity")
    if missing:
        raise RuntimeError(f"season {season}: PBP missing {missing}")

    p["season"] = int(season)
    p["week"] = pd.to_numeric(p["week"], errors="coerce")
    p["team"] = p["posteam"].map(canon_team)
    p["receiver_player_id_norm"] = p[receiver_id_col].astype("string").fillna("").str.strip() if receiver_id_col else ""
    p["receiver_name_norm"] = p[receiver_name_col].astype("string").fillna("").str.strip() if receiver_name_col else ""
    p["receiver_name_key"] = p["receiver_name_norm"].map(_key)

    targeted = (_nonblank(p["receiver_player_id_norm"]) | _nonblank(p["receiver_name_norm"])) & p["week"].notna() & p["team"].ne("")
    t = p.loc[targeted].copy()

    weekly = _load_weekly(int(season)).copy()
    weekly["week"] = pd.to_numeric(weekly["week"], errors="coerce")
    weekly["team"] = weekly["team"].map(canon_team)
    weekly["player_id_norm"] = weekly.get("player_id", "").astype("string").fillna("").str.strip()
    weekly["player_clean_key"] = weekly.get("player_clean_key", weekly.get("player", "")).astype("string").fillna("").map(_key)
    weekly["position"] = weekly.get("position", "").astype("string").fillna("").str.upper().str.strip()

    id_map = weekly.loc[_nonblank(weekly["player_id_norm"]), ["week", "team", "player_id_norm", "player_clean_key", "position"]].drop_duplicates(["week", "team", "player_id_norm"])
    id_map = id_map.rename(columns={"player_id_norm":"receiver_player_id_norm", "player_clean_key":"id_player_key", "position":"position_by_id"})
    name_map = weekly.loc[_nonblank(weekly["player_clean_key"]), ["week", "team", "player_clean_key", "position"]].drop_duplicates(["week", "team", "player_clean_key"])
    name_map = name_map.rename(columns={"player_clean_key":"receiver_name_key", "position":"position_by_name"})

    t = t.merge(id_map, on=["week","team","receiver_player_id_norm"], how="left", validate="many_to_one")
    t = t.merge(name_map, on=["week","team","receiver_name_key"], how="left", validate="many_to_one")
    t["player_clean_key"] = t["id_player_key"].replace("", pd.NA).combine_first(t["receiver_name_key"].replace("", pd.NA)).fillna("")
    t["receiver_position"] = t["position_by_id"].replace("", pd.NA).combine_first(t["position_by_name"])
    rb = t.loc[t["receiver_position"].fillna("").astype(str).str.upper().isin(RB_POS) & t["player_clean_key"].ne("")].copy()

    rb["complete_num"] = pd.to_numeric(rb["complete_pass"], errors="coerce").fillna(0.0)
    rb["yards_num"] = pd.to_numeric(rb["yards_gained"], errors="coerce").fillna(0.0)
    rb["air_num"] = pd.to_numeric(rb["air_yards"], errors="coerce")
    rb["yac_num"] = pd.to_numeric(rb["yards_after_catch"], errors="coerce")
    rb["screen_flag"] = np.where(rb["air_num"].notna(), rb["air_num"].le(0).astype(float), np.nan)
    rb["explosive20_flag"] = (rb["complete_num"].eq(1) & rb["yards_num"].ge(20)).astype(float)
    rb["explosive20_yards"] = np.where(rb["explosive20_flag"].eq(1), rb["yards_num"], 0.0)
    rb["completed_yac"] = np.where(rb["complete_num"].eq(1), rb["yac_num"], np.nan)
    rb["completed_gain"] = np.where(rb["complete_num"].eq(1), rb["yards_num"], np.nan)

    keys=["season","week","team","player_clean_key"]
    g=rb.groupby(keys, dropna=False)
    out=g.agg(
        pbp_targets=("player_clean_key","size"),
        pbp_receptions=("complete_num","sum"),
        pbp_rec_yards=("yards_num","sum"),
        actual_air_yards_per_target=("air_num","mean"),
        actual_yac_per_reception=("completed_yac","mean"),
        actual_screen_target_rate=("screen_flag","mean"),
        actual_explosive20_target_rate=("explosive20_flag","mean"),
        explosive20_yards=("explosive20_yards","sum"),
        max_receiving_gain=("completed_gain","max"),
    ).reset_index()
    out["actual_catch_rate_pbp"] = np.where(out.pbp_targets.gt(0), out.pbp_receptions/out.pbp_targets, np.nan)
    out["actual_ypr_pbp"] = np.where(out.pbp_receptions.gt(0), out.pbp_rec_yards/out.pbp_receptions, np.nan)
    out["actual_ypt_pbp"] = np.where(out.pbp_targets.gt(0), out.pbp_rec_yards/out.pbp_targets, np.nan)
    out["explosive20_yard_share"] = np.where(out.pbp_rec_yards.gt(0), out.explosive20_yards/out.pbp_rec_yards, np.nan)

    audit={
        "season":int(season),
        "pbp_rows":int(len(p)),
        "target_rows":int(len(t)),
        "rb_target_rows":int(len(rb)),
        "rb_player_games":int(len(out)),
        "air_yards_nonnull_rate":float(rb.air_num.notna().mean()) if len(rb) else np.nan,
        "yac_nonnull_completed_rate":float(rb.loc[rb.complete_num.eq(1),"yac_num"].notna().mean()) if rb.complete_num.eq(1).any() else np.nan,
        "receiver_position_resolved_rate":float(t.receiver_position.notna().mean()) if len(t) else np.nan,
    }
    return out, audit


def cohort_masks(df: pd.DataFrame) -> dict[str,pd.Series]:
    vac=df.vacancy_active.eq(1)
    rb1=vac & df.vacancy_incumbent.eq(1) & df.role.eq("RB1")
    rb2=vac & df.vacancy_incumbent.eq(1) & df.role.eq("RB2+")
    return {
        "VACANCY_RB1_INCUMBENT":rb1,
        "VACANCY_RB2PLUS_INCUMBENT":rb2,
        "2023_VACANCY_RB1_INCUMBENT":rb1 & df.season.eq(2023),
        "NON2023_VACANCY_RB1_INCUMBENT":rb1 & df.season.ne(2023),
        "2023_VACANCY_ACTIVE":vac & df.season.eq(2023),
        "NON2023_VACANCY_ACTIVE":vac & df.season.ne(2023),
        "V2_INTO_30PLUS_VACANCY_RB1":rb1 & df.b1_ae.lt(30) & df.c1_ae.ge(30),
        "V2_OUT_OF_30PLUS_VACANCY_RB1":rb1 & df.b1_ae.ge(30) & df.c1_ae.lt(30),
        "V2_BOTH_30PLUS_VACANCY_RB1":rb1 & df.b1_ae.ge(30) & df.c1_ae.ge(30),
    }


def dist_rows(df: pd.DataFrame, masks: dict[str,pd.Series]) -> pd.DataFrame:
    rows=[]
    for cname,mask in masks.items():
        g=df.loc[mask].copy()
        for metric in PHYSICAL:
            s=pd.to_numeric(g[metric], errors="coerce").dropna()
            rows.append({
                "cohort":cname,"metric":metric,"n":int(len(s)),
                "mean":float(s.mean()) if len(s) else np.nan,
                "median":float(s.median()) if len(s) else np.nan,
                "p25":float(s.quantile(.25)) if len(s) else np.nan,
                "p75":float(s.quantile(.75)) if len(s) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seasons", default="2020-2025")
    ap.add_argument("--min-targeted-join-coverage", type=float, default=0.98)
    a=ap.parse_args()
    out_dir=Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pred=pd.read_csv(a.predictions)
    pred=pred.loc[pred.actual_rec_yards.notna()].copy()
    pred["season"]=pd.to_numeric(pred.season, errors="coerce").astype(int)
    pred["week"]=pd.to_numeric(pred.week, errors="coerce").astype(int)
    pred["team"]=pred.team.map(canon_team)
    pred["player_clean_key"]=pred.player_clean_key.astype(str).map(_key)
    for p in ["b1","c1"]:
        pred[f"{p}_ae"]=(pd.to_numeric(pred[f"{p}_rec_yards"], errors="coerce")-pd.to_numeric(pred.actual_rec_yards, errors="coerce")).abs()

    token=a.seasons.strip()
    if "-" in token and "," not in token:
        lo,hi=token.split("-",1); seasons=list(range(int(lo),int(hi)+1))
    else:
        seasons=[int(x) for x in token.split(",") if x.strip()]

    target_games=[]; source_audit=[]
    for season in seasons:
        tg,audit=load_rb_target_games(season)
        target_games.append(tg); source_audit.append(audit)
        print("[r27c2-source]", json.dumps(audit, sort_keys=True))
    tg=pd.concat(target_games, ignore_index=True)
    pd.DataFrame(source_audit).to_csv(out_dir/"r27c2_pbp_source_audit.csv", index=False)

    keys=["season","week","team","player_clean_key"]
    merged=pred.merge(tg, on=keys, how="left", validate="many_to_one", indicator="pbp_join")
    for c in ["pbp_targets","pbp_receptions","pbp_rec_yards"]:
        merged[c]=pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    targeted=merged.actual_targets.fillna(0).gt(0)
    targeted_join=(merged.pbp_join.eq("both") & merged.pbp_targets.gt(0))
    targeted_join_coverage=float(targeted_join[targeted].mean()) if targeted.any() else 1.0
    merged["target_gap_pbp_minus_preserved"]=merged.pbp_targets-pd.to_numeric(merged.actual_targets, errors="coerce").fillna(0)
    merged["reception_gap_pbp_minus_preserved"]=merged.pbp_receptions-pd.to_numeric(merged.actual_receptions, errors="coerce").fillna(0)
    merged["yard_gap_pbp_minus_preserved"]=merged.pbp_rec_yards-pd.to_numeric(merged.actual_rec_yards, errors="coerce").fillna(0)

    merged["production_implied_ypr"]=np.where(pd.to_numeric(merged.production_catch_rate,errors="coerce").gt(0), pd.to_numeric(merged.production_ypt,errors="coerce")/pd.to_numeric(merged.production_catch_rate,errors="coerce"), np.nan)
    merged["catch_rate_resid_vs_production"]=merged.actual_catch_rate_pbp-pd.to_numeric(merged.production_catch_rate,errors="coerce")
    merged["ypr_resid_vs_production_implied"]=merged.actual_ypr_pbp-merged.production_implied_ypr
    merged["ypt_resid_vs_production"]=merged.actual_ypt_pbp-pd.to_numeric(merged.production_ypt,errors="coerce")
    merged["air_yards_drift_vs_player_prior"]=merged.actual_air_yards_per_target-pd.to_numeric(merged.player_air_yards_per_target_prior,errors="coerce")
    merged["yac_drift_vs_player_prior"]=merged.actual_yac_per_reception-pd.to_numeric(merged.player_yac_per_reception_prior,errors="coerce")
    merged["screen_rate_drift_vs_player_prior"]=merged.actual_screen_target_rate-pd.to_numeric(merged.player_screen_target_rate_prior,errors="coerce")
    merged["explosive20_rate_drift_vs_player_prior"]=merged.actual_explosive20_target_rate-pd.to_numeric(merged.player_explosive20_target_rate_prior,errors="coerce")

    masks=cohort_masks(merged)
    dist=dist_rows(merged,masks)
    dist.to_csv(out_dir/"r27c2_target_quality_distributions.csv",index=False)

    # 2023-vs-non2023 RB1 differences on frozen physical metrics.
    p=dist.pivot(index="metric",columns="cohort",values="mean")
    diff=pd.DataFrame({
        "metric":PHYSICAL,
        "mean_2023_rb1":[p.loc[m,"2023_VACANCY_RB1_INCUMBENT"] for m in PHYSICAL],
        "mean_non2023_rb1":[p.loc[m,"NON2023_VACANCY_RB1_INCUMBENT"] for m in PHYSICAL],
    })
    diff["difference_2023_minus_non2023"]=diff.mean_2023_rb1-diff.mean_non2023_rb1
    diff.to_csv(out_dir/"r27c2_2023_vs_non2023_rb1_physical_diff.csv",index=False)

    decomposition=[]
    for cname,mask in masks.items():
        g=merged.loc[mask & merged.pbp_targets.gt(0)].copy()
        decomposition.append({
            "cohort":cname,
            "n_targeted":int(len(g)),
            "actual_catch_rate_mean":float(g.actual_catch_rate_pbp.mean()) if len(g) else np.nan,
            "production_catch_rate_mean":float(pd.to_numeric(g.production_catch_rate,errors="coerce").mean()) if len(g) else np.nan,
            "catch_rate_resid_mean":float(g.catch_rate_resid_vs_production.mean()) if len(g) else np.nan,
            "actual_ypr_mean":float(g.actual_ypr_pbp.mean()) if len(g) else np.nan,
            "production_implied_ypr_mean":float(g.production_implied_ypr.mean()) if len(g) else np.nan,
            "ypr_resid_mean":float(g.ypr_resid_vs_production_implied.mean()) if len(g) else np.nan,
            "actual_ypt_mean":float(g.actual_ypt_pbp.mean()) if len(g) else np.nan,
            "production_ypt_mean":float(pd.to_numeric(g.production_ypt,errors="coerce").mean()) if len(g) else np.nan,
            "ypt_resid_mean":float(g.ypt_resid_vs_production.mean()) if len(g) else np.nan,
        })
    pd.DataFrame(decomposition).to_csv(out_dir/"r27c2_catch_ypr_ypt_decomposition.csv",index=False)

    drift_metrics=["air_yards_drift_vs_player_prior","yac_drift_vs_player_prior","screen_rate_drift_vs_player_prior","explosive20_rate_drift_vs_player_prior"]
    drift=[]
    for cname in ["2023_VACANCY_RB1_INCUMBENT","NON2023_VACANCY_RB1_INCUMBENT"]:
        g=merged.loc[masks[cname] & merged.pbp_targets.gt(0)]
        for m in drift_metrics:
            s=pd.to_numeric(g[m],errors="coerce").dropna()
            drift.append({"cohort":cname,"metric":m,"n":int(len(s)),"mean":float(s.mean()) if len(s) else np.nan,"median":float(s.median()) if len(s) else np.nan,"p25":float(s.quantile(.25)) if len(s) else np.nan,"p75":float(s.quantile(.75)) if len(s) else np.nan})
    pd.DataFrame(drift).to_csv(out_dir/"r27c2_prior_to_realized_shape_drift.csv",index=False)

    tail=dist.loc[dist.cohort.isin(["VACANCY_RB1_INCUMBENT","V2_INTO_30PLUS_VACANCY_RB1","V2_OUT_OF_30PLUS_VACANCY_RB1","V2_BOTH_30PLUS_VACANCY_RB1"])].copy()
    tail.to_csv(out_dir/"r27c2_tail_target_quality.csv",index=False)

    coverage=[]
    for cname,mask in masks.items():
        g=merged.loc[mask]
        gt=g.actual_targets.fillna(0).gt(0)
        cov=float((g.loc[gt,"pbp_join"].eq("both") & g.loc[gt,"pbp_targets"].gt(0)).mean()) if gt.any() else 1.0
        coverage.append({
            "cohort":cname,"n":int(len(g)),"targeted_n":int(gt.sum()),"targeted_join_coverage":cov,
            "mean_abs_target_gap":float(g.target_gap_pbp_minus_preserved.abs().mean()) if len(g) else np.nan,
            "mean_abs_reception_gap":float(g.reception_gap_pbp_minus_preserved.abs().mean()) if len(g) else np.nan,
            "mean_abs_yard_gap":float(g.yard_gap_pbp_minus_preserved.abs().mean()) if len(g) else np.nan,
            "max_abs_target_gap":float(g.target_gap_pbp_minus_preserved.abs().max()) if len(g) else np.nan,
            "max_abs_reception_gap":float(g.reception_gap_pbp_minus_preserved.abs().max()) if len(g) else np.nan,
            "max_abs_yard_gap":float(g.yard_gap_pbp_minus_preserved.abs().max()) if len(g) else np.nan,
        })
    coverage_df=pd.DataFrame(coverage)
    coverage_df.to_csv(out_dir/"r27c2_join_coverage_and_stat_gaps.csv",index=False)

    primary_cov=coverage_df.loc[coverage_df.cohort.isin(["VACANCY_RB1_INCUMBENT","VACANCY_RB2PLUS_INCUMBENT","2023_VACANCY_RB1_INCUMBENT","NON2023_VACANCY_RB1_INCUMBENT"]),"targeted_join_coverage"]
    integrity=bool(len(pred)==8429 and len(primary_cov)==4 and primary_cov.ge(a.min_targeted_join_coverage).all())
    summary={
        "study":"RB_R27C2_REALIZED_TARGET_QUALITY_FORENSIC_V1",
        "diagnostic_only":True,
        "target_game_pbp_used_as_postgame_labels_only":True,
        "new_model_fit":False,
        "new_candidate_created":False,
        "sportsbook_inputs":0,
        "production_changed":False,
        "r26_changed":False,
        "r22_changed":False,
        "evaluable_parent_rows":int(len(pred)),
        "targeted_join_coverage_overall":targeted_join_coverage,
        "minimum_primary_targeted_join_coverage":float(primary_cov.min()) if len(primary_cov) else np.nan,
        "required_minimum_targeted_join_coverage":float(a.min_targeted_join_coverage),
        "integrity_pass":integrity,
        "machine_default_disposition":"R27C2_FORENSIC_COMPLETE_NO_SINGLE_PHYSICAL_MECHANISM_IDENTIFIED" if integrity else "R27C2_MECHANICAL_OR_IDENTITY_FAILURE_NO_FORENSIC_CONCLUSION",
    }
    (out_dir/"r27c2_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    print(json.dumps(summary,indent=2,sort_keys=True))
    if not integrity:
        raise RuntimeError("R27C2 identity coverage below frozen implementation threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
