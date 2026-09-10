#!/usr/bin/env python3
"""R27D0 source/novelty audit for RB post-catch value information.

No predictive model is fit. No candidate receiving-yard projection is created.
Target-game outcomes are used only to audit source/schema coverage.
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


def num(x):
    return pd.to_numeric(x, errors="coerce")


def key_name(v) -> str:
    try:
        _, k = canonicalize_player_name_safe(v)
        if k:
            return str(k)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def first_col(df: pd.DataFrame, names) -> str | None:
    return next((c for c in names if c in df.columns), None)


def nonblank(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip().ne("")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    q = df.copy()
    q.columns = [str(c).strip().lower() for c in q.columns]
    return q


def to_pd(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def load_weekly(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    q = _normalize_weekly(_to_pandas(raw), int(season)).copy()
    q["week"] = num(q["week"])
    q["team"] = q["team"].map(canon_team)
    q["player_id_norm"] = q.get("player_id", "").astype("string").fillna("").str.strip()
    q["player_clean_key"] = q.get("player_clean_key", q.get("player", "")).astype("string").fillna("").map(key_name)
    q["position"] = q.get("position", "").astype("string").fillna("").str.upper().str.strip()
    return q


def resolve_pbp_rb(season: int) -> tuple[pd.DataFrame, dict]:
    p = get_pbp(int(season), min_rows=1).copy()
    p = lower(p)
    if "season_type" in p.columns:
        reg = p[p.season_type.fillna("").astype(str).str.upper().eq("REG")].copy()
        if len(reg):
            p = reg
    p["season"] = int(season)
    p["week"] = num(p.get("week"))
    p["team"] = p.get("posteam", "").map(canon_team)

    name_col = first_col(p, ["receiver_player_name", "receiver_name", "receiver"])
    id_col = first_col(p, ["receiver_player_id", "receiver_id"])
    if name_col is None and id_col is None:
        raise RuntimeError(f"{season}: receiver identity unavailable")
    p["receiver_id_norm"] = p[id_col].astype("string").fillna("").str.strip() if id_col else ""
    p["receiver_name_norm"] = p[name_col].astype("string").fillna("").str.strip() if name_col else ""
    p["receiver_name_key"] = p.receiver_name_norm.map(key_name)
    targeted = (nonblank(p.receiver_id_norm) | nonblank(p.receiver_name_norm)) & p.week.notna() & p.team.ne("")
    t = p[targeted].copy()

    weekly = load_weekly(int(season))
    id_map = weekly.loc[nonblank(weekly.player_id_norm), ["week","team","player_id_norm","player_clean_key","position"]].drop_duplicates(["week","team","player_id_norm"])
    id_map = id_map.rename(columns={"player_id_norm":"receiver_id_norm","player_clean_key":"id_player_key","position":"position_by_id"})
    name_map = weekly.loc[nonblank(weekly.player_clean_key), ["week","team","player_clean_key","position"]].drop_duplicates(["week","team","player_clean_key"])
    name_map = name_map.rename(columns={"player_clean_key":"receiver_name_key","position":"position_by_name"})
    t = t.merge(id_map, on=["week","team","receiver_id_norm"], how="left", validate="many_to_one")
    t = t.merge(name_map, on=["week","team","receiver_name_key"], how="left", validate="many_to_one")
    t["player_clean_key"] = t.id_player_key.replace("", pd.NA).combine_first(t.receiver_name_key.replace("", pd.NA)).fillna("")
    t["receiver_position"] = t.position_by_id.replace("", pd.NA).combine_first(t.position_by_name).fillna("")
    rb = t[t.receiver_position.astype(str).str.upper().isin(RB_POS) & t.player_clean_key.ne("")].copy()

    complete_col = first_col(rb, ["complete_pass"])
    yac_col = first_col(rb, ["yards_after_catch"])
    xyac_mean_col = first_col(rb, ["xyac_mean_yardage", "xyac_mean_yards"])
    xyac_median_col = first_col(rb, ["xyac_median_yardage", "xyac_median_yards"])
    xyac_success_col = first_col(rb, ["xyac_success"])
    xyac_fd_col = first_col(rb, ["xyac_fd"])
    air_col = first_col(rb, ["air_yards"])

    rb["complete_num"] = num(rb[complete_col]).fillna(0) if complete_col else 0.0
    rb["yac_num"] = num(rb[yac_col]) if yac_col else np.nan
    rb["xyac_mean_num"] = num(rb[xyac_mean_col]) if xyac_mean_col else np.nan
    rb["pbp_yacoe"] = rb.yac_num - rb.xyac_mean_num
    completed = rb.complete_num.eq(1)

    situational = {}
    situational_cols = {
        "down": first_col(rb, ["down"]),
        "ydstogo": first_col(rb, ["ydstogo", "yards_to_go"]),
        "shotgun": first_col(rb, ["shotgun"]),
        "no_huddle": first_col(rb, ["no_huddle"]),
        "pass_location": first_col(rb, ["pass_location"]),
        "pass_length": first_col(rb, ["pass_length"]),
        "score_differential": first_col(rb, ["score_differential", "posteam_score_differential"]),
        "air_yards": air_col,
    }
    for label, c in situational_cols.items():
        situational[f"{label}_nonnull_rate"] = float(rb[c].notna().mean()) if c and len(rb) else 0.0

    keys=["season","week","team","player_clean_key"]
    games = rb.groupby(keys, dropna=False).agg(
        pbp_targets=("player_clean_key","size"),
        pbp_receptions=("complete_num","sum"),
        pbp_yacoe_obs=("pbp_yacoe", lambda s: int(s.notna().sum())),
        pbp_yacoe_mean=("pbp_yacoe","mean"),
    ).reset_index()

    audit = {
        "season":int(season),
        "pbp_rows":int(len(p)),
        "target_rows":int(len(t)),
        "rb_target_rows":int(len(rb)),
        "rb_completed_targets":int(completed.sum()),
        "receiver_position_resolved_rate":float(t.receiver_position.ne("").mean()) if len(t) else 0.0,
        "yac_nonnull_completed_rate":float(rb.loc[completed,"yac_num"].notna().mean()) if completed.any() else 0.0,
        "xyac_mean_field":xyac_mean_col or "",
        "xyac_mean_nonnull_completed_rate":float(rb.loc[completed,"xyac_mean_num"].notna().mean()) if completed.any() else 0.0,
        "xyac_median_field":xyac_median_col or "",
        "xyac_median_nonnull_completed_rate":float(rb.loc[completed,xyac_median_col].notna().mean()) if xyac_median_col and completed.any() else 0.0,
        "xyac_success_field":xyac_success_col or "",
        "xyac_success_nonnull_completed_rate":float(rb.loc[completed,xyac_success_col].notna().mean()) if xyac_success_col and completed.any() else 0.0,
        "xyac_fd_field":xyac_fd_col or "",
        "xyac_fd_nonnull_completed_rate":float(rb.loc[completed,xyac_fd_col].notna().mean()) if xyac_fd_col and completed.any() else 0.0,
        **situational,
    }
    return games, audit


def load_ngs_rb(seasons: list[int]) -> tuple[pd.DataFrame, list[dict], dict]:
    import nflreadpy as nfl
    try:
        ngs = lower(to_pd(nfl.load_nextgen_stats(seasons=seasons, stat_type="receiving")))
    except Exception as e:
        return pd.DataFrame(), [], {"load_error":f"{type(e).__name__}:{e}"}
    if ngs.empty:
        return pd.DataFrame(), [], {"load_error":"empty"}
    if "season_type" in ngs.columns:
        reg=ngs[ngs.season_type.fillna("").astype(str).str.upper().isin(["REG","REGULAR","RS",""])].copy()
        if len(reg): ngs=reg

    season_col=first_col(ngs,["season"])
    week_col=first_col(ngs,["week"])
    team_col=first_col(ngs,["team_abbr","team"])
    name_col=first_col(ngs,["player_display_name","player_name","receiver_player_name","player"])
    id_col=first_col(ngs,["player_gsis_id","gsis_id","player_id"])
    pos_col=first_col(ngs,["player_position","position"])
    yacoe_col=first_col(ngs,["avg_yac_above_expectation"])
    eyac_col=first_col(ngs,["avg_expected_yac"])
    targets_col=first_col(ngs,["targets"])
    rec_col=first_col(ngs,["receptions"])
    sep_col=first_col(ngs,["avg_separation"])
    cushion_col=first_col(ngs,["avg_cushion"])
    adot_col=first_col(ngs,["avg_intended_air_yards","avg_air_distance"])
    required=[season_col,week_col,team_col,yacoe_col]
    if not all(required):
        return pd.DataFrame(), [], {"load_error":"missing_required_columns","columns":"|".join(ngs.columns)}

    ngs["season"]=num(ngs[season_col]).astype("Int64")
    ngs["week"]=num(ngs[week_col]).astype("Int64")
    ngs["team"]=ngs[team_col].map(canon_team)
    ngs["player_id_norm"]=ngs[id_col].astype("string").fillna("").str.strip() if id_col else ""
    ngs["player_name_key"]=ngs[name_col].astype("string").fillna("").map(key_name) if name_col else ""
    ngs["position_resolved"]=ngs[pos_col].astype("string").fillna("").str.upper().str.strip() if pos_col else ""

    # Resolve missing positions/names via nflreadpy player registry when possible.
    try:
        players=lower(to_pd(nfl.load_players()))
    except Exception:
        players=pd.DataFrame()
    if not players.empty:
        pid=first_col(players,["gsis_id","player_gsis_id","player_id"])
        ppos=first_col(players,["position","position_group"])
        pname=first_col(players,["display_name","full_name","player_name","football_name"])
        if pid:
            cols=[pid] + ([ppos] if ppos else []) + ([pname] if pname else [])
            bridge=players[cols].dropna(subset=[pid]).drop_duplicates(pid).copy()
            bridge["player_id_norm"]=bridge[pid].astype("string").str.strip()
            bridge["bridge_position"]=bridge[ppos].astype("string").fillna("").str.upper().str.strip() if ppos else ""
            bridge["bridge_name_key"]=bridge[pname].astype("string").fillna("").map(key_name) if pname else ""
            bridge=bridge[["player_id_norm","bridge_position","bridge_name_key"]]
            ngs=ngs.merge(bridge,on="player_id_norm",how="left",validate="many_to_one")
            ngs["position_resolved"]=ngs.position_resolved.replace("",pd.NA).combine_first(ngs.bridge_position).fillna("")
            ngs["player_name_key"]=ngs.player_name_key.replace("",pd.NA).combine_first(ngs.bridge_name_key).fillna("")

    rb=ngs[ngs.position_resolved.isin(RB_POS)].copy()
    rb["ngs_yacoe"]=num(rb[yacoe_col])
    rb["ngs_expected_yac"]=num(rb[eyac_col]) if eyac_col else np.nan
    rb["ngs_targets"]=num(rb[targets_col]) if targets_col else np.nan
    rb["ngs_receptions"]=num(rb[rec_col]) if rec_col else np.nan
    rb["ngs_separation"]=num(rb[sep_col]) if sep_col else np.nan
    rb["ngs_cushion"]=num(rb[cushion_col]) if cushion_col else np.nan
    rb["ngs_adot"]=num(rb[adot_col]) if adot_col else np.nan
    rb["player_clean_key"]=rb.player_name_key.astype(str)
    rb=rb[rb.player_clean_key.ne("") & rb.week.notna() & rb.season.notna() & rb.team.ne("")].copy()
    rb["season"]=rb.season.astype(int); rb["week"]=rb.week.astype(int)

    audits=[]
    for s in seasons:
        all_s=ngs[ngs.season.eq(s)]
        r=rb[rb.season.eq(s)]
        audits.append({
            "season":int(s),
            "ngs_receiving_rows":int(len(all_s)),
            "ngs_rb_rows":int(len(r)),
            "ngs_rb_yacoe_nonnull_rate":float(r.ngs_yacoe.notna().mean()) if len(r) else 0.0,
            "ngs_rb_expected_yac_nonnull_rate":float(r.ngs_expected_yac.notna().mean()) if len(r) else 0.0,
            "ngs_rb_unique_player_weeks_yacoe":int(r.loc[r.ngs_yacoe.notna(),["season","week","player_clean_key"]].drop_duplicates().shape[0]),
        })
    meta={
        "load_error":"",
        "yacoe_col":yacoe_col or "",
        "expected_yac_col":eyac_col or "",
        "targets_col":targets_col or "",
        "receptions_col":rec_col or "",
        "separation_col":sep_col or "",
        "cushion_col":cushion_col or "",
        "adot_col":adot_col or "",
        "native_position_col":pos_col or "",
        "name_col":name_col or "",
        "id_col":id_col or "",
    }
    return rb[["season","week","team","player_clean_key","ngs_yacoe","ngs_expected_yac","ngs_targets","ngs_receptions","ngs_separation","ngs_cushion","ngs_adot"]].copy(), audits, meta


def prior_count(hist: pd.DataFrame, player_key: str, season: int, week: int, value_col: str) -> int:
    if hist.empty or value_col not in hist.columns or not player_key:
        return 0
    g=hist[(hist.player_clean_key.eq(player_key)) & ((hist.season < season) | ((hist.season == season) & (hist.week < week))) & hist[value_col].notna()]
    return int(g[["season","week"]].drop_duplicates().shape[0])


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seasons", default="2020-2025")
    a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    if "-" in a.seasons and "," not in a.seasons:
        lo,hi=a.seasons.split("-",1); seasons=list(range(int(lo),int(hi)+1))
    else:
        seasons=[int(x) for x in a.seasons.split(",") if x.strip()]

    pred=pd.read_csv(a.predictions)
    pred=pred[pred.actual_rec_yards.notna()].copy()
    pred["season"]=num(pred.season).astype(int); pred["week"]=num(pred.week).astype(int)
    pred["team"]=pred.team.map(canon_team)
    pred["player_clean_key"]=pred.player_clean_key.astype(str).map(key_name)
    rb1=pred.vacancy_active.eq(1)&pred.vacancy_incumbent.eq(1)&pred.role.eq("RB1")&pred.actual_targets.fillna(0).gt(0)
    focus=pred[rb1].copy()

    pbp_games=[]; pbp_audits=[]
    for s in seasons:
        games,audit=resolve_pbp_rb(s)
        pbp_games.append(games); pbp_audits.append(audit)
        print("[R27D0-PBP]",json.dumps(audit,sort_keys=True))
    pbp_hist=pd.concat(pbp_games,ignore_index=True) if pbp_games else pd.DataFrame()
    pd.DataFrame(pbp_audits).to_csv(out/"r27d0_pbp_source_coverage.csv",index=False)

    ngs_hist,ngs_audits,ngs_meta=load_ngs_rb(seasons)
    pd.DataFrame(ngs_audits).to_csv(out/"r27d0_ngs_source_coverage.csv",index=False)
    (out/"r27d0_ngs_schema.json").write_text(json.dumps(ngs_meta,indent=2,sort_keys=True))
    print("[R27D0-NGS]",json.dumps(ngs_meta,sort_keys=True))

    counts=[]
    for _,r in focus.iterrows():
        pc=prior_count(pbp_hist,r.player_clean_key,int(r.season),int(r.week),"pbp_yacoe_mean")
        nc=prior_count(ngs_hist,r.player_clean_key,int(r.season),int(r.week),"ngs_yacoe")
        counts.append({
            "season":int(r.season),"week":int(r.week),"team":r.team,"player_clean_key":r.player_clean_key,
            "pbp_yacoe_prior_games":pc,"ngs_yacoe_prior_weeks":nc,
            "cohort":"2023_RB1" if int(r.season)==2023 else "NON2023_RB1",
        })
    cnt=pd.DataFrame(counts)
    cnt.to_csv(out/"r27d0_vacancy_rb1_prior_source_counts.csv",index=False)
    cov=[]
    for cname,g in [("VACANCY_RB1_TARGETED",cnt),("2023_VACANCY_RB1_TARGETED",cnt[cnt.cohort.eq("2023_RB1")]),("NON2023_VACANCY_RB1_TARGETED",cnt[cnt.cohort.eq("NON2023_RB1")])]:
        cov.append({
            "cohort":cname,"n":int(len(g)),
            "pbp_yacoe_prior_ge1_rate":float(g.pbp_yacoe_prior_games.ge(1).mean()) if len(g) else 0.0,
            "pbp_yacoe_prior_ge3_rate":float(g.pbp_yacoe_prior_games.ge(3).mean()) if len(g) else 0.0,
            "ngs_yacoe_prior_ge1_rate":float(g.ngs_yacoe_prior_weeks.ge(1).mean()) if len(g) else 0.0,
            "ngs_yacoe_prior_ge3_rate":float(g.ngs_yacoe_prior_weeks.ge(3).mean()) if len(g) else 0.0,
            "pbp_yacoe_prior_games_mean":float(g.pbp_yacoe_prior_games.mean()) if len(g) else 0.0,
            "ngs_yacoe_prior_weeks_mean":float(g.ngs_yacoe_prior_weeks.mean()) if len(g) else 0.0,
        })
    covdf=pd.DataFrame(cov); covdf.to_csv(out/"r27d0_vacancy_rb1_strict_prior_coverage.csv",index=False)

    pbpcov=pd.DataFrame(pbp_audits)
    ngscov=pd.DataFrame(ngs_audits)
    pbp_xyac_good=bool(len(pbpcov)==len(seasons) and pbpcov.xyac_mean_nonnull_completed_rate.min()>=0.90)
    situational_cols=[c for c in pbpcov.columns if c.endswith("_nonnull_rate") and c not in ["yac_nonnull_completed_rate","xyac_mean_nonnull_completed_rate","xyac_median_nonnull_completed_rate","xyac_success_nonnull_completed_rate","xyac_fd_nonnull_completed_rate"]]
    situational_good={c:bool(pbpcov[c].min()>=0.90) for c in situational_cols}
    ngs_exists=bool(not ngs_hist.empty and ngs_meta.get("load_error","")=="")
    summary={
        "study":"RB_R27D0_YAC_QUALITY_NOVELTY_AND_SOURCE_AUDIT",
        "source_audit_only":True,
        "model_fit_performed":False,
        "candidate_projection_created":False,
        "prediction_error_scored":False,
        "sportsbook_inputs":0,
        "production_changed":False,
        "r26_changed":False,
        "r22_changed":False,
        "parent_rows":int(len(pred)),
        "vacancy_rb1_targeted_rows":int(len(focus)),
        "pbp_xyac_90pct_all_seasons":pbp_xyac_good,
        "ngs_rb_yacoe_source_exists":ngs_exists,
        "situational_fields_90pct_all_seasons":situational_good,
        "ngs_schema":ngs_meta,
        "strict_prior_coverage":cov,
        "machine_disposition":"R27D0_YAC_QUALITY_SOURCES_SUPPORT_SEPARATELY_FROZEN_PREDICTIVE_STUDY" if pbp_xyac_good and float(covdf.pbp_yacoe_prior_ge3_rate.min())>=0.70 else "R27D0_YAC_QUALITY_SOURCES_PARTIAL_SUPPORT_NEEDS_SCOPE_REDUCTION",
    }
    (out/"r27d0_source_audit_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
