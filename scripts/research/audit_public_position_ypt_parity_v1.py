#!/usr/bin/env python3
"""Public position YPT allowed historical/live parity audit.

No predictive candidate, sportsbook input, target Week-5 outcome, or production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas

VERSION="PUBLIC_POSITION_YPT_ALLOWED_PARITY_V1"
HIST_SEASONS=(2024,2025)
LIVE_SEASON=2026
LIVE_TARGET_WEEK=5
TARGET_WEEKS=tuple(range(2,19))
RB_POS={"RB","HB","FB"}
TOL=1e-10


def _pd(x):
    return _to_pandas(x)


def _num(x):
    return pd.to_numeric(x,errors="coerce")


def load_schedule(season:int)->pd.DataFrame:
    import nflreadpy as nfl
    try:
        raw=nfl.load_schedules(seasons=[int(season)])
    except TypeError:
        raw=nfl.load_schedules(int(season))
    x=_pd(raw)
    x.columns=[str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"schedule returned zero rows season={season}")
    if "game_type" in x.columns:
        x=x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
    elif "season_type" in x.columns:
        x=x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
    x["season"]=_num(x.get("season",season)).fillna(season).astype(int)
    x["week"]=_num(x["week"])
    hcol="home_team" if "home_team" in x.columns else "home"
    acol="away_team" if "away_team" in x.columns else "away"
    gid="game_id" if "game_id" in x.columns else None
    rows=[]
    for r in x.itertuples(index=False):
        w=int(getattr(r,"week"))
        h=canon_team(getattr(r,hcol)); a=canon_team(getattr(r,acol))
        g=str(getattr(r,gid)) if gid else f"{season}_{w:02d}_{a}_{h}"
        rows.append({"season":season,"week":w,"team":h,"opponent":a,"game_id":g})
        rows.append({"season":season,"week":w,"team":a,"opponent":h,"game_id":g})
    out=pd.DataFrame(rows)
    if out.duplicated(["season","week","team"]).any():
        raise RuntimeError(f"duplicate schedule team-week season={season}")
    return out


def load_logs(season:int,schedule:pd.DataFrame)->pd.DataFrame:
    import nflreadpy as nfl
    raw=nfl.load_player_stats(seasons=[int(season)],summary_level="week")
    x=_normalize_weekly(_pd(raw),int(season))
    x=x.merge(
        schedule[["season","week","team","opponent","game_id"]],
        on=["season","week","team"],
        how="left",
        validate="many_to_one",
    )
    missing=x["opponent"].isna()|x["opponent"].astype(str).eq("")
    if missing.any():
        bad=x.loc[missing,["season","week","team"]].drop_duplicates().head(20)
        raise RuntimeError(f"player logs missing opponent: {bad.to_dict('records')}")
    return x


def build_position_ypt(logs:pd.DataFrame,target_weeks:list[int]|tuple[int,...])->pd.DataFrame:
    x=logs.copy()
    x["position"]=x["position"].astype(str).str.upper().str.strip()
    x["pos_group"]=np.select(
        [x["position"].eq("WR"),x["position"].eq("TE"),x["position"].isin(RB_POS)],
        ["WR","TE","RB"],default="OTHER"
    )
    x=x.loc[x["pos_group"].isin(["WR","TE","RB"])].copy()
    x["targets"]=_num(x["targets"]).fillna(0.0)
    x["rec_yards"]=_num(x["rec_yards"]).fillna(0.0)
    x["defense"]=x["opponent"].map(canon_team)
    weekly=x.groupby(["season","week","defense","pos_group"],as_index=False).agg(
        targets=("targets","sum"),rec_yards=("rec_yards","sum")
    )
    rows=[]
    for season in sorted(weekly["season"].dropna().astype(int).unique()):
        sx=weekly.loc[weekly["season"].eq(season)].copy()
        defenses=sorted(set(sx["defense"].dropna().astype(str)))
        for target_week in target_weeks:
            h=sx.loc[sx["week"].lt(target_week)].copy()
            for defense in defenses:
                d=h.loc[h["defense"].eq(defense)].copy()
                if d.empty:
                    continue
                keep_weeks=sorted(d["week"].dropna().astype(int).unique())[-8:]
                d=d.loc[d["week"].isin(keep_weeks)]
                rec={"season":int(season),"week":int(target_week),"defense":defense,
                     "source_max_week":int(d["week"].max()) if len(d) else np.nan}
                for pos in ("WR","TE","RB"):
                    p=d.loc[d["pos_group"].eq(pos)]
                    den=float(p["targets"].sum())
                    rec[f"{pos.lower()}_targets_faced_public"]=den
                    rec[f"{pos.lower()}_ypt_allowed_public"]=float(p["rec_yards"].sum()/den) if den>0 else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def historical_compare(authority:pd.DataFrame,rebuild:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    a=authority.copy()
    a.columns=[str(c).strip().lower() for c in a.columns]
    for c in ("season","week"):
        a[c]=_num(a[c]).astype(int)
    a["team"]=a["team"].map(canon_team)
    a["opponent"]=a["opponent"].map(canon_team)
    a=a.loc[a["season"].isin(HIST_SEASONS)&a["week"].isin(TARGET_WEEKS)].copy()

    b=rebuild.copy().rename(columns={"defense":"opponent"})
    joined=a.merge(
        b,
        on=["season","week","opponent"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    rows=[]
    gates={}
    for pos in ("wr","te","rb"):
        ac=f"def_{pos}_ypt_allowed"
        bc=f"{pos}_ypt_allowed_public"
        av=_num(joined[ac]); bv=_num(joined[bc])
        either=av.notna()|bv.notna()
        both=av.notna()&bv.notna()
        missing_agree=float((av.notna()==bv.notna())[either].mean()) if either.any() else 1.0
        max_gap=float((av[both]-bv[both]).abs().max()) if both.any() else 0.0
        rows.append({
            "position":pos.upper(),
            "authority_rows":int(len(joined)),
            "either_finite_rows":int(either.sum()),
            "both_finite_rows":int(both.sum()),
            "missingness_agreement":missing_agree,
            "max_abs_gap":max_gap,
        })
        gates[pos]=bool(missing_agree>=.995 and max_gap<=TOL)
    identity_coverage=float(joined["_merge"].eq("both").mean()) if len(joined) else 0.0
    return pd.DataFrame(rows),{
        "authority_rows":int(len(joined)),
        "identity_coverage":identity_coverage,
        "all_positions_exact":bool(all(gates.values())),
        "position_gates":gates,
    }


def live_readiness(logs:pd.DataFrame,schedule:pd.DataFrame,feature:pd.DataFrame)->dict:
    week5=schedule.loc[schedule["week"].eq(LIVE_TARGET_WEEK)].copy()
    teams=sorted(week5["team"].unique())
    f=feature.loc[feature["week"].eq(LIVE_TARGET_WEEK)].copy()
    f=f.loc[f["defense"].isin(teams)].copy()
    covered=set(f.loc[_num(f["te_targets_faced_public"]).gt(0)&_num(f["te_ypt_allowed_public"]).notna(),"defense"])
    max_source=int(_num(f["source_max_week"]).max()) if len(f) else -1
    logs_rows=int(len(logs.loc[_num(logs["week"]).between(1,LIVE_TARGET_WEEK-1)]))
    return {
        "week5_scheduled_teams":int(len(teams)),
        "week5_te_ypt_covered_teams":int(len(covered)),
        "week5_te_ypt_missing_teams":sorted(set(teams)-covered),
        "week5_all_te_covered":bool(set(teams)==covered and len(teams)>0),
        "live_2026_completed_prior_stat_rows":logs_rows,
        "max_source_week_used":max_source,
        "chronology_violations":int(max(0,max_source-(LIVE_TARGET_WEEK-1))>0),
        "target_week5_outcomes_read":0,
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase-bc-features",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    schedules={s:load_schedule(s) for s in (*HIST_SEASONS,LIVE_SEASON)}
    logs={s:load_logs(s,schedules[s]) for s in (*HIST_SEASONS,LIVE_SEASON)}

    hist_logs=pd.concat([logs[2024],logs[2025]],ignore_index=True)
    hist=build_position_ypt(hist_logs,TARGET_WEEKS)
    authority=pd.read_csv(a.phase_bc_features,low_memory=False)
    comparison,parity=historical_compare(authority,hist)

    live=build_position_ypt(logs[LIVE_SEASON],[LIVE_TARGET_WEEK])
    readiness=live_readiness(logs[LIVE_SEASON],schedules[LIVE_SEASON],live)

    hist_ok=bool(parity["identity_coverage"]>=.995 and parity["all_positions_exact"])
    live_ok=bool(
        readiness["live_2026_completed_prior_stat_rows"]>0
        and readiness["week5_all_te_covered"]
        and readiness["chronology_violations"]==0
        and readiness["max_source_week_used"]<LIVE_TARGET_WEEK
    )
    disposition=(
        "PUBLIC_POSITION_YPT_EXACT_PARITY_READY" if hist_ok and live_ok else
        "PUBLIC_POSITION_YPT_HISTORICAL_PARITY_ONLY" if hist_ok else
        "PUBLIC_POSITION_YPT_PARITY_FAILED"
    )
    result={
        "version":VERSION,
        "disposition":disposition,
        "historical_parity":parity,
        "live_2026_week5":readiness,
        "source":"nflreadpy.load_player_stats(summary_level=week) + nflreadpy.load_schedules",
        "formula":"latest_8_prior_same_season_weeks sum(rec_yards)/sum(targets) by opponent and position group",
        "predictive_candidates_scored":0,
        "parameters_fit":0,
        "sportsbook_inputs_used":0,
        "target_week5_outcomes_read":0,
        "production_changed":False,
    }
    comparison.to_csv(a.out_dir/"public_position_ypt_historical_parity.csv",index=False)
    hist.to_csv(a.out_dir/"public_position_ypt_historical_features.csv",index=False)
    live.to_csv(a.out_dir/"public_position_ypt_2026_week5_features.csv",index=False)
    (a.out_dir/"public_position_ypt_parity_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    print(comparison.to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
