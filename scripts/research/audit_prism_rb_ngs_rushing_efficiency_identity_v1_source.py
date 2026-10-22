#!/usr/bin/env python3
"""PRISM RB NGS rushing efficiency identity V1 source audit.

Source/chronology only. Historical NGS rushing seasons 2023-2025. No 2026
rows, predictive scoring, sportsbook/OddsAPI input, model fitting, or
production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION="PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_V1_SOURCE_AUDIT"
SEASONS=(2023,2024,2025)
TARGET_SEASONS=(2024,2025)
RB_POS={"RB","HB","FB"}


def num(x:Any)->pd.Series:
    return pd.to_numeric(x,errors="coerce")


def to_pd(x:Any)->pd.DataFrame:
    if isinstance(x,pd.DataFrame):
        return x.copy()
    if hasattr(x,"to_pandas"):
        return x.to_pandas()
    if hasattr(x,"to_dicts"):
        return pd.DataFrame(x.to_dicts())
    return pd.DataFrame(x)


def lower(x:pd.DataFrame)->pd.DataFrame:
    y=x.copy()
    y.columns=[str(c).strip().lower() for c in y.columns]
    return y


def first_col(df:pd.DataFrame,names:list[str])->str|None:
    for c in names:
        if c in df.columns:
            return c
    return None


def regular_rows(x:pd.DataFrame)->pd.DataFrame:
    q=x.copy()
    st=first_col(q,["season_type","game_type"])
    if st:
        q=q[q[st].fillna("").astype(str).str.upper().isin(["REG","REGULAR","RS",""])].copy()
    if "week" not in q.columns:
        raise RuntimeError("NGS rushing source missing week")
    q["week"]=num(q["week"])
    q=q[q["week"].between(1,18,inclusive="both")].copy()
    return q


def load_ngs()->tuple[pd.DataFrame,dict]:
    import nflreadpy as nfl

    raw=lower(to_pd(nfl.load_nextgen_stats(seasons=list(SEASONS),stat_type="rushing")))
    if raw.empty:
        raise RuntimeError("NGS rushing source empty")
    q=regular_rows(raw)

    aliases={
        "season":["season"],
        "week":["week"],
        "player_id":["player_gsis_id","player_id","gsis_id"],
        "player":["player_display_name","player_name","player"],
        "team":["team_abbr","team","club_code"],
        "position":["player_position","position","position_group"],
        "attempts":["rush_attempts","attempts","carries"],
        "rush_yards":["rush_yards","rushing_yards","yards"],
        "ryoe_pa":["rush_yards_over_expected_per_att","rush_yards_over_expected_per_attempt","ryoe_per_att","ryoe_per_attempt"],
        "ryoe_total":["rush_yards_over_expected","ryoe"],
        "rush_pct_oe":["rush_pct_over_expected","rush_percent_over_expected","percent_rush_yards_over_expected"],
        "box8_pct":["percent_attempts_gte_eight_defenders","pct_attempts_gte_eight_defenders","percent_attempts_gte_8_defenders","pct_attempts_gte_8_defenders"],
        "time_los":["avg_time_to_los","average_time_to_los","avg_time_to_line_of_scrimmage"],
        "efficiency":["efficiency","rushing_efficiency","rush_efficiency"],
    }
    resolved={k:first_col(q,v) for k,v in aliases.items()}
    hard=["season","week","player_id","team","position","ryoe_pa","box8_pct","time_los","efficiency"]
    missing=[k for k in hard if not resolved[k]]
    if missing:
        raise RuntimeError("NGS_REQUIRED_COLUMNS_MISSING:"+json.dumps({"missing":missing,"resolved":resolved,"columns":list(q.columns)},sort_keys=True))
    if not resolved["ryoe_total"] and not resolved["rush_pct_oe"]:
        raise RuntimeError("NGS_REQUIRED_RYOE_SECONDARY_FAMILY_MISSING:"+json.dumps({"resolved":resolved,"columns":list(q.columns)},sort_keys=True))

    out=pd.DataFrame({
        "season":num(q[resolved["season"]]).astype("Int64"),
        "week":num(q[resolved["week"]]).astype("Int64"),
        "player_id":q[resolved["player_id"]].astype("string").fillna("").str.strip(),
        "player":q[resolved["player"]].astype("string").fillna("").str.strip() if resolved["player"] else "",
        "team":q[resolved["team"]].astype("string").fillna("").str.strip().str.upper(),
        "position":q[resolved["position"]].astype("string").fillna("").str.strip().str.upper(),
        "attempts":num(q[resolved["attempts"]]) if resolved["attempts"] else np.nan,
        "rush_yards":num(q[resolved["rush_yards"]]) if resolved["rush_yards"] else np.nan,
        "ryoe_per_att":num(q[resolved["ryoe_pa"]]),
        "ryoe_total":num(q[resolved["ryoe_total"]]) if resolved["ryoe_total"] else np.nan,
        "rush_pct_over_expected":num(q[resolved["rush_pct_oe"]]) if resolved["rush_pct_oe"] else np.nan,
        "box8_pct":num(q[resolved["box8_pct"]]),
        "avg_time_to_los":num(q[resolved["time_los"]]),
        "ngs_efficiency":num(q[resolved["efficiency"]]),
    })
    if out["season"].ge(2026).any():
        raise RuntimeError("2026 NGS rows loaded")
    out=out[out["season"].isin(SEASONS)].copy()
    if out.duplicated(["season","week","player_id"]).any():
        dup=out[out.duplicated(["season","week","player_id"],keep=False)].head(20)
        raise RuntimeError("duplicate season/week/player NGS identity:"+json.dumps(dup.to_dict("records"),default=str))
    meta={"resolved_columns":resolved,"raw_rows":int(len(raw)),"regular_rows":int(len(out))}
    return out,meta


def season_metrics(ngs:pd.DataFrame,season:int)->dict:
    s=ngs[ngs["season"].eq(season)].copy()
    rb=s[s["position"].isin(RB_POS)].copy()
    nrows=len(rb)
    finite_secondary=rb["ryoe_total"].notna()|rb["rush_pct_over_expected"].notna()
    usable=(
        rb["player_id"].ne("")
        & rb["team"].ne("")
        & rb["position"].isin(RB_POS)
        & rb["ryoe_per_att"].notna()
        & finite_secondary
        & rb["box8_pct"].notna()
        & rb["avg_time_to_los"].notna()
        & rb["ngs_efficiency"].notna()
    )
    def rate(z:pd.Series)->float:
        return float(z.mean()) if len(z) else 0.0
    return {
        "season":season,
        "regular_weeks_present":int(s["week"].nunique()),
        "ngs_rb_rows":int(nrows),
        "stable_player_id_rate":rate(rb["player_id"].ne("")),
        "team_identity_rate":rate(rb["team"].ne("")),
        "rb_position_identity_rate":rate(rb["position"].isin(RB_POS)),
        "ryoe_per_att_finite_rate":rate(rb["ryoe_per_att"].notna()),
        "ryoe_secondary_finite_rate":rate(finite_secondary),
        "box8_pct_finite_rate":rate(rb["box8_pct"].notna()),
        "avg_time_to_los_finite_rate":rate(rb["avg_time_to_los"].notna()),
        "ngs_efficiency_finite_rate":rate(rb["ngs_efficiency"].notna()),
        "usable_rb_week_rows":int(usable.sum()),
        "usable_distinct_rbs":int(rb.loc[usable,"player_id"].nunique()),
        "usable_teams":int(rb.loc[usable,"team"].nunique()),
        "duplicate_identity_rows":int(rb.duplicated(["season","week","player_id"]).sum()),
    }


def load_pregame_universe(path:Path)->pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    need={"season","week","team","player_id","position","family","stable_roster_identity","known_official_inactive"}
    miss=need-set(x.columns)
    if miss:
        raise RuntimeError(f"pregame universe missing {sorted(miss)}")
    x["season"]=num(x["season"]).astype("Int64")
    x["week"]=num(x["week"]).astype("Int64")
    x["team"]=x["team"].astype("string").fillna("").str.strip().str.upper()
    x["player_id"]=x["player_id"].astype("string").fillna("").str.strip()
    x["position"]=x["position"].astype("string").fillna("").str.strip().str.upper()
    x["family"]=x["family"].astype("string").fillna("").str.strip().str.upper()
    x["stable_roster_identity"]=num(x["stable_roster_identity"]).fillna(0).astype(int)
    x["known_official_inactive"]=num(x["known_official_inactive"]).fillna(0).astype(int)
    x=x[x["season"].isin(TARGET_SEASONS)].copy()
    if int(x["known_official_inactive"].sum())!=0:
        raise RuntimeError("pregame active universe unexpectedly contains official-inactive rows")
    if x.duplicated(["season","week","team","player_id"]).any():
        raise RuntimeError("duplicate pregame target identity")
    return x


def target_history(ngs:pd.DataFrame,pregame:pd.DataFrame,season:int)->tuple[pd.DataFrame,dict]:
    hist=ngs[(ngs["season"].eq(season))&ngs["position"].isin(RB_POS)].copy()
    hist["secondary_finite"]=hist["ryoe_total"].notna()|hist["rush_pct_over_expected"].notna()
    hist["usable"]=(
        hist["player_id"].ne("")
        & hist["team"].ne("")
        & hist["ryoe_per_att"].notna()
        & hist["secondary_finite"]
        & hist["box8_pct"].notna()
        & hist["avg_time_to_los"].notna()
        & hist["ngs_efficiency"].notna()
    )
    hist=hist[hist["usable"]].sort_values(["player_id","week"]).copy()

    target=pregame[
        pregame["season"].eq(season)
        & pregame["family"].eq("RB")
        & pregame["stable_roster_identity"].eq(1)
        & pregame["player_id"].ne("")
    ].copy()

    rows=[]
    for r in target.itertuples(index=False):
        h=hist[(hist["player_id"].eq(str(r.player_id)))&num(hist["week"]).lt(int(r.week))]
        max_week=int(num(h["week"]).max()) if len(h) else pd.NA
        rows.append({
            "season":season,"week":int(r.week),"team":str(r.team),"player_id":str(r.player_id),
            "position":str(r.position),
            "prior_same_season_ngs_games":int(h["week"].nunique()),
            "max_prior_history_week":max_week,
            "chronology_violation":bool(len(h) and num(h["week"]).ge(int(r.week)).any()),
        })
    t=pd.DataFrame(rows)
    ready=t[t["prior_same_season_ngs_games"].ge(3)].copy() if len(t) else t
    m={
        "season":season,
        "target_pregame_rb_rows":int(len(t)),
        "history_ready_rows":int(len(ready)),
        "history_ready_distinct_rbs":int(ready["player_id"].nunique()) if len(ready) else 0,
        "history_ready_teams":int(ready["team"].nunique()) if len(ready) else 0,
        "chronology_violations":int(t["chronology_violation"].sum()) if len(t) else 0,
    }
    return t,m

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--pregame-universe",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)

    ngs,meta=load_ngs()
    pregame=load_pregame_universe(a.pregame_universe)
    sm=[season_metrics(ngs,s) for s in SEASONS]
    target_frames=[];tm=[]
    for s in TARGET_SEASONS:
        t,m=target_history(ngs,pregame,s);target_frames.append(t);tm.append(m)
    targets=pd.concat(target_frames,ignore_index=True)

    gates={}
    for m in sm:
        s=m["season"]
        gates[str(s)]={
            "weeks_ge17":m["regular_weeks_present"]>=17,
            "stable_id_ge99":m["stable_player_id_rate"]>=.99,
            "team_identity_ge99":m["team_identity_rate"]>=.99,
            "rb_position_ge95":m["rb_position_identity_rate"]>=.95,
            "ryoe_per_att_ge95":m["ryoe_per_att_finite_rate"]>=.95,
            "box8_ge95":m["box8_pct_finite_rate"]>=.95,
            "time_los_ge95":m["avg_time_to_los_finite_rate"]>=.95,
            "efficiency_ge95":m["ngs_efficiency_finite_rate"]>=.95,
            "usable_rows_ge500":m["usable_rb_week_rows"]>=500,
            "distinct_rbs_ge70":m["usable_distinct_rbs"]>=70,
            "teams_ge25":m["usable_teams"]>=25,
            "duplicates_zero":m["duplicate_identity_rows"]==0,
        }
    target_gates={}
    for m in tm:
        s=m["season"]
        target_gates[str(s)]={
            "history_ready_rows_ge900":m["history_ready_rows"]>=900,
            "history_ready_rbs_ge80":m["history_ready_distinct_rbs"]>=80,
            "history_ready_teams_ge28":m["history_ready_teams"]>=28,
            "chronology_zero":m["chronology_violations"]==0,
        }

    all_pass=all(all(v.values()) for v in gates.values()) and all(all(v.values()) for v in target_gates.values())
    disposition=(
        "PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_SOURCE_READY"
        if all_pass else
        "PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_SOURCE_BLOCKED"
    )
    integrity={
        "target_game_ngs_rows_used_in_predictor_state":0,
        "target_game_rush_yards_ypc_used_in_predictor_state":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "outcomes_or_ngs_2026_loaded":0,
        "predictive_models_fit":0,
        "thresholds_searched":0,
        "production_changed":False,
        "chronology_violations":int(sum(m["chronology_violations"] for m in tm)),
    }
    result={
        "version":VERSION,
        "disposition":disposition,
        "source_seasons":list(SEASONS),
        "target_history_seasons":list(TARGET_SEASONS),
        "source_metadata":meta,
        "season_metrics":sm,
        "target_history_metrics":tm,
        "source_gates":gates,
        "target_history_gates":target_gates,
        "integrity":integrity,
    }
    ngs.to_csv(a.out_dir/"rb_ngs_rushing_source_rows.csv",index=False,float_format="%.12f")
    targets.to_csv(a.out_dir/"rb_ngs_rushing_target_history_readiness.csv",index=False)
    pd.DataFrame(sm).to_csv(a.out_dir/"rb_ngs_rushing_source_metrics.csv",index=False,float_format="%.12f")
    pd.DataFrame(tm).to_csv(a.out_dir/"rb_ngs_rushing_target_history_metrics.csv",index=False)
    (a.out_dir/"rb_ngs_rushing_efficiency_identity_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True,default=str))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

# child-pr-rerun2-trigger-v3
