#!/usr/bin/env python3
"""Frozen source-readiness audit for opponent defensive injury burden.

No target outcomes, sportsbook inputs, model candidates, or production writes.
Contract: docs/research/RB_OPPONENT_DEFENDER_INJURY_READINESS_V1_PLAN.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

VERSION = "RB_OPPONENT_DEFENDER_INJURY_READINESS_V1"
INJURY_SEASONS = (2024, 2025, 2026)
SNAP_SEASONS = (2023, 2024, 2025, 2026)
DEF_GROUPS = {"DL", "LB", "DB"}
FRONT7 = {"DL", "LB"}


def _pd(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    if hasattr(obj, "to_dicts"):
        return pd.DataFrame(obj.to_dicts())
    return pd.DataFrame(obj)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def _first(df: pd.DataFrame, names: list[str], default="") -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default,index=df.index)


def _num(s) -> pd.Series:
    return pd.to_numeric(s,errors="coerce")


def _team(v) -> str:
    try:
        return canon_team(v)
    except Exception:
        return str(v or "").strip().upper()


def _name_key(v) -> str:
    s="" if v is None or pd.isna(v) else str(v).lower()
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\b","",s)
    return "".join(ch for ch in s if ch.isalnum())


def _clean_id(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s


def classify_defense_position(v) -> str:
    s="" if v is None or pd.isna(v) else str(v).upper().strip()
    s=s.replace("-","").replace(" ","")
    if not s:
        return ""
    if s in {"DE","LDE","RDE","DT","LDT","RDT","NT","DL","EDGE"} or s.endswith("DE"):
        return "DL"
    if s in {"LB","ILB","MLB","OLB","WLB","SLB"} or s.endswith("LB"):
        return "LB"
    if s in {"DB","CB","LCB","RCB","NCB","NB","S","FS","SS"} or s.endswith("CB"):
        return "DB"
    return ""


def load_sources():
    import nflreadpy as nfl

    injuries={}
    snaps={}
    rosters={}
    source_rows=[]

    for season in INJURY_SEASONS:
        try:
            raw=_lower(_pd(nfl.load_injuries(seasons=[season])))
            if "season" in raw:
                raw=raw.loc[_num(raw["season"]).eq(season)].copy()
            injuries[season]=raw
            source_rows.append({"source":"injuries","season":season,"rows":len(raw),"error":""})
        except Exception as exc:
            injuries[season]=pd.DataFrame()
            source_rows.append({"source":"injuries","season":season,"rows":0,"error":f"{type(exc).__name__}:{exc}"})

    for season in SNAP_SEASONS:
        try:
            raw=_lower(_pd(nfl.load_snap_counts(seasons=[season])))
            if "season" in raw:
                raw=raw.loc[_num(raw["season"]).eq(season)].copy()
            snaps[season]=raw
            source_rows.append({"source":"snap_counts","season":season,"rows":len(raw),"error":""})
        except Exception as exc:
            snaps[season]=pd.DataFrame()
            source_rows.append({"source":"snap_counts","season":season,"rows":0,"error":f"{type(exc).__name__}:{exc}"})
        try:
            raw=_lower(_pd(nfl.load_rosters_weekly(int(season))))
            if "season" in raw:
                raw=raw.loc[_num(raw["season"]).eq(season)].copy()
            rosters[season]=raw
            source_rows.append({"source":"weekly_rosters","season":season,"rows":len(raw),"error":""})
        except Exception as exc:
            rosters[season]=pd.DataFrame()
            source_rows.append({"source":"weekly_rosters","season":season,"rows":0,"error":f"{type(exc).__name__}:{exc}"})

    return injuries,snaps,rosters,pd.DataFrame(source_rows)


def normalize_rosters(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["season","week","team","name_key","position","group","gsis_id","pfr_id"])
    x=_lower(raw)
    out=pd.DataFrame(index=x.index)
    out["season"]=_num(_first(x,["season"],season)).fillna(season)
    out["week"]=_num(_first(x,["week"]))
    out["team"]=_first(x,["team","team_abbr","club_code"]).map(_team)
    out["name_key"]=_first(x,["full_name","football_name","player_name","player","name"]).map(_name_key)
    out["position"]=_first(x,["position","pos","depth_chart_position","ngs_position"]).astype("string").fillna("").str.upper().str.strip()
    out["group"]=out["position"].map(classify_defense_position)
    out["gsis_id"]=_first(x,["gsis_id","player_id"]).map(_clean_id)
    out["pfr_id"]=_first(x,["pfr_id","pfr_player_id"]).map(_clean_id)
    out=out.loc[
        out["season"].eq(season)
        & out["week"].between(1,22)
        & out["team"].ne("")
        & out["name_key"].ne("")
    ].copy()
    out["week"]=out["week"].astype(int)
    return out.drop_duplicates(["season","week","team","name_key"],keep="last")


def normalize_injuries(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["season","week","team","name_key","player","status","practice_status"])
    x=_lower(raw)
    out=pd.DataFrame(index=x.index)
    out["season"]=_num(_first(x,["season"],season)).fillna(season)
    out["week"]=_num(_first(x,["week","report_week"]))
    out["team"]=_first(x,["team","team_abbr","team_abbreviation","club"]).map(_team)
    out["player"]=_first(x,["full_name","player_name","player","name"]).astype("string").fillna("").str.strip()
    out["name_key"]=out["player"].map(_name_key)
    out["status"]=_first(x,["report_status","game_status","status"]).astype("string").fillna("").str.upper().str.strip()
    out["practice_status"]=_first(x,["practice_status","practice_participation"]).astype("string").fillna("").str.upper().str.strip()
    out=out.loc[
        out["season"].eq(season)
        & out["week"].between(1,22)
        & out["team"].ne("")
        & out["name_key"].ne("")
    ].copy()
    out["week"]=out["week"].astype(int)
    return out.drop_duplicates(["season","week","team","name_key"],keep="last")


def normalize_snaps(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["season","week","team","name_key","pfr_id","defense_snaps","defense_pct"])
    x=_lower(raw)
    out=pd.DataFrame(index=x.index)
    out["season"]=_num(_first(x,["season"],season)).fillna(season)
    out["week"]=_num(_first(x,["week"]))
    out["team"]=_first(x,["team","team_abbr"]).map(_team)
    out["name_key"]=_first(x,["player","player_name","full_name"]).map(_name_key)
    out["pfr_id"]=_first(x,["pfr_player_id","pfr_id"]).map(_clean_id)
    out["defense_snaps"]=_num(_first(x,["defense_snaps","def_snaps"],np.nan))
    out["defense_pct"]=_num(_first(x,["defense_pct","def_pct","defense_percentage"],np.nan))
    out=out.loc[
        out["season"].eq(season)
        & out["week"].between(1,22)
        & (out["name_key"].ne("") | out["pfr_id"].ne(""))
    ].copy()
    out["week"]=out["week"].astype(int)
    return out.drop_duplicates(["season","week","team","pfr_id","name_key"],keep="last")


def roster_bridge(inj: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if inj.empty:
        return inj.assign(position="",group="",gsis_id="",pfr_id="",identity_method="")
    keys=["season","week","team","name_key"]
    counts=roster.groupby(keys,dropna=False).size().rename("_roster_matches").reset_index() if len(roster) else pd.DataFrame(columns=keys+["_roster_matches"])
    unique=roster.merge(counts,on=keys,how="left") if len(roster) else roster
    unique=unique.loc[unique["_roster_matches"].eq(1)].copy() if len(unique) else unique
    cols=keys+["position","group","gsis_id","pfr_id"]
    out=inj.merge(unique[cols],on=keys,how="left",validate="many_to_one")
    out["identity_method"]=np.where(
        out["group"].astype(str).ne("") & (out["gsis_id"].astype(str).ne("") | out["pfr_id"].astype(str).ne("")),
        "ROSTER_MEDIATED_STABLE_ID",
        np.where(out["group"].astype(str).ne(""),"ROSTER_MEDIATED_NAME_ONLY","UNRESOLVED"),
    )
    return out


def latest_prior_snap(row, snaps: pd.DataFrame) -> dict:
    season=int(row["season"]); week=int(row["week"])
    q=snaps.loc[
        (snaps["season"].lt(season) | (snaps["season"].eq(season) & snaps["week"].lt(week)))
    ].copy()
    method=""
    pid=_clean_id(row.get("pfr_id",""))
    if pid:
        z=q.loc[q["pfr_id"].eq(pid)].copy()
        method="PFR_ID"
    else:
        z=pd.DataFrame()
    if z.empty:
        nk=str(row.get("name_key",""))
        team=str(row.get("team",""))
        z=q.loc[q["name_key"].eq(nk) & q["team"].eq(team)].copy()
        method="NAME_TEAM_FALLBACK" if len(z) else ""
    if z.empty:
        return {"snap_joined":False,"snap_join_method":"","snap_source_season":np.nan,"snap_source_week":np.nan,"prior_defense_snaps":np.nan,"prior_defense_pct":np.nan,"chronology_valid":True}
    z=z.sort_values(["season","week"])
    r=z.iloc[-1]
    valid=bool(int(r.season)<season or (int(r.season)==season and int(r.week)<week))
    return {
        "snap_joined":True,
        "snap_join_method":method,
        "snap_source_season":int(r.season),
        "snap_source_week":int(r.week),
        "prior_defense_snaps":float(r.defense_snaps) if pd.notna(r.defense_snaps) else np.nan,
        "prior_defense_pct":float(r.defense_pct) if pd.notna(r.defense_pct) else np.nan,
        "chronology_valid":valid,
    }


def audit(injuries, snaps, rosters, source_rows: pd.DataFrame):
    inj_parts=[]
    roster_parts=[]
    snap_parts=[]
    for season in INJURY_SEASONS:
        inj_parts.append(normalize_injuries(injuries[season],season))
    for season in SNAP_SEASONS:
        roster_parts.append(normalize_rosters(rosters[season],season))
        snap_parts.append(normalize_snaps(snaps[season],season))
    inj=pd.concat(inj_parts,ignore_index=True) if inj_parts else pd.DataFrame()
    roster=pd.concat(roster_parts,ignore_index=True) if roster_parts else pd.DataFrame()
    snap=pd.concat(snap_parts,ignore_index=True) if snap_parts else pd.DataFrame()

    if inj.empty:
        evidence=pd.DataFrame()
    else:
        evidence=roster_bridge(inj,roster)
        evidence=evidence.loc[evidence["group"].isin(DEF_GROUPS)].copy()
        prior=[latest_prior_snap(r,snap) for _,r in evidence.iterrows()]
        evidence=pd.concat([evidence.reset_index(drop=True),pd.DataFrame(prior)],axis=1) if len(evidence) else evidence
        st=evidence["status"].fillna("").astype(str).str.upper()
        evidence["out_doubtful"]=st.str.contains(r"\bOUT\b|DOUBT",regex=True)
        evidence["front7"]=evidence["group"].isin(FRONT7)

    src={(str(r.source),int(r.season)):int(r.rows) for r in source_rows.itertuples() if pd.notna(r.season)}
    injury_present={s:src.get(("injuries",s),0)>0 for s in INJURY_SEASONS}
    snap_present={s:src.get(("snap_counts",s),0)>0 for s in SNAP_SEASONS}

    def cov(mask=None):
        if evidence.empty:
            return 0.0,0
        q=evidence if mask is None else evidence.loc[mask]
        if q.empty:
            return 0.0,0
        return float(q["snap_joined"].mean()),int(len(q))

    all_cov,all_n=cov()
    od_mask=evidence["out_doubtful"] if len(evidence) else pd.Series(dtype=bool)
    od_cov,od_n=cov(od_mask) if len(evidence) else (0.0,0)
    f7_mask=(evidence["out_doubtful"] & evidence["front7"]) if len(evidence) else pd.Series(dtype=bool)
    f7_cov,f7_n=cov(f7_mask) if len(evidence) else (0.0,0)

    live=evidence.loc[evidence["season"].eq(2026)] if len(evidence) else evidence
    live_teams=int(live["team"].nunique()) if len(live) else 0
    collisions=0
    if len(roster):
        rc=roster.groupby(["season","week","team","name_key"]).size()
        collisions=int(rc.gt(1).sum())

    same_future=int((~evidence["chronology_valid"]).sum()) if len(evidence) and "chronology_valid" in evidence else 0
    matched=evidence.loc[evidence["snap_joined"]] if len(evidence) else evidence
    stable=float(
        matched["identity_method"].eq("ROSTER_MEDIATED_STABLE_ID").mean()
    ) if len(matched) else 0.0

    integrity=bool(same_future==0 and collisions==0)
    ready=bool(
        all(injury_present.values())
        and all(snap_present.values())
        and live_teams>=30
        and all_cov>=.90
        and od_cov>=.90
        and f7_cov>=.90
        and integrity
        and stable>=.80
    )
    partial=bool(
        all(injury_present.values())
        and all(snap_present.values())
        and integrity
        and all_cov>=.70
    )
    disposition=(
        "RB_OPPONENT_DEFENDER_INJURY_SOURCE_READY" if ready else
        "RB_OPPONENT_DEFENDER_INJURY_SOURCE_PARTIAL" if partial else
        "RB_OPPONENT_DEFENDER_INJURY_SOURCE_NOT_READY"
    )

    summary={
        "version":VERSION,
        "disposition":disposition,
        "injury_source_rows":{str(s):src.get(("injuries",s),0) for s in INJURY_SEASONS},
        "snap_source_rows":{str(s):src.get(("snap_counts",s),0) for s in SNAP_SEASONS},
        "injury_seasons_present":{str(s):bool(injury_present[s]) for s in INJURY_SEASONS},
        "snap_seasons_present":{str(s):bool(snap_present[s]) for s in SNAP_SEASONS},
        "live_2026_defensive_injury_teams":live_teams,
        "defensive_injury_rows":int(len(evidence)),
        "all_defense_join_rows":all_n,
        "all_defense_prior_snap_coverage":all_cov,
        "out_doubtful_rows":od_n,
        "out_doubtful_prior_snap_coverage":od_cov,
        "front7_out_doubtful_rows":f7_n,
        "front7_out_doubtful_prior_snap_coverage":f7_cov,
        "stable_or_roster_mediated_share_of_matched":stable,
        "same_future_snap_violations":same_future,
        "unresolved_identity_collisions":collisions,
        "sportsbook_inputs_used":0,
        "target_game_outcomes_read":0,
        "candidate_variants_constructed":0,
        "candidate_variants_scored":0,
        "parameters_fit":0,
        "production_mutations":0,
    }
    return evidence,summary


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    injuries,snaps,rosters,source_rows=load_sources()
    evidence,result=audit(injuries,snaps,rosters,source_rows)
    source_rows.to_csv(a.out_dir/"rb_opponent_defender_injury_source_rows.csv",index=False)
    # Row-level evidence contains only free public-source fields, but remains a research artifact.
    evidence.to_csv(a.out_dir/"rb_opponent_defender_injury_readiness_rows.csv",index=False)
    (a.out_dir/"rb_opponent_defender_injury_readiness_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
