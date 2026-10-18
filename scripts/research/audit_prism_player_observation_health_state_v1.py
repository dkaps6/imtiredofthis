#!/usr/bin/env python3
"""PRISM Player Observation Health-State V1 source audit.

Source/lineage only. Builds completed historical player-game observation-state
labels from schedule, weekly rosters, snap participation and injury reports.
No predictive outcome is scored and no projection is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.player_identity_v3 import player_name_key

VERSION = "PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "731541d42c82b94cf4261d31d12202ad0911e8e8"
SEASONS = (2024, 2025)
WEEKS = tuple(range(1, 19))
SKILL = {"QB", "RB", "HB", "TB", "FB", "WR", "TE"}

STATE_NO_REPORT_PARTICIPATED = "NO_REPORT_PARTICIPATED"
STATE_REPORTED_PARTICIPATED = "REPORTED_INJURY_PARTICIPATED"
STATE_REPORTED_NO_SNAPS = "REPORTED_INJURY_NO_OFFENSIVE_SNAPS"
STATE_UNRESOLVED_NO_SNAPS = "NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED"
STATES = (
    STATE_NO_REPORT_PARTICIPATED,
    STATE_REPORTED_PARTICIPATED,
    STATE_REPORTED_NO_SNAPS,
    STATE_UNRESOLVED_NO_SNAPS,
)

def json_default(value):
    """Serialize numpy scalar types without changing audit semantics."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")



def pdx(v) -> pd.DataFrame:
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    if hasattr(v, "to_dicts"):
        return pd.DataFrame(v.to_dicts())
    return pd.DataFrame(v)


def lower(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    y.columns = [str(c).strip().lower() for c in y.columns]
    return y


def first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def stable(v) -> str:
    return text(v)


def name_key(v) -> str:
    return player_name_key(v, strip_suffix=True)


def family(v) -> str:
    p = text(v).upper()
    if p == "QB" or p.startswith("QB"):
        return "QB"
    if p in {"RB", "HB", "TB", "FB"} or p.startswith(("RB", "HB", "TB", "FB")):
        return "RB"
    if p == "WR" or p.startswith("WR"):
        return "WR"
    if p == "TE" or p.startswith("TE"):
        return "TE"
    return ""


def normalize_schedule(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = lower(raw)
    need = {
        "season": first_col(x, ("season",)),
        "week": first_col(x, ("week",)),
        "game_type": first_col(x, ("game_type", "season_type")),
        "home": first_col(x, ("home_team", "home")),
        "away": first_col(x, ("away_team", "away")),
    }
    miss = [k for k,v in need.items() if v is None]
    if miss:
        raise RuntimeError(f"schedule missing semantics: {miss}")
    q = pd.DataFrame({
        "season": pd.to_numeric(x[need["season"]], errors="coerce"),
        "week": pd.to_numeric(x[need["week"]], errors="coerce"),
        "game_type": x[need["game_type"]].astype("string").str.upper().str.strip(),
        "home": x[need["home"]].map(canon_team),
        "away": x[need["away"]].map(canon_team),
    })
    q = q.loc[
        q["season"].isin(SEASONS)
        & q["week"].isin(WEEKS)
        & q["game_type"].eq("REG")
        & q["home"].ne("")
        & q["away"].ne("")
    ].copy()
    rows = []
    for r in q.itertuples(index=False):
        rows.append({"season": int(r.season), "week": int(r.week), "team": str(r.home)})
        rows.append({"season": int(r.season), "week": int(r.week), "team": str(r.away)})
    tw = pd.DataFrame(rows).drop_duplicates().sort_values(["season","week","team"]).reset_index(drop=True)
    if tw.duplicated(["season","week","team"]).any():
        raise RuntimeError("schedule team-week duplicates")
    return tw, {"columns": need, "team_weeks": int(len(tw))}


def normalize_rosters(parts: list[tuple[int, pd.DataFrame]], scheduled: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = []
    schemas = {}
    scheduled_keys = set(zip(scheduled["season"], scheduled["week"], scheduled["team"]))
    for requested, raw in parts:
        x = lower(raw)
        cols = {
            "season": first_col(x, ("season",)),
            "week": first_col(x, ("week",)),
            "team": first_col(x, ("team", "team_abbr", "team_abbreviation", "club")),
            "stable_id": first_col(x, ("gsis_id", "player_id", "nflverse_id")),
            "name": first_col(x, ("player_name", "full_name", "player", "name")),
            "position": first_col(x, ("position", "pos", "position_group")),
            "status": first_col(x, ("status", "roster_status")),
        }
        required = ["week","team","name","position"]
        miss = [k for k in required if cols[k] is None]
        if miss:
            raise RuntimeError(f"weekly roster {requested} missing semantics {miss}")
        season = (
            pd.to_numeric(x[cols["season"]], errors="coerce")
            if cols["season"] else
            pd.Series(int(requested), index=x.index, dtype=float)
        )
        q = pd.DataFrame({
            "season": season,
            "week": pd.to_numeric(x[cols["week"]], errors="coerce"),
            "team": x[cols["team"]].map(canon_team),
            "stable_id": x[cols["stable_id"]].map(stable) if cols["stable_id"] else "",
            "player": x[cols["name"]].map(text),
            "name_key": x[cols["name"]].map(name_key),
            "position_raw": x[cols["position"]].map(text),
            "roster_status": x[cols["status"]].map(text) if cols["status"] else "",
        })
        q["family"] = q["position_raw"].map(family)
        q = q.loc[
            q["season"].eq(int(requested))
            & q["week"].isin(WEEKS)
            & q["team"].ne("")
            & q["name_key"].ne("")
            & q["family"].ne("")
        ].copy()
        q["season"] = q["season"].astype(int)
        q["week"] = q["week"].astype(int)
        q["scheduled_team_week"] = [
            (int(s), int(w), str(t)) in scheduled_keys
            for s,w,t in zip(q["season"], q["week"], q["team"])
        ]
        q = q.loc[q["scheduled_team_week"]].copy()
        schemas[str(requested)] = cols
        out.append(q)
    y = pd.concat(out, ignore_index=True)
    y["history_key"] = np.where(
        y["stable_id"].astype(str).ne(""),
        "ID:" + y["stable_id"].astype(str),
        "NAME:" + y["name_key"].astype(str),
    )
    raw_duplicates = int(y.duplicated(["season","week","team","history_key"], keep=False).sum())
    y = y.sort_values(["season","week","team","history_key","player"], kind="stable")
    y = y.drop_duplicates(["season","week","team","history_key"], keep="last").reset_index(drop=True)
    y["roster_row_id"] = np.arange(len(y), dtype=int)
    post_duplicates = int(y.duplicated(["season","week","team","history_key"]).sum())
    return y, {
        "schemas": schemas,
        "raw_duplicate_rows": raw_duplicates,
        "post_dedup_duplicate_rows": post_duplicates,
    }


def source_scope_mask(
    x: pd.DataFrame,
    *,
    roster_ids_by_season: set[tuple[int,str]],
    roster_names_by_season_team: set[tuple[int,str,str]],
    season_col: str,
    team_col: str,
    stable_col: str | None,
    name_col: str,
) -> pd.Series:
    season = pd.to_numeric(x[season_col], errors="coerce")
    team = x[team_col].map(canon_team)
    sid = x[stable_col].map(stable) if stable_col else pd.Series("", index=x.index)
    nk = x[name_col].map(name_key)
    vals = []
    for s,t,i,n in zip(season,team,sid,nk):
        if pd.isna(s):
            vals.append(False)
            continue
        si=int(s)
        vals.append(
            (bool(i) and (si,str(i)) in roster_ids_by_season)
            or ((si,str(t),str(n)) in roster_names_by_season_team)
        )
    return pd.Series(vals, index=x.index)


def build_roster_maps(roster: pd.DataFrame):
    id_map = {}
    name_map = {}
    for key,g in roster.loc[roster["stable_id"].ne("")].groupby(["season","week","team","stable_id"]):
        id_map[key] = int(g["roster_row_id"].iloc[0]) if len(g)==1 else None
    for key,g in roster.groupby(["season","week","team","name_key"]):
        name_map[key] = int(g["roster_row_id"].iloc[0]) if len(g)==1 else None
    return id_map, name_map


def match_source_rows(
    src: pd.DataFrame,
    *,
    roster: pd.DataFrame,
    id_map: dict,
    name_map: dict,
    stable_field_present: bool,
) -> tuple[pd.DataFrame, dict]:
    q = src.copy()
    matched=[]
    methods=[]
    ambiguous=0
    for r in q.itertuples(index=False):
        sid=str(getattr(r,"stable_id","") or "")
        rid=None
        method=""
        if stable_field_present and sid:
            rid=id_map.get((int(r.season),int(r.week),str(r.team),sid))
            method="stable_id"
        else:
            key=(int(r.season),int(r.week),str(r.team),str(r.name_key))
            if key in name_map and name_map[key] is None:
                ambiguous += 1
            rid=name_map.get(key)
            method="name_team_fallback"
        matched.append(rid if rid is not None else -1)
        methods.append(method)
    q["roster_row_id"]=matched
    q["identity_method"]=methods
    q["matched"]=q["roster_row_id"].ge(0)
    return q, {
        "rows":int(len(q)),
        "matched_rows":int(q["matched"].sum()),
        "coverage":float(q["matched"].mean()) if len(q) else np.nan,
        "fallback_ambiguity_rows":int(ambiguous),
    }


def normalize_snaps(raw: pd.DataFrame, roster: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x=lower(raw)
    cols={
        "season":first_col(x,("season",)),
        "week":first_col(x,("week",)),
        "team":first_col(x,("team","team_abbr","club")),
        "stable_id":first_col(x,("gsis_id","player_id","nflverse_id")),
        "name":first_col(x,("player","player_name","full_name","name")),
        "offense_snaps":first_col(x,("offense_snaps",)),
        "offense_pct":first_col(x,("offense_pct","offense_percentage")),
    }
    miss=[k for k in ("season","week","team","name") if cols[k] is None]
    if miss or (cols["offense_snaps"] is None and cols["offense_pct"] is None):
        raise RuntimeError(f"snap source missing semantics {miss}")
    q=pd.DataFrame({
        "season":pd.to_numeric(x[cols["season"]],errors="coerce"),
        "week":pd.to_numeric(x[cols["week"]],errors="coerce"),
        "team":x[cols["team"]].map(canon_team),
        "stable_id":x[cols["stable_id"]].map(stable) if cols["stable_id"] else "",
        "player":x[cols["name"]].map(text),
        "name_key":x[cols["name"]].map(name_key),
        "offense_snaps":pd.to_numeric(x[cols["offense_snaps"]],errors="coerce") if cols["offense_snaps"] else np.nan,
        "offense_pct":pd.to_numeric(x[cols["offense_pct"]],errors="coerce") if cols["offense_pct"] else np.nan,
    })
    q=q.loc[q["season"].isin(SEASONS)&q["week"].isin(WEEKS)&q["team"].ne("")&q["name_key"].ne("")].copy()
    q["season"]=q["season"].astype(int); q["week"]=q["week"].astype(int)

    ids={(int(s),str(i)) for s,i in zip(roster["season"],roster["stable_id"]) if str(i)}
    names={(int(s),str(t),str(n)) for s,t,n in zip(roster["season"],roster["team"],roster["name_key"])}
    mask=[
        (bool(i) and (int(s),str(i)) in ids) or ((int(s),str(t),str(n)) in names)
        for s,t,i,n in zip(q["season"],q["team"],q["stable_id"],q["name_key"])
    ]
    q=q.loc[mask].copy()
    q["positive_participation"]=(q["offense_snaps"].fillna(0).gt(0)|q["offense_pct"].fillna(0).gt(0))
    source_dups=int(q.duplicated(["season","week","team","stable_id","name_key"],keep=False).sum())
    q=q.sort_values(["season","week","team","name_key"],kind="stable").drop_duplicates(
        ["season","week","team","stable_id","name_key"],keep="last"
    ).reset_index(drop=True)

    id_map,name_map=build_roster_maps(roster)
    m,match=match_source_rows(q,roster=roster,id_map=id_map,name_map=name_map,stable_field_present=bool(cols["stable_id"]))
    positive=m.loc[m["positive_participation"]].copy()
    positive_field=float(
        (positive["offense_snaps"].notna()|positive["offense_pct"].notna()).mean()
    ) if len(positive) else 0.0
    stable_cov=float(m["stable_id"].ne("").mean()) if cols["stable_id"] and len(m) else np.nan
    match.update({
        "columns":cols,
        "stable_id_field_present":bool(cols["stable_id"]),
        "stable_id_coverage":stable_cov,
        "source_duplicate_rows":source_dups,
        "positive_rows":int(len(positive)),
        "positive_join_coverage":float(positive["matched"].mean()) if len(positive) else 0.0,
        "positive_field_coverage":positive_field,
    })
    return m,match


def normalize_injuries(raw: pd.DataFrame, roster: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x=lower(raw)
    cols={
        "season":first_col(x,("season",)),
        "week":first_col(x,("week","report_week")),
        "team":first_col(x,("team","team_abbr","team_abbreviation","club")),
        "stable_id":first_col(x,("gsis_id","player_id","nflverse_id")),
        "name":first_col(x,("full_name","player_name","player","name")),
        "report_status":first_col(x,("report_status","game_status","status")),
        "practice_status":first_col(x,("practice_status","practice_participation")),
        "body_part":first_col(x,("report_primary_injury","primary_injury","injury","injury_type")),
    }
    miss=[k for k in ("season","week","team","name") if cols[k] is None]
    if miss:
        raise RuntimeError(f"injury source missing semantics {miss}")
    q=pd.DataFrame({
        "season":pd.to_numeric(x[cols["season"]],errors="coerce"),
        "week":pd.to_numeric(x[cols["week"]],errors="coerce"),
        "team":x[cols["team"]].map(canon_team),
        "stable_id":x[cols["stable_id"]].map(stable) if cols["stable_id"] else "",
        "player":x[cols["name"]].map(text),
        "name_key":x[cols["name"]].map(name_key),
        "report_status":x[cols["report_status"]].map(text) if cols["report_status"] else "",
        "practice_status":x[cols["practice_status"]].map(text) if cols["practice_status"] else "",
        "body_part":x[cols["body_part"]].map(text) if cols["body_part"] else "",
        "_order":np.arange(len(x)),
    })
    q=q.loc[q["season"].isin(SEASONS)&q["week"].isin(WEEKS)&q["team"].ne("")&q["name_key"].ne("")].copy()
    q["season"]=q["season"].astype(int); q["week"]=q["week"].astype(int)
    ids={(int(s),str(i)) for s,i in zip(roster["season"],roster["stable_id"]) if str(i)}
    names={(int(s),str(t),str(n)) for s,t,n in zip(roster["season"],roster["team"],roster["name_key"])}
    scope=[
        (bool(i) and (int(s),str(i)) in ids) or ((int(s),str(t),str(n)) in names)
        for s,t,i,n in zip(q["season"],q["team"],q["stable_id"],q["name_key"])
    ]
    q=q.loc[scope].copy()
    dedup_keys=["season","week","team","stable_id","name_key"]
    source_dups=int(q.duplicated(dedup_keys,keep=False).sum())
    q=q.sort_values("_order",kind="stable").drop_duplicates(dedup_keys,keep="last").reset_index(drop=True)

    id_map,name_map=build_roster_maps(roster)
    m,match=match_source_rows(q,roster=roster,id_map=id_map,name_map=name_map,stable_field_present=bool(cols["stable_id"]))
    joined=m.loc[m["matched"]].copy()
    has_info=(
        joined["report_status"].astype(str).str.strip().ne("")
        | joined["practice_status"].astype(str).str.strip().ne("")
        | joined["body_part"].astype(str).str.strip().ne("")
    )
    stable_cov=float(m["stable_id"].ne("").mean()) if cols["stable_id"] and len(m) else np.nan
    match.update({
        "columns":cols,
        "stable_id_field_present":bool(cols["stable_id"]),
        "stable_id_coverage":stable_cov,
        "source_duplicate_rows":source_dups,
        "joined_info_coverage":float(has_info.mean()) if len(joined) else 0.0,
    })
    return m,match


def construct_states(roster: pd.DataFrame, snaps: pd.DataFrame, injuries: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    y=roster.copy()
    # Aggregate matched snap evidence onto roster rows.
    sm=snaps.loc[snaps["matched"]].copy()
    if len(sm):
        sg=sm.groupby("roster_row_id",as_index=False).agg(
            offense_snaps=("offense_snaps","max"),
            offense_pct=("offense_pct","max"),
            positive_participation=("positive_participation","max"),
        )
        y=y.merge(sg,on="roster_row_id",how="left",validate="one_to_one")
    else:
        y["offense_snaps"]=np.nan; y["offense_pct"]=np.nan; y["positive_participation"]=False
    y["positive_participation"]=y["positive_participation"].fillna(False).astype(bool)

    im=injuries.loc[injuries["matched"]].copy()
    injury_ids=set(int(v) for v in im["roster_row_id"].unique())
    y["injury_reported"]=y["roster_row_id"].isin(injury_ids)

    y["observation_state"]=np.select(
        [
            y["positive_participation"] & ~y["injury_reported"],
            y["positive_participation"] & y["injury_reported"],
            ~y["positive_participation"] & y["injury_reported"],
        ],
        [
            STATE_NO_REPORT_PARTICIPATED,
            STATE_REPORTED_PARTICIPATED,
            STATE_REPORTED_NO_SNAPS,
        ],
        default=STATE_UNRESOLVED_NO_SNAPS,
    )
    if not y["observation_state"].isin(STATES).all():
        raise RuntimeError("observation-state construction emitted unknown state")

    y=y.sort_values(["season","team","history_key","week"],kind="stable").reset_index(drop=True)
    y["prior_observation_week"]=y.groupby(["season","team","history_key"])["week"].shift(1)
    y["prior_observation_state"]=y.groupby(["season","team","history_key"])["observation_state"].shift(1)
    chronology_bad=int(
        (
            y["prior_observation_week"].notna()
            & y["prior_observation_week"].ge(y["week"])
        ).sum()
    )
    y["return_after_reported_nonparticipation"]=(
        y["positive_participation"]
        & y["prior_observation_state"].eq(STATE_REPORTED_NO_SNAPS)
    )
    return y,{
        "chronology_violations":chronology_bad,
        "return_rows":int(y["return_after_reported_nonparticipation"].sum()),
    }


def season_summary(
    season:int,
    *,
    scheduled:pd.DataFrame,
    roster:pd.DataFrame,
    snaps:pd.DataFrame,
    injuries:pd.DataFrame,
    snap_audit:dict,
    injury_audit:dict,
    states:pd.DataFrame,
) -> dict:
    sch=scheduled.loc[scheduled["season"].eq(season)]
    r=roster.loc[roster["season"].eq(season)]
    st=states.loc[states["season"].eq(season)]
    sn=snaps.loc[snaps["season"].eq(season)]
    inj=injuries.loc[injuries["season"].eq(season)]

    stable_cov=float(r["stable_id"].ne("").mean()) if len(r) else 0.0
    pos_map_cov=float(r["family"].ne("").mean()) if len(r) else 0.0
    snap_pos=sn.loc[sn["positive_participation"]]
    snap_join=float(snap_pos["matched"].mean()) if len(snap_pos) else 0.0
    snap_fields=float(
        (snap_pos["offense_snaps"].notna()|snap_pos["offense_pct"].notna()).mean()
    ) if len(snap_pos) else 0.0
    inj_join=float(inj["matched"].mean()) if len(inj) else 0.0
    inj_joined=inj.loc[inj["matched"]]
    inj_info=float(
        (
            inj_joined["report_status"].astype(str).str.strip().ne("")
            |inj_joined["practice_status"].astype(str).str.strip().ne("")
            |inj_joined["body_part"].astype(str).str.strip().ne("")
        ).mean()
    ) if len(inj_joined) else 0.0

    state_counts={s:int((st["observation_state"]==s).sum()) for s in STATES}
    scheduled_set=set(zip(sch["week"],sch["team"]))
    bye_retained=sum((int(w),str(t)) not in scheduled_set for w,t in zip(r["week"],r["team"]))
    weeks=sorted(int(v) for v in sch["week"].unique())
    teams=int(sch["team"].nunique())

    gates={
        "schedule_reg_weeks_1_18":weeks==list(WEEKS),
        "schedule_all_32_teams":teams==32,
        "bye_week_rows_retained_zero":bye_retained==0,
        "roster_stable_id_coverage_ge_098":stable_cov>=0.98,
        "position_mapping_coverage_ge_099":pos_map_cov>=0.99,
        "roster_post_dedup_duplicates_zero":not r.duplicated(["season","week","team","history_key"]).any(),
        "snap_explicit_season_week":True,
        "snap_stable_id_coverage_ge_098_if_present":(
            True if not snap_audit["stable_id_field_present"] else float(sn["stable_id"].ne("").mean())>=0.98
        ),
        "positive_snap_roster_join_ge_098":snap_join>=0.98,
        "positive_snap_field_coverage_ge_099":snap_fields>=0.99,
        "snap_post_dedup_duplicates_zero":not sn.duplicated(["season","week","team","stable_id","name_key"]).any(),
        "injury_explicit_season_week":True,
        "injury_roster_join_ge_095":inj_join>=0.95,
        "injury_info_coverage_ge_095":inj_info>=0.95,
        "fallback_ambiguity_rows_zero":int(injury_audit["fallback_ambiguity_rows"])+int(snap_audit["fallback_ambiguity_rows"])==0,
        "all_roster_rows_state_labeled":len(st)==len(r) and st["observation_state"].isin(STATES).all(),
        "ordinary_participation_nonzero":state_counts[STATE_NO_REPORT_PARTICIPATED]>0,
        "reported_participation_nonzero":state_counts[STATE_REPORTED_PARTICIPATED]>0,
        "reported_no_snap_nonzero":state_counts[STATE_REPORTED_NO_SNAPS]>0,
        "return_rows_nonzero":int(st["return_after_reported_nonparticipation"].sum())>0,
    }
    by_family=[]
    for fam,g in st.groupby("family",sort=True):
        counts={s:int((g["observation_state"]==s).sum()) for s in STATES}
        by_family.append({
            "family":str(fam),
            "rows":int(len(g)),
            "players":int(g["history_key"].nunique()),
            "injury_reported_rows":int(g["injury_reported"].sum()),
            "positive_participation_rows":int(g["positive_participation"].sum()),
            "return_rows":int(g["return_after_reported_nonparticipation"].sum()),
            "state_counts":counts,
        })
    return {
        "season":season,
        "schedule_weeks":weeks,
        "schedule_teams":teams,
        "roster_rows":int(len(r)),
        "players":int(r["history_key"].nunique()),
        "roster_stable_id_coverage":stable_cov,
        "position_mapping_coverage":pos_map_cov,
        "positive_snap_rows":int(len(snap_pos)),
        "positive_snap_roster_join_coverage":snap_join,
        "positive_snap_field_coverage":snap_fields,
        "injury_rows_scoped":int(len(inj)),
        "injury_roster_join_coverage":inj_join,
        "injury_joined_info_coverage":inj_info,
        "state_counts":state_counts,
        "return_rows":int(st["return_after_reported_nonparticipation"].sum()),
        "bye_rows_retained":int(bye_retained),
        "family_summary":by_family,
        "gates":gates,
        "source_ready":bool(all(gates.values())),
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    import nflreadpy as nfl

    schedule_raw=pdx(nfl.load_schedules(seasons=list(SEASONS)))
    scheduled,schedule_audit=normalize_schedule(schedule_raw)

    roster_parts=[(s,pdx(nfl.load_rosters_weekly(int(s)))) for s in SEASONS]
    roster,roster_audit=normalize_rosters(roster_parts,scheduled)

    snaps,snap_audit=normalize_snaps(pdx(nfl.load_snap_counts(seasons=list(SEASONS))),roster)
    injuries,injury_audit=normalize_injuries(pdx(nfl.load_injuries(seasons=list(SEASONS))),roster)

    states,state_audit=construct_states(roster,snaps,injuries)
    summaries=[
        season_summary(
            s,scheduled=scheduled,roster=roster,snaps=snaps,injuries=injuries,
            snap_audit=snap_audit,injury_audit=injury_audit,states=states,
        )
        for s in SEASONS
    ]
    chronology_ok=state_audit["chronology_violations"]==0
    all_ready=all(s["source_ready"] for s in summaries) and chronology_ok

    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "disposition":(
            "PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_READY"
            if all_ready else
            "PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_BLOCKED"
        ),
        "schedule_audit":schedule_audit,
        "roster_audit":roster_audit,
        "snap_audit":snap_audit,
        "injury_audit":injury_audit,
        "state_audit":state_audit,
        "season_summaries":summaries,
        "predictive_relationships_scored":0,
        "outcome_production_fields_scored":0,
        "models_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
        "explicit_non_labels":[
            "injury_severity","early_exit","snap_limitation","healthy_full_strength",
            "role_decline","target_game_availability","future_production_effect",
        ],
    }

    states.to_csv(a.out_dir/"player_observation_health_state_rows.csv",index=False)
    injuries.loc[~injuries["matched"]].to_csv(a.out_dir/"injury_identity_exceptions.csv",index=False)
    pd.DataFrame([
        {
            "season":s["season"],
            "family":f["family"],
            "rows":f["rows"],
            "players":f["players"],
            "injury_reported_rows":f["injury_reported_rows"],
            "positive_participation_rows":f["positive_participation_rows"],
            "return_rows":f["return_rows"],
            **{f"state_{k}":v for k,v in f["state_counts"].items()},
        }
        for s in summaries for f in s["family_summary"]
    ]).to_csv(a.out_dir/"player_observation_health_state_family_summary.csv",index=False)
    (a.out_dir/"player_observation_health_state_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,default=json_default)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True,default=json_default))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
