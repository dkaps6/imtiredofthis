#!/usr/bin/env python3
"""Source-only audit of unresolved Phase-4B target identities.

No receiving-yard fields are loaded. This script does not impute target counts.
It only classifies unresolved weekly-stat identities using exact weekly roster
identity and exact candidate-authority overlap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.canonical_names import canonicalize_player_name_safe

CANDIDATE = "WR_R15_WR1_ANCHORED_PARTICIPATION"
TG = ["season", "week", "team"]
IDENT = TG + ["player_clean_key"]


def load_weekly_stats(seasons=(2023, 2024)) -> pd.DataFrame:
    import nflreadpy as nfl
    frames=[]
    for s in seasons:
        raw=nfl.load_player_stats(seasons=[int(s)], summary_level="week")
        x=_normalize_weekly(_to_pandas(raw), int(s))
        x=x.loc[pd.to_numeric(x["week"], errors="coerce").between(1,18)].copy()
        x["season"]=int(s)
        x["week"]=pd.to_numeric(x["week"],errors="raise").astype(int)
        x["team"]=x["team"].map(canon_team)
        x["player_clean_key"]=x["player_clean_key"].astype(str)
        x["targets"]=pd.to_numeric(x["targets"],errors="raise").astype(float)
        frames.append(x[IDENT+["targets"]])
    out=pd.concat(frames,ignore_index=True)
    if out.duplicated(IDENT).any():
        raise RuntimeError("weekly stats duplicate identity")
    return out


def _first(frame, candidates, default=""):
    for c in candidates:
        if c in frame.columns:
            return frame[c]
    return pd.Series(default,index=frame.index)


def load_weekly_rosters(seasons=(2023,2024)) -> pd.DataFrame:
    import nflreadpy as nfl
    frames=[]
    for s in seasons:
        raw=_to_pandas(nfl.load_rosters_weekly(int(s)))
        x=raw.copy()
        x.columns=[str(c).strip().lower() for c in x.columns]
        x["season"]=pd.to_numeric(_first(x,["season"],s),errors="coerce").fillna(s).astype(int)
        x["week"]=pd.to_numeric(_first(x,["week"]),errors="coerce")
        x=x.loc[x["season"].eq(int(s)) & x["week"].between(1,18)].copy()
        x["week"]=x["week"].astype(int)
        x["team"]=_first(x,["team","team_abbr","club_code"]).map(canon_team)
        raw_name=_first(x,["full_name","football_name","player_name","player","name"]).astype("string").fillna("").str.strip()
        canon=raw_name.map(canonicalize_player_name_safe)
        x["player"]=canon.map(lambda p:p[0])
        x["player_clean_key"]=canon.map(lambda p:p[1]).astype(str)
        x["player_id"]=_first(x,["gsis_id","player_id"]).astype("string").fillna("").str.strip()
        x=x.loc[x["team"].astype(str).str.len().gt(0)&x["player_clean_key"].astype(str).str.len().gt(0)].copy()
        frames.append(x[IDENT+["player","player_id"]])
    out=pd.concat(frames,ignore_index=True)
    counts=out.groupby(IDENT).size().rename("n_roster_rows").reset_index()
    one=counts.loc[counts["n_roster_rows"].eq(1),IDENT]
    out=out.merge(one,on=IDENT,how="inner",validate="many_to_one")
    out=out.drop_duplicates(IDENT)
    return out


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--features",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()

    pred=pd.read_csv(args.predictions,usecols=[
        "variant","season","week","team","player_clean_key","player","wr_rank","actual_targets"
    ])
    pred=pred.loc[pred["variant"].astype(str).eq(CANDIDATE)].copy()
    pred["season"]=pd.to_numeric(pred["season"],errors="raise").astype(int)
    pred["week"]=pd.to_numeric(pred["week"],errors="raise").astype(int)
    pred["team"]=pred["team"].map(canon_team)
    pred["player_clean_key"]=pred["player_clean_key"].astype(str)
    pred["wr_rank"]=pd.to_numeric(pred["wr_rank"],errors="raise").astype(int)
    pred["actual_targets"]=pd.to_numeric(pred["actual_targets"],errors="raise").astype(float)
    if pred.duplicated(IDENT).any(): raise RuntimeError("candidate duplicate identity")

    feat=pd.read_csv(args.features,usecols=[
        "season","week","team","player_clean_key","player","baseline_wr_rank"
    ])
    feat["season"]=pd.to_numeric(feat["season"],errors="raise").astype(int)
    feat["week"]=pd.to_numeric(feat["week"],errors="raise").astype(int)
    feat["team"]=feat["team"].map(canon_team)
    feat["player_clean_key"]=feat["player_clean_key"].astype(str)
    feat["baseline_wr_rank"]=pd.to_numeric(feat["baseline_wr_rank"],errors="raise").astype(int)
    feat=feat.loc[feat["baseline_wr_rank"].ge(2)].copy()
    if feat.duplicated(IDENT).any(): raise RuntimeError("feature duplicate identity")

    stats=load_weekly_stats()
    rosters=load_weekly_rosters()

    l4=feat[IDENT+["player"]].merge(stats,on=IDENT,how="left",validate="one_to_one",indicator=True)
    unresolved=l4.loc[l4["_merge"].eq("left_only"),IDENT+["player"]].copy()
    unresolved=unresolved.merge(
        rosters[IDENT+["player_id"]].assign(exact_weekly_roster_match=True),
        on=IDENT,how="left",validate="one_to_one"
    )
    unresolved["exact_weekly_roster_match"]=unresolved["exact_weekly_roster_match"].fillna(False).astype(bool)
    cand_secondary=pred.loc[pred["wr_rank"].ge(2),IDENT+["actual_targets"]].copy()
    unresolved=unresolved.merge(
        cand_secondary.rename(columns={"actual_targets":"authority_actual_targets"}),
        on=IDENT,how="left",validate="one_to_one"
    )
    unresolved["candidate_authority_overlap"]=unresolved["authority_actual_targets"].notna()
    unresolved["candidate_authority_zero_targets"]=(
        unresolved["candidate_authority_overlap"] & unresolved["authority_actual_targets"].abs().le(1e-9)
    )
    unresolved["candidate_authority_positive_targets"]=(
        unresolved["candidate_authority_overlap"] & unresolved["authority_actual_targets"].gt(1e-9)
    )

    anchors=pred.loc[pred["wr_rank"].eq(1),IDENT+["player","actual_targets"]].copy()
    ag=anchors[TG].drop_duplicates()
    sec=feat.merge(ag,on=TG,how="inner",validate="many_to_one")
    canonical=pd.concat([
        anchors[IDENT+["player","actual_targets"]].rename(columns={"actual_targets":"authority_actual_targets"}),
        sec[IDENT+["player"]].assign(authority_actual_targets=pd.NA),
    ],ignore_index=True)
    canonical=canonical.drop_duplicates(IDENT)
    l23=canonical.merge(stats,on=IDENT,how="left",validate="one_to_one",indicator=True)
    l23u=l23.loc[l23["_merge"].eq("left_only"),IDENT+["player","authority_actual_targets"]].copy()
    l23u=l23u.merge(
        rosters[IDENT+["player_id"]].assign(exact_weekly_roster_match=True),
        on=IDENT,how="left",validate="one_to_one"
    )
    l23u["exact_weekly_roster_match"]=l23u["exact_weekly_roster_match"].fillna(False).astype(bool)
    l23u=l23u.merge(
        pred[IDENT+["actual_targets"]].rename(columns={"actual_targets":"candidate_actual_targets"}),
        on=IDENT,how="left",validate="one_to_one"
    )
    l23u["candidate_authority_overlap"]=l23u["candidate_actual_targets"].notna()
    l23u["candidate_authority_zero_targets"]=(
        l23u["candidate_authority_overlap"] & l23u["candidate_actual_targets"].abs().le(1e-9)
    )
    l23u["candidate_authority_positive_targets"]=(
        l23u["candidate_authority_overlap"] & l23u["candidate_actual_targets"].gt(1e-9)
    )

    def summarize(u):
        return {
            "unresolved_rows":int(len(u)),
            "exact_weekly_roster_matches":int(u["exact_weekly_roster_match"].sum()),
            "exact_weekly_roster_match_pct":float(u["exact_weekly_roster_match"].mean()) if len(u) else 0.0,
            "candidate_authority_overlap":int(u["candidate_authority_overlap"].sum()),
            "candidate_authority_zero_targets":int(u["candidate_authority_zero_targets"].sum()),
            "candidate_authority_positive_targets":int(u["candidate_authority_positive_targets"].sum()),
            "roster_match_and_candidate_zero":int((u["exact_weekly_roster_match"]&u["candidate_authority_zero_targets"]).sum()),
            "no_roster_match_and_no_candidate_overlap":int((~u["exact_weekly_roster_match"]&~u["candidate_authority_overlap"]).sum()),
        }

    result={
        "specification":"WR_PHASE4B_UNRESOLVED_IDENTITY_SOURCE_AUDIT_V1",
        "layer4":summarize(unresolved),
        "layer23":summarize(l23u),
        "receiving_yard_fields_loaded":False,
        "zero_imputation_performed":False,
        "sportsbook_inputs":0,
        "production_change":False,
        "challenger_model_authorized":False,
    }
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    unresolved.to_csv(out/"layer4_unresolved_source_audit.csv",index=False)
    l23u.to_csv(out/"layer23_unresolved_source_audit.csv",index=False)
    (out/"unresolved_source_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
