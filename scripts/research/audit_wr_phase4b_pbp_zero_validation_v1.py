#!/usr/bin/env python3
"""Source-only PBP validation for Phase-4B missing weekly-stat rows.

Validates an official-target event counting rule against weekly player-stat
targets for exact roster identities, then audits rostered/stat-absent canonical
WR identities. No receiving-yard fields and no zero imputation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.canonical_names import canonicalize_player_name_safe

CANDIDATE="WR_R15_WR1_ANCHORED_PARTICIPATION"
TG=["season","week","team"]
IDENT=TG+["player_clean_key"]


def _first(frame,cands,default=""):
    for c in cands:
        if c in frame.columns: return frame[c]
    return pd.Series(default,index=frame.index)


def weekly_stats(seasons=(2023,2024)):
    import nflreadpy as nfl
    fs=[]
    for s in seasons:
        x=_normalize_weekly(_to_pandas(nfl.load_player_stats(seasons=[s],summary_level="week")),s)
        x=x.loc[pd.to_numeric(x["week"],errors="coerce").between(1,18)].copy()
        x["season"]=s; x["week"]=pd.to_numeric(x["week"],errors="raise").astype(int)
        x["team"]=x["team"].map(canon_team)
        x["player_clean_key"]=x["player_clean_key"].astype(str)
        x["targets"]=pd.to_numeric(x["targets"],errors="raise").astype(float)
        fs.append(x[IDENT+["targets"]])
    out=pd.concat(fs,ignore_index=True)
    if out.duplicated(IDENT).any(): raise RuntimeError("weekly stats duplicate")
    return out


def weekly_rosters(seasons=(2023,2024)):
    import nflreadpy as nfl
    fs=[]
    for s in seasons:
        x=_to_pandas(nfl.load_rosters_weekly(s)).copy()
        x.columns=[str(c).lower().strip() for c in x.columns]
        x["season"]=pd.to_numeric(_first(x,["season"],s),errors="coerce").fillna(s).astype(int)
        x["week"]=pd.to_numeric(_first(x,["week"]),errors="coerce")
        x=x.loc[x["season"].eq(s)&x["week"].between(1,18)].copy()
        x["week"]=x["week"].astype(int)
        x["team"]=_first(x,["team","team_abbr","club_code"]).map(canon_team)
        nm=_first(x,["full_name","football_name","player_name","player","name"]).astype("string").fillna("").str.strip()
        c=nm.map(canonicalize_player_name_safe)
        x["player_clean_key"]=c.map(lambda z:z[1]).astype(str)
        x["player_id"]=_first(x,["gsis_id","player_id"]).astype("string").fillna("").str.strip()
        x=x.loc[x["team"].astype(str).str.len().gt(0)&x["player_clean_key"].str.len().gt(0)&x["player_id"].str.len().gt(0)]
        fs.append(x[IDENT+["player_id"]])
    out=pd.concat(fs,ignore_index=True)
    counts=out.groupby(IDENT).size().rename("n").reset_index()
    one=counts.loc[counts["n"].eq(1),IDENT]
    out=out.merge(one,on=IDENT,how="inner",validate="many_to_one").drop_duplicates(IDENT)
    return out


def pbp_targets(seasons=(2023,2024)):
    import nflreadpy as nfl
    x=_to_pandas(nfl.load_pbp(seasons=list(seasons))).copy()
    x.columns=[str(c).lower().strip() for c in x.columns]
    required={"season","week","posteam","receiver_player_id","pass_attempt"}
    missing=required-set(x.columns)
    if missing: raise RuntimeError(f"PBP missing {sorted(missing)}")
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x=x.loc[x["season"].isin(seasons)&x["week"].between(1,18)].copy()
    x["season"]=x["season"].astype(int); x["week"]=x["week"].astype(int)
    x["team"]=x["posteam"].map(canon_team)
    x["receiver_player_id"]=x["receiver_player_id"].astype("string").fillna("").str.strip()
    pa=pd.to_numeric(x["pass_attempt"],errors="coerce").fillna(0).eq(1)
    valid=pa & x["receiver_player_id"].str.len().gt(0)
    if "two_point_attempt" in x.columns:
        valid &= ~pd.to_numeric(x["two_point_attempt"],errors="coerce").fillna(0).eq(1)
    if "no_play" in x.columns:
        valid &= ~pd.to_numeric(x["no_play"],errors="coerce").fillna(0).eq(1)
    q=x.loc[valid,["season","week","team","receiver_player_id"]].copy()
    return q.groupby(["season","week","team","receiver_player_id"],as_index=False).size().rename(columns={"size":"pbp_targets"})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--features",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    pred=pd.read_csv(a.predictions,usecols=["variant","season","week","team","player_clean_key","wr_rank"])
    pred=pred.loc[pred["variant"].astype(str).eq(CANDIDATE)].copy()
    pred["season"]=pd.to_numeric(pred["season"],errors="raise").astype(int)
    pred["week"]=pd.to_numeric(pred["week"],errors="raise").astype(int)
    pred["team"]=pred["team"].map(canon_team); pred["player_clean_key"]=pred["player_clean_key"].astype(str)
    pred["wr_rank"]=pd.to_numeric(pred["wr_rank"],errors="raise").astype(int)

    feat=pd.read_csv(a.features,usecols=["season","week","team","player_clean_key","baseline_wr_rank"])
    feat["season"]=pd.to_numeric(feat["season"],errors="raise").astype(int)
    feat["week"]=pd.to_numeric(feat["week"],errors="raise").astype(int)
    feat["team"]=feat["team"].map(canon_team); feat["player_clean_key"]=feat["player_clean_key"].astype(str)
    feat["baseline_wr_rank"]=pd.to_numeric(feat["baseline_wr_rank"],errors="raise").astype(int)
    feat=feat.loc[feat["baseline_wr_rank"].ge(2)].copy()

    stats=weekly_stats(); roster=weekly_rosters(); pt=pbp_targets()

    base=feat[IDENT].merge(roster,on=IDENT,how="left",validate="one_to_one")
    base["roster_resolved"]=base["player_id"].notna()
    base=base.merge(stats,on=IDENT,how="left",validate="one_to_one")
    base["weekly_stats_present"]=base["targets"].notna()
    base=base.merge(pt,left_on=TG+["player_id"],right_on=TG+["receiver_player_id"],how="left",validate="one_to_one")
    base["pbp_targets"]=pd.to_numeric(base["pbp_targets"],errors="coerce").fillna(0.0)

    validate=base.loc[base["roster_resolved"]&base["weekly_stats_present"]].copy()
    validate["target_delta"]=validate["pbp_targets"]-validate["targets"]
    exact=validate["target_delta"].abs().le(1e-9)

    absent=base.loc[base["roster_resolved"]&~base["weekly_stats_present"]].copy()
    zero_pbp=absent["pbp_targets"].abs().le(1e-9)

    anchors=pred.loc[pred["wr_rank"].eq(1),IDENT].copy()
    ag=anchors[TG].drop_duplicates()
    sec=feat.merge(ag,on=TG,how="inner",validate="many_to_one")
    canon23=pd.concat([anchors[IDENT],sec[IDENT]],ignore_index=True).drop_duplicates(IDENT)
    c23=canon23.merge(roster,on=IDENT,how="left",validate="one_to_one")
    c23["roster_resolved"]=c23["player_id"].notna()
    c23=c23.merge(stats,on=IDENT,how="left",validate="one_to_one")
    c23["weekly_stats_present"]=c23["targets"].notna()
    c23=c23.merge(pt,left_on=TG+["player_id"],right_on=TG+["receiver_player_id"],how="left",validate="one_to_one")
    c23["pbp_targets"]=pd.to_numeric(c23["pbp_targets"],errors="coerce").fillna(0.0)
    a23=c23.loc[c23["roster_resolved"]&~c23["weekly_stats_present"]].copy()

    result={
        "specification":"WR_PHASE4B_PBP_ZERO_VALIDATION_V1",
        "pbp_target_rule":"pass_attempt==1 AND receiver_player_id nonnull AND two_point_attempt!=1 AND no_play!=1",
        "weekly_stats_validation":{
            "resolved_roster_and_stats_rows":int(len(validate)),
            "exact_target_parity_rows":int(exact.sum()),
            "target_parity_fail_rows":int((~exact).sum()),
            "max_abs_target_delta":float(validate["target_delta"].abs().max()) if len(validate) else None,
        },
        "layer4_rostered_stat_absent":{
            "rows":int(len(absent)),
            "pbp_zero_target_rows":int(zero_pbp.sum()),
            "pbp_positive_target_rows":int((~zero_pbp).sum()),
            "max_pbp_targets":float(absent["pbp_targets"].max()) if len(absent) else 0.0,
        },
        "layer23_rostered_stat_absent":{
            "rows":int(len(a23)),
            "pbp_zero_target_rows":int(a23["pbp_targets"].abs().le(1e-9).sum()),
            "pbp_positive_target_rows":int(a23["pbp_targets"].gt(1e-9).sum()),
            "max_pbp_targets":float(a23["pbp_targets"].max()) if len(a23) else 0.0,
        },
        "receiving_yard_fields_loaded":False,
        "zero_imputation_performed":False,
        "sportsbook_inputs":0,
        "production_change":False,
    }
    out=a.out_dir; out.mkdir(parents=True,exist_ok=True)
    validate.loc[~exact].to_csv(out/"pbp_vs_weekly_target_parity_failures.csv",index=False)
    absent.loc[absent["pbp_targets"].gt(0)].to_csv(out/"layer4_rostered_stat_absent_positive_pbp_targets.csv",index=False)
    a23.loc[a23["pbp_targets"].gt(0)].to_csv(out/"layer23_rostered_stat_absent_positive_pbp_targets.csv",index=False)
    (out/"pbp_zero_validation.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
