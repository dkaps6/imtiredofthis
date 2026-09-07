#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
TARGET_SEASONS = [2021, 2022, 2023, 2024, 2025]
TEAM_MAP = {"OAK":"LV","SD":"LAC","STL":"LAR","LA":"LAR","JAC":"JAX","ARZ":"ARI","WSH":"WAS"}


def pdx(v) -> pd.DataFrame:
    if isinstance(v, pd.DataFrame): return v.copy()
    if hasattr(v, "to_pandas"): return v.to_pandas()
    if hasattr(v, "to_dicts"): return pd.DataFrame(v.to_dicts())
    return pd.DataFrame(v)


def lower(x: pd.DataFrame) -> pd.DataFrame:
    x=x.copy(); x.columns=[str(c).strip().lower() for c in x.columns]; return x


def key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def team(v) -> str:
    s=str(v).strip().upper() if not pd.isna(v) else ""
    return TEAM_MAP.get(s,s) if s not in {"", "NAN", "NONE", "<NA>"} else ""


def first(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    for c in cols:
        if c in df.columns: return df[c]
    return pd.Series(pd.NA,index=df.index)


def one(root: Path, name: str) -> Path:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected one {name} below {root}, got {len(h)}")
    return h[0]


def load_snaps() -> tuple[pd.DataFrame, float, list[int]]:
    import nflreadpy as nfl
    q=lower(pdx(nfl.load_snap_counts(seasons=SOURCE_SEASONS)))
    if "season" not in q or "week" not in q:
        raise RuntimeError("snap source missing season/week")
    q["season"]=pd.to_numeric(q["season"],errors="coerce")
    q["week"]=pd.to_numeric(q["week"],errors="coerce")
    q=q.loc[q["season"].isin(SOURCE_SEASONS)&q["week"].between(1,18)].copy()
    q["team"]=first(q,["team","team_abbr","club"]).map(team)
    q["player_key"]=first(q,["player","player_name","full_name"]).map(key)
    q["offense_pct"]=pd.to_numeric(first(q,["offense_pct","offense_percentage"]),errors="coerce")
    q["offense_snaps"]=pd.to_numeric(first(q,["offense_snaps"]),errors="coerce")
    q["ordinal"]=q["season"]*100+q["week"]
    q=q.loc[q["team"].ne("")&q["player_key"].ne("")].copy()
    keys=["season","week","team","player_key"]
    dup=float(q.duplicated(keys,keep=False).mean()) if len(q) else 1.0
    q=q.sort_values(keys,kind="stable").drop_duplicates(keys,keep="last").reset_index(drop=True)
    seasons=sorted(int(v) for v in q["season"].dropna().unique())
    return q,dup,seasons


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--joint-root",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    ref=lower(pd.read_csv(one(a.joint_root,"joint_v1_paired_player_casebook.csv"),low_memory=False))
    need={"season","week","team","player_clean_key","position_group"}
    if need-set(ref.columns): raise RuntimeError(f"reference missing {sorted(need-set(ref.columns))}")
    ref["season"]=pd.to_numeric(ref["season"],errors="coerce")
    ref["week"]=pd.to_numeric(ref["week"],errors="coerce")
    ref=ref.loc[ref["position_group"].eq("TE")&ref["season"].isin(TARGET_SEASONS)&ref["week"].between(1,18)].copy()
    ref["team"]=ref["team"].map(team)
    ref["player_key"]=ref["player_clean_key"].map(key)
    ref["ordinal"]=ref["season"]*100+ref["week"]
    ref=ref[["season","week","team","player_key","ordinal"]].drop_duplicates().reset_index(drop=True)

    snaps,dup_rate,seasons_present=load_snaps()
    source_keys=["season","week","team","player_key"]
    same=ref.merge(snaps[source_keys+["offense_pct","offense_snaps"]],on=source_keys,how="left",indicator=True,validate="one_to_one")
    same_game_match_rate=float(same["_merge"].eq("both").mean()) if len(same) else 0.0

    any_maps={k:g.sort_values("ordinal",kind="stable") for k,g in snaps.groupby("player_key",sort=False)}
    same_maps={k:g.sort_values("ordinal",kind="stable") for k,g in snaps.groupby(["player_key","team"],sort=False)}

    rows=[]; same_future=0
    for _,r in ref.iterrows():
        o=float(r.ordinal); pk=r.player_key; tm=r.team
        ah=any_maps.get(pk,pd.DataFrame())
        if len(ah): ah=ah.loc[ah["ordinal"].lt(o)]
        sh=same_maps.get((pk,tm),pd.DataFrame())
        if len(sh): sh=sh.loc[sh["ordinal"].lt(o)]
        if len(ah) and float(ah["ordinal"].max())>=o: same_future+=1
        if len(sh) and float(sh["ordinal"].max())>=o: same_future+=1
        a1=ah.tail(1); a3=ah.tail(3); s1=sh.tail(1); s3=sh.tail(3)
        rows.append({
            "prior_count_anyteam":int(len(ah)),"prior_count_same_team":int(len(sh)),
            "prior1_anyteam":bool(len(a1)>=1),"prior3_anyteam":bool(len(a3)>=3),
            "prior1_same_team":bool(len(s1)>=1),"prior3_same_team":bool(len(s3)>=3),
            "prior1_anyteam_offense_pct":float(pd.to_numeric(a1["offense_pct"],errors="coerce").iloc[-1]) if len(a1) and pd.notna(pd.to_numeric(a1["offense_pct"],errors="coerce").iloc[-1]) else np.nan,
            "prior1_anyteam_offense_snaps":float(pd.to_numeric(a1["offense_snaps"],errors="coerce").iloc[-1]) if len(a1) and pd.notna(pd.to_numeric(a1["offense_snaps"],errors="coerce").iloc[-1]) else np.nan,
            "prior3_anyteam_offense_pct":float(pd.to_numeric(a3["offense_pct"],errors="coerce").mean()) if len(a3)>=3 and pd.to_numeric(a3["offense_pct"],errors="coerce").notna().any() else np.nan,
            "prior3_anyteam_offense_snaps":float(pd.to_numeric(a3["offense_snaps"],errors="coerce").mean()) if len(a3)>=3 and pd.to_numeric(a3["offense_snaps"],errors="coerce").notna().any() else np.nan,
            "prior1_same_team_offense_pct":float(pd.to_numeric(s1["offense_pct"],errors="coerce").iloc[-1]) if len(s1) and pd.notna(pd.to_numeric(s1["offense_pct"],errors="coerce").iloc[-1]) else np.nan,
            "prior1_same_team_offense_snaps":float(pd.to_numeric(s1["offense_snaps"],errors="coerce").iloc[-1]) if len(s1) and pd.notna(pd.to_numeric(s1["offense_snaps"],errors="coerce").iloc[-1]) else np.nan,
        })
    out=pd.concat([ref.reset_index(drop=True),pd.DataFrame(rows)],axis=1)

    summary=[]
    for label,g in [("POOLED",out)]+[(str(y),out.loc[out["season"].eq(y)]) for y in TARGET_SEASONS]:
        summary.append({
            "season":label,"target_rows":int(len(g)),
            "prior1_anyteam_rate":float(g["prior1_anyteam"].mean()) if len(g) else 0.0,
            "prior3_anyteam_rate":float(g["prior3_anyteam"].mean()) if len(g) else 0.0,
            "prior1_same_team_rate":float(g["prior1_same_team"].mean()) if len(g) else 0.0,
            "prior3_same_team_rate":float(g["prior3_same_team"].mean()) if len(g) else 0.0,
            "prior1_anyteam_offense_pct_nonnull":float(g["prior1_anyteam_offense_pct"].notna().mean()) if len(g) else 0.0,
            "prior1_anyteam_offense_snaps_nonnull":float(g["prior1_anyteam_offense_snaps"].notna().mean()) if len(g) else 0.0,
            "prior3_anyteam_offense_pct_nonnull":float(g["prior3_anyteam_offense_pct"].notna().mean()) if len(g) else 0.0,
            "prior3_anyteam_offense_snaps_nonnull":float(g["prior3_anyteam_offense_snaps"].notna().mean()) if len(g) else 0.0,
            "median_prior_count_anyteam":float(g["prior_count_anyteam"].median()) if len(g) else 0.0,
            "median_prior_count_same_team":float(g["prior_count_same_team"].median()) if len(g) else 0.0,
        })
    s=pd.DataFrame(summary)
    pooled=s.loc[s.season.eq("POOLED")].iloc[0]
    season_rows=s.loc[s.season.ne("POOLED")]
    coverage_gates={
        "pooled_prior1_anyteam_ge075":bool(pooled.prior1_anyteam_rate>=.75),
        "pooled_prior3_anyteam_ge060":bool(pooled.prior3_anyteam_rate>=.60),
        "every_season_prior1_anyteam_ge065":bool((season_rows.prior1_anyteam_rate>=.65).all()),
        "both_prior1_fields_ge070":bool(pooled.prior1_anyteam_offense_pct_nonnull>=.70 and pooled.prior1_anyteam_offense_snaps_nonnull>=.70),
        "pooled_prior1_same_team_ge055":bool(pooled.prior1_same_team_rate>=.55),
    }
    integrity={
        "all_six_source_seasons":bool(seasons_present==SOURCE_SEASONS),
        "duplicate_rate_le001":bool(dup_rate<=.01),
        "zero_same_future_used":bool(same_future==0),
        "sportsbook_inputs_zero":True,
    }
    if not all(integrity.values()): disp="STRICT_PRIOR_TE_PARTICIPATION_INELIGIBLE"
    elif all(coverage_gates.values()): disp="STRICT_PRIOR_TE_PARTICIPATION_ELIGIBLE"
    else: disp="STRICT_PRIOR_TE_PARTICIPATION_PARTIAL_ONLY"
    result={
        "migration":"TE_R4_STRICT_PRIOR_PARTICIPATION_SOURCE",
        "source":"nflreadpy.load_snap_counts",
        "source_rows":int(len(snaps)),"source_seasons":seasons_present,
        "target_rows":int(len(out)),"target_seasons":TARGET_SEASONS,
        "duplicate_rate":dup_rate,"same_game_exact_key_match_rate":same_game_match_rate,
        "same_or_future_observations_used":same_future,
        "coverage_gates":coverage_gates,"integrity_gates":integrity,
        "sportsbook_inputs_used":False,"model_fitting_used":False,"production_changed":False,
        "disposition":disp,
    }
    a.out_dir.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.out_dir/"te_r4_prior_participation_casebook.csv",index=False)
    s.to_csv(a.out_dir/"te_r4_availability_summary.csv",index=False)
    (a.out_dir/"te_r4_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True)); print("\nSUMMARY"); print(s.to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
