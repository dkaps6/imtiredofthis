#!/usr/bin/env python3
"""Score frozen PRISM RB short-yardage carry-type identity V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import audit_prism_rb_short_yardage_carry_type_identity_v1_source as src

VERSION = "PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_V1_SIGNAL"
TARGET_SEASONS = (2024, 2025)
MIN_PRIOR_SAME_TEAM_CARRIES = 10
PRIOR_STRENGTH = 20.0
BOOT_REPS = 5000
BOOT_SEED = 20261017
EPS = 1e-9


def brier(y, p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    return float(np.mean((y-p)**2))


def logloss(y, p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))


def add_team_prior(player_games: pd.DataFrame) -> pd.DataFrame:
    """Attach strictly-prior same-team RB carry-type baseline to each player-game."""
    tg=(player_games.groupby(["season","week","game_id","team"],as_index=False)
        .agg(team_rb_carries=("carries","sum"),
             team_rb_short_yardage_carries=("short_yardage_carries","sum"))
        .sort_values(["team","season","week","game_id"],kind="stable")
        .reset_index(drop=True))
    tg["prior_team_rb_carries"]=(
        tg.groupby("team",sort=False)["team_rb_carries"].cumsum()-tg["team_rb_carries"]
    )
    tg["prior_team_rb_short_yardage_carries"]=(
        tg.groupby("team",sort=False)["team_rb_short_yardage_carries"].cumsum()
        -tg["team_rb_short_yardage_carries"]
    )
    tg["p_team"]=(
        tg.prior_team_rb_short_yardage_carries+0.5
    )/(tg.prior_team_rb_carries+1.0)
    return player_games.merge(
        tg[["season","week","game_id","team","prior_team_rb_carries",
            "prior_team_rb_short_yardage_carries","p_team"]],
        on=["season","week","game_id","team"],how="left",validate="many_to_one",
    )


def candidate_probability(prior_short, prior_carries, p_team):
    return (
        np.asarray(prior_short,dtype=float)
        + PRIOR_STRENGTH*np.asarray(p_team,dtype=float)
    )/(
        np.asarray(prior_carries,dtype=float)+PRIOR_STRENGTH
    )


def cluster_ci(df, delta_col, cluster_col, reps=BOOT_REPS, seed=BOOT_SEED):
    z=(df.groupby(cluster_col,dropna=False)[delta_col]
       .agg(["sum","count"]).reset_index(drop=True))
    if len(z)<2:
        return [float("nan"),float("nan")]
    sums=z["sum"].to_numpy(float); counts=z["count"].to_numpy(float)
    rng=np.random.default_rng(seed); n=len(z)
    vals=np.empty(reps,dtype=float)
    for i in range(reps):
        idx=rng.integers(0,n,size=n)
        vals[i]=sums[idx].sum()/counts[idx].sum()
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def equal_player_ci(df, delta_col, reps=BOOT_REPS, seed=BOOT_SEED):
    v=df.groupby("player_id",dropna=False)[delta_col].mean().to_numpy(float)
    if len(v)<2:
        return [float("nan"),float("nan")]
    rng=np.random.default_rng(seed); n=len(v)
    vals=np.empty(reps,dtype=float)
    for i in range(reps):
        vals[i]=rng.choice(v,size=n,replace=True).mean()
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def build_frames():
    meta=[]
    roster=src.hv.load_rosters(meta)
    events=[]
    source_metrics=[]
    for season in src.SEASONS:
        e,m=src.load_events(season,roster)
        events.append(e)
        source_metrics.append(m)
    events=pd.concat(events,ignore_index=True)

    games=src.add_prior_support(events)
    games=add_team_prior(games)
    games=games.loc[
        games.season.isin(TARGET_SEASONS)
        & games.prior_same_team_carries.ge(MIN_PRIOR_SAME_TEAM_CARRIES)
        & games.p_team.notna()
    ].copy()

    games["p_player"]=candidate_probability(
        games.prior_same_team_short_yardage_carries,
        games.prior_same_team_carries,
        games.p_team,
    )
    games["actual_any_short"]=games.short_yardage_carries.gt(0).astype(int)
    games["p_team_any"]=1.0-(1.0-games.p_team)**games.carries
    games["p_player_any"]=1.0-(1.0-games.p_player)**games.carries
    games["game_cluster"]=games.season.astype(str)+":"+games.game_id.astype(str)

    pred=games[["season","week","game_id","team","player_id","p_team","p_player"]]
    event=events.merge(
        pred,
        on=["season","week","game_id","team","player_id"],
        how="inner",
        validate="many_to_one",
    )
    event["actual_short"]=pd.to_numeric(event.short_yardage,errors="coerce").astype(int)
    event["game_cluster"]=event.season.astype(str)+":"+event.game_id.astype(str)
    event["brier_base"]=(event.actual_short-event.p_team)**2
    event["brier_player"]=(event.actual_short-event.p_player)**2
    event["brier_gain"]=event.brier_base-event.brier_player

    games["brier_base"]=(games.actual_any_short-games.p_team_any)**2
    games["brier_player"]=(games.actual_any_short-games.p_player_any)**2
    games["brier_gain"]=games.brier_base-games.brier_player
    return event,games,meta,source_metrics


def score_season(event, games, season):
    e=event.loc[event.season.eq(season)].copy()
    g=games.loc[games.season.eq(season)].copy()
    if e.empty or g.empty:
        raise RuntimeError(f"empty score season {season}")
    pp=e.groupby("player_id").brier_gain.mean()
    row={
        "season":season,
        "carry_rows":int(len(e)),
        "player_games":int(len(g)),
        "players":int(e.player_id.nunique()),
        "carry_brier_base":brier(e.actual_short,e.p_team),
        "carry_brier_player":brier(e.actual_short,e.p_player),
        "carry_brier_gain":float(e.brier_gain.mean()),
        "carry_logloss_base":logloss(e.actual_short,e.p_team),
        "carry_logloss_player":logloss(e.actual_short,e.p_player),
        "carry_logloss_gain":logloss(e.actual_short,e.p_team)-logloss(e.actual_short,e.p_player),
        "game_brier_base":brier(g.actual_any_short,g.p_team_any),
        "game_brier_player":brier(g.actual_any_short,g.p_player_any),
        "game_brier_gain":float(g.brier_gain.mean()),
        "game_logloss_base":logloss(g.actual_any_short,g.p_team_any),
        "game_logloss_player":logloss(g.actual_any_short,g.p_player_any),
        "game_logloss_gain":logloss(g.actual_any_short,g.p_team_any)-logloss(g.actual_any_short,g.p_player_any),
        "equal_player_carry_brier_gain":float(pp.mean()),
        "median_player_carry_brier_gain":float(pp.median()),
        "positive_player_share":float((pp>0).mean()),
    }
    if season==2025:
        row["carry_player_cluster_ci"]=cluster_ci(e,"brier_gain","player_id",seed=BOOT_SEED+1)
        row["carry_game_cluster_ci"]=cluster_ci(e,"brier_gain","game_cluster",seed=BOOT_SEED+2)
        row["carry_equal_player_ci"]=equal_player_ci(e,"brier_gain",seed=BOOT_SEED+3)
        row["game_player_cluster_ci"]=cluster_ci(g,"brier_gain","player_id",seed=BOOT_SEED+4)
        row["game_game_cluster_ci"]=cluster_ci(g,"brier_gain","game_cluster",seed=BOOT_SEED+5)
    return row


def adjudicate(rows):
    a={int(r["season"]):r for r in rows}
    d=a[2024]; c=a[2025]
    gates={
        "development_2024_carry_brier_gain_gt_0":d["carry_brier_gain"]>0,
        "development_2024_game_brier_gain_gt_0":d["game_brier_gain"]>0,
        "confirm_2025_carry_brier_gain_gt_0":c["carry_brier_gain"]>0,
        "confirm_2025_carry_logloss_gain_ge_0":c["carry_logloss_gain"]>=0,
        "confirm_2025_game_brier_gain_gt_0":c["game_brier_gain"]>0,
        "confirm_2025_game_logloss_gain_ge_0":c["game_logloss_gain"]>=0,
        "confirm_2025_carry_player_ci_low_gt_0":c["carry_player_cluster_ci"][0]>0,
        "confirm_2025_carry_game_ci_low_gt_0":c["carry_game_cluster_ci"][0]>0,
        "confirm_2025_carry_equal_player_ci_low_gt_0":c["carry_equal_player_ci"][0]>0,
        "confirm_2025_game_player_ci_low_gt_0":c["game_player_cluster_ci"][0]>0,
        "confirm_2025_game_game_ci_low_gt_0":c["game_game_cluster_ci"][0]>0,
    }
    qualified=all(gates.values())
    return {
        "qualified":qualified,
        "disposition":(
            "RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_SUPPORTED"
            if qualified else "RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_CLOSED"
        ),
        "gates":gates,
        "failed_gates":[k for k,v in gates.items() if not v],
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    event,games,meta,source_metrics=build_frames()
    rows=[score_season(event,games,s) for s in TARGET_SEASONS]
    adj=adjudicate(rows)
    result={
        "version":VERSION,
        "overall_disposition":adj["disposition"],
        "qualified":adj["qualified"],
        "frozen_definition":{"downs":[3,4],"ydstogo_min_exclusive":0,"ydstogo_max_inclusive":2},
        "frozen_min_prior_same_team_carries":MIN_PRIOR_SAME_TEAM_CARRIES,
        "frozen_prior_strength":PRIOR_STRENGTH,
        "development_season":2024,
        "confirmation_season":2025,
        "source_authority_run":39094323921,
        "source_authority_head":"5dbfe88491a7398bb9ce34450a5dffb22bda7a84",
        "source_authority_artifact":12123703725,
        "adjudication":adj,
        "rows":rows,
        "outcomes_2026_read":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }
    pd.DataFrame(rows).to_csv(a.out_dir/"score_metrics.csv",index=False)
    event.to_csv(a.out_dir/"carry_scores.csv",index=False)
    games.to_csv(a.out_dir/"conditional_game_scores.csv",index=False)
    pd.DataFrame(meta).to_csv(a.out_dir/"source_manifest.csv",index=False)
    pd.DataFrame(source_metrics).to_csv(a.out_dir/"source_metrics_recheck.csv",index=False)
    (a.out_dir/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
