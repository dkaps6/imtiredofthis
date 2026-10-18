#!/usr/bin/env python3
"""Score the frozen PRISM player high-value conversion identity V1 mechanism."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import audit_prism_player_high_value_conversion_identity_v1_source as src

VERSION = "PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_V1_SIGNAL"
CHANNELS = ("RB_RUSH_INSIDE10", "TE_RECEIVING_REDZONE")
TARGET_SEASONS = (2024, 2025)
MIN_PRIOR = 3
PRIOR_STRENGTH = 4.0
BOOT_REPS = 5000
BOOT_SEED = 20261017
EPS = 1e-9


def brier(y, p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    return float(np.mean((y-p)**2))


def logloss(y, p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))


def add_channel_baseline(events: pd.DataFrame) -> pd.DataFrame:
    w=(events.groupby(["channel","season","week"],as_index=False)
       .agg(channel_opps=("play_id","size"),channel_tds=("td_raw","sum"))
       .sort_values(["channel","season","week"],kind="stable")
       .reset_index(drop=True))
    w["prior_channel_opps"]=(
        w.groupby("channel",sort=False)["channel_opps"].cumsum()-w["channel_opps"]
    )
    w["prior_channel_tds"]=(
        w.groupby("channel",sort=False)["channel_tds"].cumsum()-w["channel_tds"]
    )
    w["p_base"]=(w.prior_channel_tds+0.5)/(w.prior_channel_opps+1.0)
    return w


def candidate_probability(prior_td, prior_opp, p_base):
    return (np.asarray(prior_td,dtype=float)+PRIOR_STRENGTH*np.asarray(p_base,dtype=float))/(
        np.asarray(prior_opp,dtype=float)+PRIOR_STRENGTH
    )


def cluster_ci(df, delta_col, cluster_col, reps=BOOT_REPS, seed=BOOT_SEED):
    z=(df.groupby(cluster_col,dropna=False)[delta_col]
       .agg(["sum","count"]).reset_index(drop=True))
    if len(z)<2:
        return [float("nan"),float("nan")]
    sums=z["sum"].to_numpy(float); counts=z["count"].to_numpy(float)
    rng=np.random.default_rng(seed)
    vals=np.empty(reps,dtype=float)
    n=len(z)
    for i in range(reps):
        idx=rng.integers(0,n,size=n)
        vals[i]=sums[idx].sum()/counts[idx].sum()
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def equal_player_ci(df, delta_col, reps=BOOT_REPS, seed=BOOT_SEED):
    v=(df.groupby("player_id",dropna=False)[delta_col].mean().to_numpy(float))
    if len(v)<2:
        return [float("nan"),float("nan")]
    rng=np.random.default_rng(seed)
    vals=np.empty(reps,dtype=float); n=len(v)
    for i in range(reps):
        vals[i]=rng.choice(v,size=n,replace=True).mean()
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def build_frames():
    meta=[]
    roster=src.hv.load_rosters(meta)
    events=pd.concat([src._base_events(s,roster) for s in src.SEASONS],ignore_index=True)
    events=events.loc[events.channel.isin(CHANNELS)].copy()
    games=src.add_prior_support(events)

    base=add_channel_baseline(events)[
        ["channel","season","week","prior_channel_opps","prior_channel_tds","p_base"]
    ]
    games=games.merge(base,on=["channel","season","week"],how="left",validate="many_to_one")
    games=games.loc[
        games.season.isin(TARGET_SEASONS)
        & games.prior_high_value_opps.ge(MIN_PRIOR)
        & games.p_base.notna()
    ].copy()
    games["p_player"]=candidate_probability(
        games.prior_touchdowns,games.prior_high_value_opps,games.p_base
    )
    games["actual_any_td"]=games.touchdowns.gt(0).astype(int)
    games["p_base_any"]=1.0-(1.0-games.p_base)**games.high_value_opps
    games["p_player_any"]=1.0-(1.0-games.p_player)**games.high_value_opps
    games["game_cluster"]=games.season.astype(str)+":"+games.game_id.astype(str)

    pred=games[["channel","season","week","game_id","team","player_id","p_base","p_player"]]
    event=events.merge(
        pred,
        on=["channel","season","week","game_id","team","player_id"],
        how="inner",
        validate="many_to_one",
    )
    event["actual_td"]=pd.to_numeric(event.td_raw,errors="coerce").astype(int)
    event["game_cluster"]=event.season.astype(str)+":"+event.game_id.astype(str)
    event["brier_base"]=(event.actual_td-event.p_base)**2
    event["brier_player"]=(event.actual_td-event.p_player)**2
    event["brier_gain"]=event.brier_base-event.brier_player

    games["brier_base"]=(games.actual_any_td-games.p_base_any)**2
    games["brier_player"]=(games.actual_any_td-games.p_player_any)**2
    games["brier_gain"]=games.brier_base-games.brier_player
    return event,games,meta


def score_cell(event, games, channel, season):
    e=event.loc[event.channel.eq(channel)&event.season.eq(season)].copy()
    g=games.loc[games.channel.eq(channel)&games.season.eq(season)].copy()
    if e.empty or g.empty:
        raise RuntimeError(f"empty score cell {channel} {season}")
    pp=e.groupby("player_id").brier_gain.mean()
    row={
        "channel":channel,
        "season":season,
        "event_rows":int(len(e)),
        "player_games":int(len(g)),
        "players":int(e.player_id.nunique()),
        "event_brier_base":brier(e.actual_td,e.p_base),
        "event_brier_player":brier(e.actual_td,e.p_player),
        "event_brier_gain":float(e.brier_gain.mean()),
        "event_logloss_base":logloss(e.actual_td,e.p_base),
        "event_logloss_player":logloss(e.actual_td,e.p_player),
        "event_logloss_gain":logloss(e.actual_td,e.p_base)-logloss(e.actual_td,e.p_player),
        "game_brier_base":brier(g.actual_any_td,g.p_base_any),
        "game_brier_player":brier(g.actual_any_td,g.p_player_any),
        "game_brier_gain":float(g.brier_gain.mean()),
        "game_logloss_base":logloss(g.actual_any_td,g.p_base_any),
        "game_logloss_player":logloss(g.actual_any_td,g.p_player_any),
        "game_logloss_gain":logloss(g.actual_any_td,g.p_base_any)-logloss(g.actual_any_td,g.p_player_any),
        "equal_player_event_brier_gain":float(pp.mean()),
        "median_player_event_brier_gain":float(pp.median()),
        "positive_player_share":float((pp>0).mean()),
    }
    if season==2025:
        row["event_player_cluster_ci"]=cluster_ci(e,"brier_gain","player_id",seed=BOOT_SEED+1)
        row["event_game_cluster_ci"]=cluster_ci(e,"brier_gain","game_cluster",seed=BOOT_SEED+2)
        row["event_equal_player_ci"]=equal_player_ci(e,"brier_gain",seed=BOOT_SEED+3)
        row["game_player_cluster_ci"]=cluster_ci(g,"brier_gain","player_id",seed=BOOT_SEED+4)
        row["game_game_cluster_ci"]=cluster_ci(g,"brier_gain","game_cluster",seed=BOOT_SEED+5)
    return row


def adjudicate(rows, channel):
    a={int(r["season"]):r for r in rows if r["channel"]==channel}
    d=a[2024]; c=a[2025]
    gates={
        "development_2024_event_brier_gain_gt_0":d["event_brier_gain"]>0,
        "development_2024_game_brier_gain_gt_0":d["game_brier_gain"]>0,
        "confirm_2025_event_brier_gain_gt_0":c["event_brier_gain"]>0,
        "confirm_2025_event_logloss_gain_ge_0":c["event_logloss_gain"]>=0,
        "confirm_2025_game_brier_gain_gt_0":c["game_brier_gain"]>0,
        "confirm_2025_game_logloss_gain_ge_0":c["game_logloss_gain"]>=0,
        "confirm_2025_event_player_ci_low_gt_0":c["event_player_cluster_ci"][0]>0,
        "confirm_2025_event_game_ci_low_gt_0":c["event_game_cluster_ci"][0]>0,
        "confirm_2025_event_equal_player_ci_low_gt_0":c["event_equal_player_ci"][0]>0,
        "confirm_2025_game_player_ci_low_gt_0":c["game_player_cluster_ci"][0]>0,
        "confirm_2025_game_game_ci_low_gt_0":c["game_game_cluster_ci"][0]>0,
    }
    qualified=all(gates.values())
    return {
        "channel":channel,
        "qualified":qualified,
        "disposition":(
            f"{channel}_PLAYER_CONVERSION_IDENTITY_QUALIFIED"
            if qualified else f"{channel}_PLAYER_CONVERSION_IDENTITY_CLOSED"
        ),
        "gates":gates,
        "failed_gates":[k for k,v in gates.items() if not v],
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    event,games,meta=build_frames()
    rows=[score_cell(event,games,ch,s) for ch in CHANNELS for s in TARGET_SEASONS]
    adjudications=[adjudicate(rows,ch) for ch in CHANNELS]
    q=sum(int(x["qualified"]) for x in adjudications)
    if q==len(CHANNELS):
        overall="PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_QUALIFIED_ALL"
    elif q:
        overall="PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_QUALIFIED_PARTIAL"
    else:
        overall="PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_CLOSED_ALL"
    result={
        "version":VERSION,
        "overall_disposition":overall,
        "frozen_prior_strength":PRIOR_STRENGTH,
        "frozen_min_prior_opportunities":MIN_PRIOR,
        "development_season":2024,
        "confirmation_season":2025,
        "adjudications":adjudications,
        "rows":rows,
        "rb_receiving_scored":False,
        "outcomes_2026_read":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }
    pd.DataFrame(rows).to_csv(a.out_dir/"score_metrics.csv",index=False)
    pd.DataFrame(adjudications).to_json(a.out_dir/"adjudications.json",orient="records",indent=2)
    event.to_csv(a.out_dir/"event_scores.csv",index=False)
    games.to_csv(a.out_dir/"conditional_game_scores.csv",index=False)
    pd.DataFrame(meta).to_csv(a.out_dir/"source_manifest.csv",index=False)
    (a.out_dir/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
