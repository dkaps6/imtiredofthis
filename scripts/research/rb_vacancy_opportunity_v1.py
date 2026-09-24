#!/usr/bin/env python3
"""RB Vacancy Opportunity V1: no-outcome vacancy-state constructor.

Research-only. Implements the frozen Phase-1 contract without target-game outcomes,
sportsbook inputs, coefficient fitting, or fuzzy identity matching.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TOL = 1e-10
FORBIDDEN = {"actual", "actual_rushes", "actual_rush_yards", "target_game_snaps", "line", "odds", "sportsbook", "bookmaker"}
STATE_COLUMNS = [
    "target_season", "target_week", "team", "successor_player_clean_key",
    "prior_offense_pct", "snap_source_season", "snap_source_week",
    "vacated_rush_share", "successor_weight", "transfer_rush_share",
    "unavailable_players", "unavailable_prior_rush_shares",
]
EXCLUSION_COLUMNS = ["team", "player_clean_key", "reason"]


def _norm_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _assert_no_forbidden(df: pd.DataFrame, label: str) -> None:
    bad = [c for c in df.columns if c.lower() in FORBIDDEN or c.lower().startswith("sportsbook_")]
    if bad:
        raise RuntimeError(f"{label} contains forbidden target/market fields: {bad}")


def _latest_prior(df: pd.DataFrame, target_season: int, target_week: int, value: str) -> pd.DataFrame:
    x = df.copy()
    x["season"] = _num(x["season"])
    x["week"] = _num(x["week"])
    x["team"] = _norm_team(x["team"])
    x["player_clean_key"] = x["player_clean_key"].astype(str).str.strip()
    x[value] = _num(x[value])
    prior = (x.season < target_season) | ((x.season == target_season) & (x.week < target_week))
    x = x.loc[prior & x[value].notna()].copy()
    x["ord"] = x.season * 100 + x.week
    x = x.sort_values(["team", "player_clean_key", "ord"]).drop_duplicates(["team", "player_clean_key"], keep="last")
    return x[["team", "player_clean_key", "season", "week", value]].rename(columns={"season": f"{value}_source_season", "week": f"{value}_source_week"})


def build_state(availability: pd.DataFrame, roles: pd.DataFrame, logs: pd.DataFrame, snaps: pd.DataFrame, target_season: int, target_week: int):
    for label, df in [("availability", availability), ("roles", roles), ("logs", logs), ("snaps", snaps)]:
        _assert_no_forbidden(df, label)
    req_av = {"team", "player_clean_key", "position_group", "definitive_unavailable", "final_availability_state"}
    req_roles = {"team", "player_clean_key", "position_group"}
    req_logs = {"season", "week", "team", "player_clean_key", "rush_share_game"}
    req_snaps = {"season", "week", "team", "player_clean_key", "offense_pct"}
    for label, df, req in [("availability", availability, req_av), ("roles", roles, req_roles), ("logs", logs, req_logs), ("snaps", snaps, req_snaps)]:
        miss = req - set(df.columns)
        if miss: raise RuntimeError(f"{label} missing {sorted(miss)}")

    av = availability.copy(); roles = roles.copy()
    for x in (av, roles):
        x["team"] = _norm_team(x["team"]); x["player_clean_key"] = x["player_clean_key"].astype(str).str.strip()
        x["position_group"] = x["position_group"].astype(str).str.upper().str.strip()
    av["definitive_unavailable"] = _num(av.definitive_unavailable).fillna(0).astype(int)
    if av.duplicated(["team", "player_clean_key"]).any(): raise RuntimeError("ambiguous availability identity")
    if roles.duplicated(["team", "player_clean_key"]).any(): raise RuntimeError("ambiguous role identity")

    rush = _latest_prior(logs, target_season, target_week, "rush_share_game")
    snap = _latest_prior(snaps, target_season, target_week, "offense_pct")
    back = av.loc[av.position_group.isin(["RB", "FB"])].copy()
    unavailable = back.loc[(back.definitive_unavailable == 1) & back.final_availability_state.astype(str).str.startswith("UNAVAILABLE_")].copy()
    active = roles.loc[roles.position_group.isin(["RB", "FB"])].merge(
        back[["team", "player_clean_key", "definitive_unavailable"]], on=["team", "player_clean_key"], how="left", validate="one_to_one")
    active = active.loc[active.definitive_unavailable.fillna(1).eq(0)].copy()

    unavailable = unavailable.merge(rush, on=["team", "player_clean_key"], how="left", validate="one_to_one")
    active = active.merge(snap, on=["team", "player_clean_key"], how="left", validate="one_to_one")
    rows=[]; exclusions=[]
    teams = sorted(unavailable.team.unique())
    for team in teams:
        u = unavailable.loc[unavailable.team.eq(team)].copy()
        missing_u = u.loc[u.rush_share_game.isna()]
        for _, r in missing_u.iterrows(): exclusions.append({"team":team,"player_clean_key":r.player_clean_key,"reason":"NO_PRIOR_RUSH_SHARE"})
        u = u.loc[u.rush_share_game.notna()].copy()
        if u.empty: continue
        a = active.loc[active.team.eq(team) & active.offense_pct.notna() & active.offense_pct.gt(0)].copy()
        if a.empty:
            exclusions.append({"team":team,"player_clean_key":"","reason":"NO_SUCCESSOR_PRIOR_SNAP_WEIGHT"}); continue
        v = float(u.rush_share_game.clip(lower=0).sum())
        denom = float(a.offense_pct.sum())
        if denom <= 0: exclusions.append({"team":team,"player_clean_key":"","reason":"NO_SUCCESSOR_PRIOR_SNAP_WEIGHT"}); continue
        for _, r in a.iterrows():
            w=float(r.offense_pct/denom)
            rows.append({"target_season":target_season,"target_week":target_week,"team":team,"successor_player_clean_key":r.player_clean_key,
                         "prior_offense_pct":float(r.offense_pct),"snap_source_season":int(r.offense_pct_source_season),"snap_source_week":int(r.offense_pct_source_week),
                         "vacated_rush_share":v,"successor_weight":w,"transfer_rush_share":v*w,
                         "unavailable_players":"|".join(sorted(u.player_clean_key.astype(str))),
                         "unavailable_prior_rush_shares":"|".join(f"{x:.12g}" for x in u.rush_share_game)})
    # Keep zero-event artifacts parseable and schema-stable. A legitimate NO_EVENT
    # cohort is scientific state, not an exceptional/empty-file condition.
    out=pd.DataFrame(rows, columns=STATE_COLUMNS); exc=pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS)
    if not out.empty:
        if (out.successor_weight < 0).any() or (out.transfer_rush_share < 0).any(): raise RuntimeError("negative transfer")
        chk=out.groupby("team",as_index=False).agg(weight_sum=("successor_weight","sum"),transfer_sum=("transfer_rush_share","sum"),vacated=("vacated_rush_share","first"))
        if not np.allclose(chk.weight_sum,1.0,atol=TOL,rtol=0): raise RuntimeError("successor weights do not conserve")
        if not np.allclose(chk.transfer_sum,chk.vacated,atol=TOL,rtol=0): raise RuntimeError("vacated rush share does not conserve")
        if not ((out.snap_source_season*100+out.snap_source_week) < (target_season*100+target_week)).all(): raise RuntimeError("non-prior snap source")
    return out, exc


def main():
    p=argparse.ArgumentParser(); p.add_argument("--availability",required=True); p.add_argument("--roles",required=True); p.add_argument("--logs",required=True); p.add_argument("--snaps",required=True); p.add_argument("--season",type=int,required=True); p.add_argument("--week",type=int,required=True); p.add_argument("--outdir",required=True)
    a=p.parse_args(); outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
    out,exc=build_state(pd.read_csv(a.availability),pd.read_csv(a.roles),pd.read_csv(a.logs),pd.read_csv(a.snaps),a.season,a.week)
    out.to_csv(outdir/"rb_vacancy_state_no_outcomes.csv",index=False); exc.to_csv(outdir/"rb_vacancy_exclusions.csv",index=False)
    print(f"candidate_rows={len(out)} exclusions={len(exc)} teams={out.team.nunique() if not out.empty else 0}")
if __name__ == "__main__": main()
