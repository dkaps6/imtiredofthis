#!/usr/bin/env python3
"""Mechanical M95Q compatibility patch.

Run #1 exposed a deterministic M95A->M95B short-name bridge mismatch. Run #2
exposed pandas-1.5 aggregation syntax. Run #3 exposed that the original M95G
prior-leader helper both hard-coded 2024/2025 and unpacked TEAM_KEYS in a
legacy order. This wrapper fixes only those historical-reconstruction mechanics;
scientific definitions, frozen tail families and predeclared gates are unchanged.
"""
from __future__ import annotations

import pandas as pd

import scripts.backtest.evaluate_rb_m95q_expanded_historical_stable_workhorse as m


def build_matchup_trace_v2(logs, pbp_root, pfr_root, ngs_file):
    a, b, e, g = m.a, m.b, m.e, m.g

    a.TARGET_SEASONS = m.TRACE_SEASONS
    a.PBP_SEASONS = m.PBP_SEASONS
    b.TARGET_SEASONS = m.TRACE_SEASONS
    b.PBP_SEASONS = m.PBP_SEASONS

    apbp = a._read_pbp(pbp_root)
    schedule = a._schedule_from_pbp(apbp)
    team_games = a._team_game_from_logs(logs)
    rb_games = a._player_prior_features(logs, team_games)
    rb_allowed = a._rb_allowed_games(rb_games, schedule)
    pbp_def = a._pbp_defense_games(apbp)
    defense_games = pbp_def.merge(
        rb_allowed, on=["season", "week", "defense"], how="outer", validate="one_to_one"
    )
    metric_cols = [c for c in defense_games.columns if c not in {"season", "week", "defense"}]
    profiles = a._rolling_defense_profiles(defense_games, schedule, metric_cols)
    profiles = a._add_defense_composite(profiles)
    trace = a._truth_trace(rb_games, schedule, profiles)
    trace["player_clean_key"] = trace["player"].map(g.norm_name)
    trace["player_short_key"] = trace["player"].map(b.short_name)

    bpbp = b.read_pbp(pbp_root)
    pfr = b.read_pfr(pfr_root)
    ngs = b.read_ngs(ngs_file)
    x = b.add_offense(trace, bpbp, pfr, ngs)
    x = b.add_scores(x)
    x = e.add_priors(x)
    x["actual_20plus"] = m.num(x["actual_carries"]).ge(20).astype(int)
    x["actual_25plus"] = m.num(x["actual_carries"]).ge(25).astype(int)
    x["team"] = x["team"].map(g.canon)
    return x.reset_index(drop=True), profiles


def previous_team_leaders_historical(trace):
    """Exact M95G prior-game leader semantics, generalized across M95Q years."""
    g = m.g
    z = trace.copy()
    if "actual_carries" not in z.columns:
        if "actual_rush_att" in z.columns:
            z["actual_carries"] = m.num(z["actual_rush_att"])
        else:
            raise RuntimeError("M95Q trace missing actual carry truth for prior-game leader construction")
    rows = []
    # TEAM_KEYS is [season, week, team]; keep that exact key order.
    for (season, week, team), grp in z.groupby(g.TEAM_KEYS):
        q = grp.loc[m.num(grp["actual_carries"]).notna()].copy()
        if q.empty:
            continue
        q["actual_carries"] = m.num(q["actual_carries"])
        q = q.sort_values(["actual_carries", "player_clean_key"], ascending=[False, True])
        rows.append({
            "season": int(season),
            "week": int(week),
            "team": g.canon(team),
            "game_top1_key": str(q.iloc[0]["player_clean_key"]),
            "game_top1_carries": float(q.iloc[0]["actual_carries"]),
            "game_top2_key": str(q.iloc[1]["player_clean_key"]) if len(q) > 1 else "",
            "game_top2_carries": float(q.iloc[1]["actual_carries"]) if len(q) > 1 else 0.0,
        })
    if not rows:
        return pd.DataFrame(columns=g.TEAM_KEYS + [
            "prior_top1_key", "prior_top1_carries", "prior_top2_key", "prior_top2_carries"
        ])
    game = pd.DataFrame(rows).sort_values(["season", "team", "week"])
    grp = game.groupby(["season", "team"], sort=False)
    game["prior_top1_key"] = grp["game_top1_key"].shift(1)
    game["prior_top1_carries"] = grp["game_top1_carries"].shift(1)
    game["prior_top2_key"] = grp["game_top2_key"].shift(1)
    game["prior_top2_carries"] = grp["game_top2_carries"].shift(1)
    return game[g.TEAM_KEYS + [
        "prior_top1_key", "prior_top1_carries", "prior_top2_key", "prior_top2_carries"
    ]]


def enrich_stable_v2(trace, hold_scored, seasons):
    g, k = m.g, m.k
    rosters, injuries, depth, provider_audit = g.load_provider_sources(seasons)
    rosters = g.add_roster_transition_features(rosters)
    if hasattr(g, "add_depth_transition_features"):
        depth = g.add_depth_transition_features(depth)

    base = hold_scored.copy()
    role_trace = trace.loc[trace["season"].isin(seasons)].copy()
    cov = base[m.PLAYER_KEYS].merge(
        rosters[m.PLAYER_KEYS + ["self_roster_present"]].drop_duplicates(m.PLAYER_KEYS),
        on=m.PLAYER_KEYS,
        how="left",
    )
    coverage = (
        cov.groupby("season")["self_roster_present"]
        .agg(["size", lambda s: int(m.num(s).fillna(0).gt(0).sum())])
        .reset_index()
    )
    coverage.columns = ["season", "rows", "roster_matches"]
    coverage["roster_join_rate"] = coverage["roster_matches"] / coverage["rows"]

    z = g.enrich_base(base, role_trace, rosters, injuries, depth)
    z["stable_workhorse_m95k"] = k.stable_workhorse(z).astype(int)
    z["actual_20plus"] = m.num(z["actual_carries"]).ge(20).astype(int)
    z["actual_25plus"] = m.num(z["actual_carries"]).ge(25).astype(int)
    return z, provider_audit, coverage


m.build_matchup_trace = build_matchup_trace_v2
m.enrich_stable = enrich_stable_v2
m.g.previous_team_leaders = previous_team_leaders_historical

if __name__ == "__main__":
    raise SystemExit(m.main())
