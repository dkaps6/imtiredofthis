#!/usr/bin/env python3
"""Mechanical M95Q compatibility patch.

M95Q Run #1 reached the frozen M95B offense enrichment but the generalized
M95A reconstruction did not expose M95B's historical ``player_short_key``
bridge column. This wrapper adds only that pre-existing deterministic name key
before invoking the otherwise unchanged M95Q reconstruction.
"""
from __future__ import annotations

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
    # Exact deterministic bridge expected by the frozen M95B enrichment.
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


m.build_matchup_trace = build_matchup_trace_v2

if __name__ == "__main__":
    raise SystemExit(m.main())
