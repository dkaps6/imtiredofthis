#!/usr/bin/env python3
"""M95Q v4 mechanical wrapper: pandas-1.5-safe v3 role coverage aggregation."""
from __future__ import annotations

import scripts.backtest.evaluate_rb_m95q_expanded_historical_stable_workhorse_v3 as q

m = q.m


def enrich_stable_complete_v4(trace, hold_scored, seasons, aliases):
    g, k = m.g, m.k
    rosters, injuries, depth, provider_audit = g.load_provider_sources(seasons)
    rosters = q.bridge_provider_to_stats(rosters, aliases)
    injuries = q.bridge_provider_to_stats(injuries, aliases)
    depth = q.bridge_provider_to_stats(depth, aliases)
    rosters = g.add_roster_transition_features(rosters)
    if hasattr(g, "add_depth_transition_features"):
        depth = g.add_depth_transition_features(depth)

    base = hold_scored.copy()
    role_trace = trace.loc[trace["season"].isin(seasons)].copy()
    cov = base[m.PLAYER_KEYS].merge(
        rosters[m.PLAYER_KEYS + ["self_roster_present"]].drop_duplicates(m.PLAYER_KEYS),
        on=m.PLAYER_KEYS, how="left",
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


q.enrich_stable_complete = enrich_stable_complete_v4

if __name__ == "__main__":
    raise SystemExit(q.main())
