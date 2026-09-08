#!/usr/bin/env python3
"""RB-R15B mechanical team-code canonicalization wrapper.

The first R15 attempt failed before science because historical PBP team aliases
(e.g. relocation-era abbreviations) were not canonicalized before joining the
canonical schedule. This wrapper changes only that join key and then executes the
exact frozen R15 diagnostic.
"""
from __future__ import annotations

from scripts._opponent_map import canon_team
from scripts.backtest import diagnose_rb_r15_efficiency_context_v1 as r15

_ORIGINAL = r15._pbp_rb_games


def _canonical_pbp_games(logs, seasons):
    games, audit = _ORIGINAL(logs, seasons)
    before = sorted(set(games["team"].dropna().astype(str)))
    games = games.copy()
    games["team"] = games["team"].map(canon_team)
    after = sorted(set(games["team"].dropna().astype(str)))
    audit = dict(audit)
    audit.update({
        "mechanical_team_code_canonicalization": True,
        "team_codes_before": before,
        "team_codes_after": after,
        "football_features_changed": 0,
        "science_thresholds_changed": 0,
    })
    return games, audit


r15._pbp_rb_games = _canonical_pbp_games

if __name__ == "__main__":
    raise SystemExit(r15.main())
