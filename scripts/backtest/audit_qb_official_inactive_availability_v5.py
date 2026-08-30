#!/usr/bin/env python3
"""M78 v5: remove the invalid <=8 inactive-player assumption.

Run #21 proved exact identity completeness across every schedule window. The
only failed historical gate was an artificial 3-8 section-size bound. Official
NFL lists legitimately contained 11 PHI inactives in 2024 Week 18 and 9 PIT
inactives in 2025 Week 16. Section validity therefore has no upper-count bound;
it still requires >=3 candidate bullets, complete unique identities, and the
unchanged roster/position/coverage gates from v4.
"""
from __future__ import annotations

from scripts.backtest import audit_qb_official_inactive_availability_v4 as v4

_original_parse_article = v4.parse_article


def parse_article(url, season, snapshots, roster_idx):
    records, sections = _original_parse_article(url, season, snapshots, roster_idx)
    for section in sections:
        # NFL game-day lists can exceed eight (for example Week 18 rest lists).
        # Keep only the defensible lower bound used to reject stray prose lists.
        section["reasonable_count"] = int(section.get("candidate_bullets", 0)) >= 3
    return records, sections


v4.parse_article = parse_article

if __name__ == "__main__":
    raise SystemExit(v4.main())
