"""Regression guard for a real production incident (2026-09-13, live run
34775307362): football_qb_rows and current_wr1_anchor_rows are always the
full 32-team football-only universe count -- they are contracted to be
independent of sportsbook/kickoff-timing eligibility, per
scripts/slate_universe_v2.py's documented contract. `expected` (the certified
current-eligible team count) legitimately shrinks over the course of a game
day as more games kick off, so exact equality between the two is fatal on any
run where fewer than 32 teams are still pre-kickoff. Only fewer QB rows/WR
anchors than the eligible floor is a real bug.
"""
from pathlib import Path


def test_qb_c2_and_wr_r15_current_coverage_checks_are_a_floor_not_exact_match():
    workflow = Path(".github/workflows/full-slate.yml").read_text(encoding="utf-8")
    assert "football_qb_rows',0)) < expected" in workflow
    assert "football_qb_rows',0)) != expected" not in workflow
    assert "current_wr1_anchor_rows',0)) < expected" in workflow
    assert "current_wr1_anchor_rows',0)) != expected" not in workflow
