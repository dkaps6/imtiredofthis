"""Full Slate data-quality classification must not hard-require 32 scheduled
teams. Bye weeks legitimately shrink the active week's schedule below 32 from
roughly Week 4 onward, and the classifier must certify that smaller slate
rather than crashing the whole live-pricing data-quality gate.
"""
from __future__ import annotations

from pathlib import Path


def test_schedule_check_accepts_bye_week_sized_slates_not_only_32():
    src = Path("scripts/validate_full_slate_data_quality_v1.py").read_text(encoding="utf-8")
    assert "if len(scheduled_teams) != 32:" not in src
    assert "len(scheduled_teams) % 2" in src
    assert 'f"teams={len(scheduled_teams)} games={len(scheduled_teams) // 2}"' in src


def test_injury_scope_ledger_check_uses_actual_scheduled_team_count():
    src = Path("scripts/validate_full_slate_data_quality_v1.py").read_text(encoding="utf-8")
    assert "if len(scope) != 32 or scope_teams != scheduled_teams:" not in src
    assert "if len(scope) != len(scheduled_teams) or scope_teams != scheduled_teams:" in src
    assert '"scheduled_teams_checked": 32,' not in src
    assert '"teams_with_report_rows": 32,' not in src
