"""Regression tests for the Week-1-only RB P3 production boundary.

The canonical Full Slate explicitly disables P3 outside Week 1.  Downstream
pricing/governance must therefore certify an empty promoted scope rather than
requiring a context the workflow intentionally removed.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.modeling.rb_pricing_adapter_v1 import (
    RB_CONTEXT_COLUMNS,
    load_rb_context,
    rb_context_teams,
)
import scripts.run_pricing_with_full_roster_universe_v5_production as v5
import scripts.validate_rb_rush_rec_conservation_v1 as conservation


def test_schema_only_rb_context_represents_empty_promoted_scope(tmp_path):
    path = tmp_path / "rb_rush_synthesis_context.csv"
    pd.DataFrame(columns=RB_CONTEXT_COLUMNS).to_csv(path, index=False)

    context = load_rb_context(path)

    assert context.empty
    assert rb_context_teams(context) == set()
    assert "player_base_key" in context.columns


def test_nonweek1_v5_materializes_zero_row_scope_sentinel(tmp_path, monkeypatch):
    context_path = tmp_path / "rb_rush_synthesis_context.csv"
    audit_path = tmp_path / "rb_p3_nonweek1_scope_audit.json"
    # A stale Week-1 row must not survive into a Week-2 governance read.
    pd.DataFrame([{"team": "CHI", "week": 1, "rb_synthesis_proj": 50.0}]).to_csv(context_path, index=False)

    monkeypatch.setattr(v5, "resolve_week", lambda: 2)
    monkeypatch.setattr(v5, "RB_CONTEXT_PATH", context_path)
    monkeypatch.setattr(v5, "RB_NONWEEK1_SCOPE_AUDIT", audit_path)

    payload = v5._materialize_nonweek1_rb_p3_scope_sentinel()
    rebuilt = pd.read_csv(context_path)

    assert payload["disposition"] == "RB_P3_NOT_APPLICABLE_OUTSIDE_WEEK1_EMPTY_SCOPE"
    assert payload["week"] == 2
    assert payload["promoted_players"] == 0
    assert payload["promoted_teams"] == 0
    assert payload["sportsbook_inputs_used"] is False
    assert rebuilt.empty
    assert set(RB_CONTEXT_COLUMNS).issubset(rebuilt.columns)
    assert json.loads(audit_path.read_text(encoding="utf-8"))["football_values_fabricated"] is False


def _install_conservation_paths(tmp_path, monkeypatch, *, applied: int = 0):
    priced_path = tmp_path / "props_priced_clean.csv"
    meta_path = tmp_path / "rb_rush_rec_conservation_input_audit.json"
    out_path = tmp_path / "rb_rush_rec_conservation_final_audit.csv"
    pd.DataFrame([
        {
            "player": "Breece Hall",
            "team": "NYJ",
            "week": 2,
            "source_market": "player_rush_yds",
            "rb_synthesis_applied": applied,
        }
    ]).to_csv(priced_path, index=False)
    meta_path.write_text(
        json.dumps({
            "disposition": "NO_ELIGIBLE_RB_RUSH_REC_ROWS",
            "players": 0,
            "sportsbook_inputs_used": False,
            "out_of_p3_scope_teams": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(conservation, "PRICED", priced_path)
    monkeypatch.setattr(conservation, "CONSERVATION", meta_path)
    monkeypatch.setattr(conservation, "OUT", out_path)
    return out_path


def test_nonweek1_conservation_noop_is_certified(tmp_path, monkeypatch):
    out_path = _install_conservation_paths(tmp_path, monkeypatch, applied=0)

    assert conservation.main() == 0
    assert out_path.exists()
    assert pd.read_csv(out_path).empty


def test_nonweek1_conservation_still_fails_if_p3_was_applied(tmp_path, monkeypatch):
    _install_conservation_paths(tmp_path, monkeypatch, applied=1)

    with pytest.raises(RuntimeError, match="incorrectly claim RB P3 application"):
        conservation.main()
