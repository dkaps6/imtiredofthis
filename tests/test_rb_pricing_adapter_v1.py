from pathlib import Path

import pandas as pd
import pytest

from scripts.modeling.rb_pricing_adapter_v1 import load_rb_context, lookup_rb_projection


def _context() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "season": 2026,
            "week": 1,
            "player": "D'Andre Swift",
            "player_clean_key": "dandreswift",
            "team": "CHI",
            "opponent": "CAR",
            "rb_synthesis_proj": 51.25,
            "rb_synthesis_route": "WEEK1_STACK_OVERRIDE",
            "rb_synthesis_version": "RB_P3_SYNTHESIS_V1",
            "rb_synthesis_applied": 1,
            "football_only_no_odds": 1,
            "sportsbook_inputs_used": 0,
            "rb_stack_implied_ypc": 4.1,
            "rb_ypc_fallback_used": 0,
        }
    ])


def test_lookup_matches_punctuation_normalized_identity():
    row = pd.Series({
        "season": 2026,
        "week": 1,
        "player": "D’Andre Swift",
        "player_clean_key": "D'Andre-Swift",
        "team": "chi",
        "opponent": "car",
    })
    got = lookup_rb_projection(row, _context())
    assert got["rb_synthesis_proj"] == pytest.approx(51.25)
    assert got["rb_synthesis_route"] == "WEEK1_STACK_OVERRIDE"
    assert got["rb_synthesis_applied"] == 1


def test_lookup_fails_closed_on_missing_identity():
    row = pd.Series({"season": 2026, "week": 1, "player": "Nobody", "team": "CHI", "opponent": "CAR"})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        lookup_rb_projection(row, _context())


def test_lookup_fails_closed_on_opponent_mismatch():
    row = pd.Series({"season": 2026, "week": 1, "player": "D'Andre Swift", "team": "CHI", "opponent": "GB"})
    with pytest.raises(RuntimeError, match="opponent mismatch"):
        lookup_rb_projection(row, _context())


def test_load_rejects_sportsbook_leakage(tmp_path: Path):
    bad = _context()
    bad.loc[:, "sportsbook_inputs_used"] = 1
    path = tmp_path / "rb.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="sportsbook leakage"):
        load_rb_context(path)


def test_load_rejects_duplicate_identity(tmp_path: Path):
    bad = pd.concat([_context(), _context()], ignore_index=True)
    path = tmp_path / "rb.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="duplicate promoted RB context identities"):
        load_rb_context(path)


def test_canonical_pricing_and_full_slate_are_wired_to_p3():
    pricing = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/full-slate.yml").read_text(encoding="utf-8")
    assert "load_rb_context" in pricing
    assert "lookup_rb_projection" in pricing
    assert "target_mean = rb_synthesis_proj" in pricing
    assert "promoted RB production pricing is currently locked to Week 1" in pricing
    assert "Build promoted RB P3 football-only context" in workflow
    assert "final RB model projection is not the promoted P3 synthesis mean" in workflow
