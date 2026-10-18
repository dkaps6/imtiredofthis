import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "audit_prism_wr_te_opportunity_bias_conservation_bridge_v1_source.py"
spec = importlib.util.spec_from_file_location("bridge_source", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)


def _authority():
    return pd.DataFrame([
        {
            "position": "WR", "season": 2024, "week": 5, "team": "AAA",
            "player_clean_key": "alpha", "pred_targets_bridge": 8.0,
            "pred_rec_yards_bridge": 64.0, "room_target_sum": 14.0, "room_size": 2,
        },
        {
            "position": "WR", "season": 2024, "week": 5, "team": "AAA",
            "player_clean_key": "bravo", "pred_targets_bridge": 6.0,
            "pred_rec_yards_bridge": 42.0, "room_target_sum": 14.0, "room_size": 2,
        },
        {
            "position": "WR", "season": 2024, "week": 6, "team": "BBB",
            "player_clean_key": "charlie", "pred_targets_bridge": 5.0,
            "pred_rec_yards_bridge": 35.0, "room_target_sum": 5.0, "room_size": 1,
        },
    ])


def _signal():
    return pd.DataFrame([
        {
            "position": "WR", "component": "OPPORTUNITY", "season": 2024,
            "week": 5, "team": "AAA", "player_clean_key": "alpha",
            "source_eligible": True, "hierarchical_signed_bias": 8.0,
        },
        {
            "position": "WR", "component": "OPPORTUNITY", "season": 2024,
            "week": 6, "team": "BBB", "player_clean_key": "charlie",
            "source_eligible": True, "hierarchical_signed_bias": -3.5,
        },
    ])


def test_summarize_requires_multi_player_room_for_bridge_eligibility(monkeypatch):
    monkeypatch.setitem(m.SUPPORT, ("WR", 2024), {"rows": 1, "players": 1})
    rec, eligible = m.summarize("WR", 2024, _authority(), _signal())
    assert rec["signal_supported_rows"] == 2
    assert rec["exact_matched_rows"] == 2
    assert rec["bridge_eligible_rows"] == 1
    assert rec["single_player_room_supported_rows"] == 1
    assert eligible.iloc[0]["player_clean_key"] == "alpha"


def test_bias_factor_is_positive_and_matches_frozen_formula(monkeypatch):
    monkeypatch.setitem(m.SUPPORT, ("WR", 2024), {"rows": 1, "players": 1})
    _, eligible = m.summarize("WR", 2024, _authority(), _signal())
    # exp(-8 / 64)
    assert eligible.iloc[0]["bias_factor"] == pytest.approx(0.8824969025845955)
    assert eligible.iloc[0]["bias_factor"] > 0


def test_exact_join_failure_is_visible(monkeypatch):
    monkeypatch.setitem(m.SUPPORT, ("WR", 2024), {"rows": 1, "players": 1})
    s = _signal().copy()
    s.loc[0, "player_clean_key"] = "not_present"
    rec, _ = m.summarize("WR", 2024, _authority(), s)
    assert rec["join_coverage"] < 1.0
    assert rec["source_gates"]["join_coverage_100pct"] is False
