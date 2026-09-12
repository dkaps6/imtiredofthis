"""Tests for the edge-threshold-aware full-stack Vegas benchmark grader.

Reuses the exact PLAY/LEAN gate formulas from
scripts/master_betting_workbook_core_v2.py (implied_prob, no_vig, ev_roi,
STRONG EDGE >= .05 EV and .03 prob edge, LEAN EDGE > 0 EV). Pure/deterministic:
no network, no file I/O.
"""
from __future__ import annotations

import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import (
    ev_roi,
    grade,
    implied_prob,
    no_vig,
    signal,
)


def test_implied_prob_matches_known_values():
    assert abs(implied_prob(-110) - (110 / 210)) < 1e-9
    assert abs(implied_prob(150) - (100 / 250)) < 1e-9


def test_no_vig_normalizes_pair():
    a, b = implied_prob(-110), implied_prob(-110)
    assert abs(no_vig(a, b) - 0.5) < 1e-9


def test_ev_roi_breakeven_at_market_implied_prob():
    p = implied_prob(-110)
    assert abs(ev_roi(p, -110)) < 1e-6


def test_signal_thresholds():
    assert signal(0.06, 0.04) == "STRONG_EDGE"
    assert signal(0.06, 0.02) == "LEAN_EDGE"  # EV clears but prob edge doesn't reach STRONG
    assert signal(0.01, 0.10) == "LEAN_EDGE"
    assert signal(-0.01, 0.10) == "NO_EDGE"


def _proj(mean=280.0, spread_component="tight"):
    ml = mean + (2.0 if spread_component == "tight" else 20.0)
    state = mean - (2.0 if spread_component == "tight" else 20.0)
    return pd.DataFrame(
        [
            {
                "season": 2024, "week": 1, "team": "KC", "player_clean_key": "patrickmahomes",
                "market": "pass_yards", "game_id": "2024_01_KC_BAL",
                "mc_proj": mean, "ml_proj": ml, "state_proj": state, "ensemble_proj": mean,
                "actual": 300.0,
            }
        ]
    )


def _props(over_odds=-110, under_odds=-110, line=265.5):
    return pd.DataFrame(
        [{
            "game_id": "2024_01_KC_BAL", "player_clean_key": "patrickmahomes", "market": "pass_yards",
            "book": "draftkings", "line": line, "over_odds": over_odds, "under_odds": under_odds,
            "player": "p.mahomes",
        }]
    )


def test_tight_component_agreement_produces_higher_confidence_strong_edge():
    tight, _ = grade(_proj(spread_component="tight"), _props())
    wide, _ = grade(_proj(spread_component="wide"), _props())
    assert tight.iloc[0]["best_ev"] > wide.iloc[0]["best_ev"]
    assert tight.iloc[0]["side"] == "OVER"


def test_summary_has_all_three_tiers_and_all_markets_rollup():
    _, summary = grade(_proj(), _props())
    assert set(summary.tier) == {"ALL_NO_FILTER", "LEAN_OR_STRONG", "STRONG_ONLY_PLAY_TIER"}
    assert "ALL_MARKETS" in set(summary.market)
