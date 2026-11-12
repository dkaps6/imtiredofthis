"""Cushion is a gate, not a ranking key.

G3 cushion correctly REJECTS a rung that sits below our own projection. But
build_alt_cards_v1 also ORDERED by cushion, which keeps the deepest rung on
every ladder -- and deepest is most expensive. The ordering was therefore
anti-correlated with value, and the module's own "structural limit" (all 76
week-9 survivors between -225 and -1800, zero plus-money legs, three-leg cards
paying about +60) was largely an artifact of that rule.

Measured on the 2026 week-10 TNF board with real model_proj (run 43327080728),
every leg below passing all nine gates, own_rate shrunk toward market by the
measured PRISM overdispersion slope 0.159:

    cushion-ranked  2 legs   -506   needs 91.4%/leg   ROI  +3.0%
    cushion-ranked  3 legs   -466   needs 93.7%/leg   ROI  +0.3%
    value-ranked    2 legs   +115   needs 68.3%/leg   ROI +15.4%

The cushion-ranked cards need per-leg rates of 91-94%; the gates' own measured
week-9 leg hit rate was 88.2%.
"""
from __future__ import annotations

import pandas as pd

from scripts.research.build_alt_cards_v1 import (
    CUSHION_MIN, apply_gates, build_cards, leg_value,
)


def _leg(player, thr, price, proj, main, own_hits=8, own_games=8, market="rec_yards"):
    return {
        "player": player, "market": market, "book": "fanduel",
        "offered_american": price, "alt_threshold": thr, "model_proj": proj,
        "main_line": main, "frac_of_line": thr / main, "snap_pct": 0.70,
        "own_games": own_games, "own_hits": own_hits, "def_allowed": 500.0,
        "injury_status": "", "quarantined": False,
    }


def test_value_prefers_the_cheaper_rung_on_one_ladder():
    deep = _leg("Stefon Diggs", 9.5, -1400, 44.4, 38.5, own_hits=7, own_games=7)
    shallow = _leg("Stefon Diggs", 29.5, -182, 44.4, 38.5, own_hits=7, own_games=7)
    assert leg_value(shallow) > leg_value(deep)


def test_gates_keep_the_value_rung_not_the_deepest():
    df = pd.DataFrame([
        _leg("Stefon Diggs", 9.5, -1400, 44.4, 38.5, own_hits=7, own_games=7),
        _leg("Stefon Diggs", 29.5, -182, 44.4, 38.5, own_hits=7, own_games=7),
    ])
    surv, _rej = apply_gates(df, "fanduel")
    assert len(surv) == 1
    assert float(surv.iloc[0].alt_threshold) == 29.5
    assert float(surv.iloc[0].offered_american) == -182


def test_cushion_is_still_enforced_as_a_gate():
    """A cheap rung below CUSHION_MIN must still be rejected, not ranked up."""
    df = pd.DataFrame([_leg("Someone Else", 40.0, +120, 44.4, 38.5)])
    surv, rej = apply_gates(df, "fanduel")
    assert surv.empty
    assert "G3_cushion" in set(rej.gate)
    assert 44.4 / 40.0 < CUSHION_MIN


def test_a_bet_against_our_own_projection_is_still_refused():
    df = pd.DataFrame([_leg("Emeka Egbuka", 5.0, +100, 4.6, 20.0)])
    surv, rej = apply_gates(df, "fanduel")
    assert surv.empty
    detail = " ".join(rej.detail.astype(str))
    assert "BETTING AGAINST OUR OWN PROJECTION" in detail


def test_cards_are_built_cheapest_first_and_stay_disjoint():
    df = pd.DataFrame([
        _leg("Stefon Diggs", 29.5, -182, 44.4, 38.5, own_hits=7, own_games=7),
        _leg("Stefon Diggs", 9.5, -1400, 44.4, 38.5, own_hits=7, own_games=7),
        _leg("Malachi Fields", 14.5, -260, 33.1, 27.6),
        _leg("Malachi Fields", 4.5, -850, 33.1, 27.6),
        _leg("Antonio Williams", 14.5, -587, 34.2, 37.5),
        _leg("Rachaad White", 14.5, -1120, 32.7, 38.2, own_hits=7, own_games=7,
             market="rush_yards"),
        _leg("Isaiah Likely", 9.5, -2400, 43.7, 47.1),
    ])
    surv, _rej = apply_gates(df, "fanduel")
    cards = build_cards(surv)
    assert not cards.empty
    safe = cards.loc[cards.tier.eq("SAFE")]
    assert len(safe) == 2
    # The two best-value legs, at their cheap rungs.
    assert set(safe.player) == {"Stefon Diggs", "Malachi Fields"}
    assert set(safe.alt_threshold) == {29.5, 14.5}
    assert int(safe.card_odds.iloc[0]) > 0, "value ranking should produce plus money here"
    # No leg appears on two cards.
    assert not cards.player.duplicated().any()


def test_value_ranking_does_not_admit_an_excluded_market():
    df = pd.DataFrame([_leg("Malachi Fields", 1.5, -240, 2.5, 3.0, market="receptions")])
    surv, rej = apply_gates(df, "fanduel")
    assert surv.empty
    assert "G5_market" in set(rej.gate)
