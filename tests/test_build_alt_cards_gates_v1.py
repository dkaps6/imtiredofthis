"""Gate tests for scripts/research/build_alt_cards_v1.py.

Each test pins a gate to the week-9 failure that produced it. These are
regression tests against specific bad bets, not abstract unit tests.
"""
import pandas as pd
import pytest

from scripts.research.build_alt_cards_v1 import apply_gates, build_cards


def _leg(**kw):
    base = dict(player="Base Player", market="rec_yards", alt_threshold=10.0,
                book="fanduel", offered_american=-200.0, main_line=25.0,
                frac_of_line=0.40, model_proj=20.0, own_games=7, own_hits=6,
                snap_pct=0.70, def_allowed=120.0, injury_status=None,
                quarantined=False, team="AAA", opponent="BBB", event_id="e1")
    base.update(kw)
    return base


def _gates(rows, book="fanduel"):
    return apply_gates(pd.DataFrame(rows), book)


def test_clean_leg_survives():
    surv, rej = _gates([_leg()])
    assert len(surv) == 1 and rej.empty
    assert surv.cushion.iat[0] == pytest.approx(2.0)


def test_g1_rejects_third_string_snap_share():
    """Roschon Johnson: 15.6% snaps, CHI RB3, shipped as a 3-method clear."""
    surv, rej = _gates([_leg(player="Roschon Johnson", market="rush_yards",
                             main_line=16.5, alt_threshold=9.5, model_proj=22.5,
                             frac_of_line=0.58, snap_pct=0.156)])
    assert surv.empty
    assert rej.gate.iat[0] == "G1_snap"


def test_g1_rejects_unknown_snap_share():
    surv, rej = _gates([_leg(snap_pct=None)])
    assert surv.empty and rej.gate.iat[0] == "G1_snap_unknown"


def test_g2_rejects_low_main_line_role():
    """A 4.5-yard receiving rung: the book saying he is not in the offense."""
    surv, rej = _gates([_leg(main_line=4.5, alt_threshold=2.5, model_proj=12.0,
                             frac_of_line=0.56)])
    assert surv.empty and rej.gate.iat[0] == "G2_role"


def test_g3_rejects_cushion_below_one_and_flags_it():
    """Emeka Egbuka alt rec >=5: proj 4.1 against a 4.5 rung, cushion 0.92,
    placed on four of five cards. The rung sat above our own projection."""
    surv, rej = _gates([_leg(player="Emeka Egbuka", market="rush_yards",
                             main_line=30.0, alt_threshold=4.5, model_proj=4.1,
                             frac_of_line=0.15)])
    assert surv.empty
    assert rej.gate.iat[0] == "G3_cushion"
    assert "AGAINST OUR OWN PROJECTION" in rej.detail.iat[0]


def test_g3_rejects_insufficient_cushion_above_one():
    surv, rej = _gates([_leg(model_proj=12.0, alt_threshold=10.0)])
    assert surv.empty and rej.gate.iat[0] == "G3_cushion"
    assert "AGAINST OUR OWN" not in rej.detail.iat[0]


def test_g4_rejects_near_main_shade():
    """Roschon rec_yards >2.5 off a 3.0 main at a main-line price."""
    surv, rej = _gates([_leg(main_line=25.0, alt_threshold=24.0, model_proj=40.0,
                             frac_of_line=0.96)])
    assert surv.empty and rej.gate.iat[0] == "G4_near_main"


def test_g4_rejects_deep_over_rung():
    surv, rej = _gates([_leg(frac_of_line=2.2, model_proj=40.0)])
    assert surv.empty and rej.gate.iat[0] == "G4_frac_deep"


def test_g5_rejects_receptions_market():
    """receptions returned -30.0% ROI blind on 1260 week-9 rungs."""
    surv, rej = _gates([_leg(market="receptions", main_line=3.5, alt_threshold=1.5,
                             model_proj=3.8, frac_of_line=0.43)])
    assert surv.empty and rej.gate.iat[0] == "G5_market"


def test_g6_rejects_thin_own_evidence():
    surv, rej = _gates([_leg(own_games=3, own_hits=3)])
    assert surv.empty and rej.gate.iat[0] == "G6_own_games"


def test_g6_rejects_weak_own_rate():
    surv, rej = _gates([_leg(own_games=8, own_hits=3)])
    assert surv.empty and rej.gate.iat[0] == "G6_own_rate"


def test_g7_rejects_toughest_quartile_matchup():
    """Against the toughest defensive quartile the measured edge was -1.8%."""
    rows = [_leg(player=f"P{i}", def_allowed=float(d), event_id=f"e{i}")
            for i, d in enumerate([40, 80, 120, 160, 200])]
    surv, rej = _gates(rows)
    killed = rej[rej.gate == "G7_matchup"]
    assert len(killed) == 1 and killed.player.iat[0] == "P0"
    assert "P0" not in set(surv.player)


def test_g8_rejects_questionable_player():
    """Egbuka was Out in weeks 7 and 8 and Questionable in week 9 (foot)."""
    surv, rej = _gates([_leg(player="Emeka Egbuka", injury_status="Questionable")])
    assert surv.empty and rej.gate.iat[0] == "G8_injury"


@pytest.mark.parametrize("status", ["Out", "out", "Doubtful", "QUESTIONABLE"])
def test_g8_rejects_every_unavailable_status(status):
    surv, rej = _gates([_leg(injury_status=status)])
    assert surv.empty and rej.gate.iat[0] == "G8_injury"


def test_g8_rejects_quarantined_player():
    surv, rej = _gates([_leg(quarantined=True)])
    assert surv.empty and rej.gate.iat[0] == "G8_quarantine"


def test_g9_rejects_other_books():
    surv, rej = _gates([_leg(book="draftkings")], book="fanduel")
    assert surv.empty and rej.gate.iat[0] == "G9_book"


def test_g9_rejects_plus_money_above_cap():
    surv, rej = _gates([_leg(offered_american=400.0)])
    assert surv.empty and rej.gate.iat[0] == "G9_price"


def test_no_book_filter_keeps_all_books():
    surv, _ = _gates([_leg(book="betmgm", player="A", event_id="e1"),
                      _leg(book="fanduel", player="B", event_id="e2")], book=None)
    assert len(surv) == 2


def test_cards_never_share_a_leg():
    """Egbuka appeared on four of five cards; one miss removed 80% of the board."""
    rows = [_leg(player=f"P{i}", event_id=f"e{i}", model_proj=20.0 + i)
            for i in range(20)]
    surv, _ = _gates(rows)
    cards = build_cards(surv)
    assert not cards.empty
    counts = cards.player.value_counts()
    assert counts.max() == 1, "a leg was reused across tiers"


def test_cards_are_empty_rather_than_forced():
    """No qualifying legs must yield no cards, never a relaxed card."""
    surv, _ = _gates([_leg(snap_pct=0.10)])
    assert build_cards(surv).empty


def test_one_rung_per_player_per_card():
    rows = [_leg(player="Same Guy", alt_threshold=t, model_proj=40.0,
                 frac_of_line=0.3, event_id="e1") for t in (5.0, 10.0, 15.0)]
    surv, _ = _gates(rows)
    assert len(surv) == 1, "multiple rungs of one player/market survived"


# --- market exclusions and depth cap, from PRISM_ALT_TABLE_SUPPORT_AUDIT_V1 ---


def test_rush_rec_yards_is_excluded():
    """Worst-calibrated cell family in the repository.

    Clearance table overstates it by -0.0809, every frac band negative from
    0.7x upward, -33.9% ROI blind on 321 week-9 rungs. It produces zero legs
    under the other gates anyway; the explicit exclusion exists so a future
    loosening of the role floor or cushion cannot silently admit it.
    """
    from scripts.research.build_alt_cards_v1 import EXCLUDED_MARKETS
    assert "rush_rec_yards" in EXCLUDED_MARKETS
    surv, rej = _gates([_leg(market="rush_rec_yards", main_line=80.0,
                             alt_threshold=40.0, model_proj=80.0, frac_of_line=0.5)])
    assert surv.empty and rej.gate.iat[0] == "G5_market"


def test_rush_yards_is_NOT_excluded(monkeypatch):
    """Deliberate divergence from the table audit, and the reason is recorded.

    PRISM_ALT_TABLE_SUPPORT_AUDIT_V1 disqualifies rush_yards for TABLE-priced
    selection (-0.0367 calibration error, -18.5% ROI blind). This builder never
    reads the table, and gated, rush_yards returned +1.7% on week 9. Excluding
    it would discard a positive.
    """
    from scripts.research.build_alt_cards_v1 import EXCLUDED_MARKETS
    assert "rush_yards" not in EXCLUDED_MARKETS
    surv, _ = _gates([_leg(market="rush_yards", main_line=40.0, alt_threshold=20.0,
                           model_proj=40.0, frac_of_line=0.5)])
    assert len(surv) == 1


def test_pass_yards_is_not_excluded_by_market_list():
    """It produces zero legs via the role floor, not via a market ban."""
    from scripts.research.build_alt_cards_v1 import EXCLUDED_MARKETS
    assert "pass_yards" not in EXCLUDED_MARKETS


def test_depth_cap_sits_inside_the_measured_cliff():
    """Rungs beyond 1.6x returned -55.1% ROI on 593 week-9 rungs."""
    from scripts.research.build_alt_cards_v1 import FRAC_MAX
    assert FRAC_MAX <= 1.6
