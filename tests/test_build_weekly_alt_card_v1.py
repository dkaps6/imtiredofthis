import numpy as np
from types import SimpleNamespace as NS
from scripts.research.build_weekly_alt_card_v1 import check


def leg(**kw):
    base = dict(common_fails=[], frac_of_line=0.50, own_hits=8, own_games=8, breakeven=0.80,
                model_proj=40.0, alt_threshold=19.5, offered_american=-400.0)
    base.update(kw)
    return NS(**base)


def test_clean_floor_leg_is_eligible_everywhere():
    for t in ("SAFE", "MODERATE", "MEDIUM", "HIGH", "LOTTERY"):
        assert check(leg(), t) == []


def test_common_fails_block_every_tier():
    # teammate context, team change, matchup ... apply to all five tickets
    for t in ("SAFE", "MODERATE", "MEDIUM", "HIGH", "LOTTERY"):
        assert "teammate context" in check(leg(common_fails=["teammate context"]), t)


def test_main_line_in_disguise_fails_g4_on_every_tier():
    # Bryce Young o249.5 against a 255 main line (frac 0.98)
    for t in ("SAFE", "MODERATE", "HIGH"):
        assert "G4 alt" in check(leg(frac_of_line=0.98), t)


def test_safe_needs_cushion_and_price_window_but_high_does_not():
    plus = leg(model_proj=54.0, alt_threshold=49.5, offered_american=150.0, breakeven=0.40,
               own_hits=5, own_games=8, frac_of_line=1.05 + 0.2)
    assert {"G3 cushion", "G9 price"} <= set(check(plus, "SAFE"))
    assert check(plus, "HIGH") == []


def test_high_needs_own_history_above_breakeven_mid_needs_sixty_percent():
    weak = leg(own_hits=2, own_games=8, breakeven=0.30, frac_of_line=1.3, offered_american=230.0)
    assert "own below breakeven" in check(weak, "HIGH")
    mid = leg(own_hits=4, own_games=8)
    assert "G6 own" in check(mid, "MODERATE") and "G6 own" not in check(mid, "HIGH")
