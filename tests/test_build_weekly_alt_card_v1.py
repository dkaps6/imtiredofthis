import numpy as np
from types import SimpleNamespace as NS
from scripts.research.build_weekly_alt_card_v1 import check


def leg(**kw):
    base = dict(common_fails=[], frac_of_line=0.50, own_hits=8, own_games=8, ctx_hits=8, ctx_games=8, breakeven=0.80,
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


def test_context_history_is_held_to_the_tier_rule():
    # Braelon Allen o39.5: 6/9 on the season, 1/4 once Breece-Hall-out games are dropped
    braelon = leg(own_hits=6, own_games=9, ctx_hits=1, ctx_games=4, breakeven=0.40,
                  frac_of_line=1.2, offered_american=148.0)
    assert "context own" in check(braelon, "MEDIUM")
    assert "context below breakeven" in check(braelon, "HIGH")


def test_teammate_direction_competitor_vs_complement():
    import pandas as pd
    from scripts.research.audit_card_legs_v1 import teammate_context_values
    rows = []
    for w in range(1, 9):
        rows.append(dict(player_display_name="RB2", team="NYJ", position="RB", week=w, carries=15,
                         rushing_yards=80 if w >= 5 else 20, targets=0, receiving_yards=0, passing_yards=0))
        if w < 5 or w == 8:   # RB1 missed weeks 5-7, back in week 8 and today
            rows.append(dict(player_display_name="RB1", team="NYJ", position="RB", week=w, carries=18,
                             rushing_yards=70, targets=0, receiving_yards=0, passing_yards=0))
        rows.append(dict(player_display_name="QB", team="LA", position="QB", week=w, carries=0,
                         rushing_yards=0, targets=0, receiving_yards=0, passing_yards=250))
        rows.append(dict(player_display_name="WR1", team="LA", position="WR", week=w, carries=0,
                         rushing_yards=0, targets=10, receiving_yards=90, passing_yards=0))
        rows.append(dict(player_display_name="WR2", team="LA", position="WR", week=w, carries=0,
                         rushing_yards=0, targets=6, receiving_yards=50, passing_yards=0))
    st = pd.DataFrame(rows)
    played = {p: set(st[st.player_display_name == p].week) for p in st.player_display_name.unique()}
    team_now = {"RB1": "NYJ", "RB2": "NYJ", "QB": "LA", "WR1": "LA", "WR2": "LA"}
    # competitor back today: his missed weeks (5-7) are dropped for RB2
    vals, notes = teammate_context_values(st, played, team_now, set(), "RB2", "NYJ", "rush_yards")
    assert len(vals) == 5 and "RB1 back" in notes
    # competitor OUT today: nothing dropped (more work for RB2)
    vals, _ = teammate_context_values(st, played, team_now, {"RB1"}, "RB2", "NYJ", "rush_yards")
    assert len(vals) == 8
    # QB whose top receiver (10 of 16 targets) is out today: every game he played is dropped
    vals, notes = teammate_context_values(st, played, team_now, {"WR1"}, "QB", "LAR", "pass_yards")
    assert len(vals) == 0 and "WR1 out today" in notes
    # ... but losing only the smaller share (6 of 16 = 37.5%) still counts; add a third WR so
    # the missing receiver is a minority of the targets and nothing is dropped
    extra = pd.DataFrame([dict(player_display_name="WR3", team="LA", position="WR", week=w, carries=0,
                               rushing_yards=0, targets=20, receiving_yards=150, passing_yards=0) for w in range(1, 9)])
    st2 = pd.concat([st, extra]); played["WR3"] = set(range(1, 9)); team_now["WR3"] = "LA"
    vals, _ = teammate_context_values(st2, played, team_now, {"WR2"}, "QB", "LAR", "pass_yards")
    assert len(vals) == 8
