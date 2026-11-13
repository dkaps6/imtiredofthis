import pandas as pd
from scripts.research.validate_ticket_card_v1 import validate, amer, dec

def leg(ticket, tier, player, price, book="fanduel", market="rec_yards", side="OVER",
        reason="named role and usage", state="HEALTHY_FULL", basis="quote", **x):
    return dict(ticket=ticket, tier=tier, player=player, market=market, side=side, threshold=10.5,
                price_american=price, book=book, football_reason=reason, state_label=state,
                price_basis=basis, **x)

def errs(rows, **kw): return [m for s, _, m in validate(pd.DataFrame(rows), **kw) if s == "ERROR"]
def warns(rows): return [m for s, _, m in validate(pd.DataFrame(rows)) if s == "WARNING"]

def test_amer_roundtrip():
    assert amer(dec(150)) == 150 and amer(dec(-110)) == -110

def test_safe_must_be_strictly_over_150():
    at = [leg("A", "SAFE", "P1", 100), leg("A", "SAFE", "P2", 150)]      # 2.0*2.5 = +400
    assert not errs(at)
    exact = [leg("A", "SAFE", "P1", 150)]                                  # exactly +150
    assert any("not over +150" in e for e in errs(exact))
    just = [leg("A", "SAFE", "P1", 151)]
    assert not errs(just)

def test_safe_below_floor_fails():
    # -260 and -200 combine to about +108
    assert any("SAFE" in e for e in errs([leg("A", "SAFE", "P1", -260), leg("A", "SAFE", "P2", -200)]))

def test_last_nights_safe_clears_the_revised_floor():
    # Fields -260 with White -113 combined to +161, which is over +150
    assert not errs([leg("A", "SAFE", "FIELDS", -260), leg("A", "SAFE", "WHITE", -113)])

def test_floor_is_a_setting():
    rows = [leg("A", "SAFE", "P1", 200)]
    assert not errs(rows, safe_floor=150)
    assert errs(rows, safe_floor=250)

def test_player_on_three_regular_tickets_fails():
    rows = [leg(t, tier, "FIELDS", 200) for t, tier in (("A", "SAFE"), ("B", "MODERATE"), ("C", "MEDIUM"))]
    assert any("FIELDS appears on 3" in e for e in errs(rows))

def test_two_regular_tickets_ok_and_moonshot_is_separate():
    rows = [leg("A", "SAFE", "F", 200), leg("B", "MODERATE", "F", 200), leg("M", "MOONSHOT", "F", 200)]
    assert not [e for e in errs(rows) if "appears on" in e]

def test_mixed_books_fail():
    rows = [leg("A", "MODERATE", "P1", 200, book="fanduel"), leg("A", "MODERATE", "P2", 200, book="draftkings")]
    assert any("ONE book" in e for e in errs(rows))

def test_receptions_only_ticket_fails_but_mixed_ok():
    only = [leg("A", "MODERATE", "P1", 200, market="receptions"), leg("A", "MODERATE", "P2", 200, market="receptions")]
    mixed = [leg("A", "MODERATE", "P1", 200, market="receptions"), leg("A", "MODERATE", "P2", 200)]
    assert any("only of reception" in e for e in errs(only))
    assert not [e for e in errs(mixed) if "reception" in e]

def test_missing_football_reason_fails():
    assert any("no football reason" in e for e in errs([leg("A", "MODERATE", "P1", 200, reason="  ")]))

def test_lottery_under_250x_fails_unless_unsolved():
    rows = [leg("L", "LOTTERY", "P1", 700), leg("L", "LOTTERY", "P2", 310)]   # about +3,300
    assert any("250x" in e for e in errs(rows))
    flagged = [dict(r, status="UNSOLVED") for r in rows]
    assert not [e for e in errs(flagged) if "250x" in e]

def test_naive_product_warns_and_bad_state_warns():
    w = warns([leg("A", "MODERATE", "P1", 200, basis="naive_product", state="RETURN_FROM_INJURY")])
    assert any("not a verified book quote" in m for m in w) and any("anomaly gate" in m for m in w)

def test_rb_rush_under_warns():
    assert any("RB under family" in m for m in warns([leg("A", "MODERATE", "P1", 200, market="rush_yards", side="UNDER")]))

def test_tonights_card_would_have_been_rejected():
    rows = [leg(t, tier, "FIELDS", p, basis="naive_product") for t, tier, p in
            (("SAFE", "SAFE", -260), ("MOD", "MODERATE", -260), ("MED", "MEDIUM", 134), ("HIGH", "HIGH", 320), ("LOT", "LOTTERY", 700))]
    e = errs(rows)
    assert any("appears on 5" in m for m in e) and any("250x" in m for m in e)


def test_a_leg_without_an_edge_is_refused_on_every_tier():
    # -300 has a breakeven of 0.75; a stated 0.70 is a bet against the price
    for tier in ("SAFE", "MODERATE", "HIGH", "LOTTERY"):
        rows = [leg("A", tier, "P1", -300, model_prob=0.70)]
        assert any("does not exceed breakeven" in e for e in errs(rows)), tier

def test_a_leg_with_an_edge_passes_the_edge_check():
    rows = [leg("A", "HIGH", "P1", 200, model_prob=0.40)]      # breakeven 0.333
    assert not [e for e in errs(rows) if "breakeven" in e]

def test_missing_model_prob_warns_only_on_high_and_lottery():
    base = dict(model_prob=None)
    hi = [leg("A", "HIGH", "P1", 200, **base)]
    mod = [leg("B", "MODERATE", "P1", 200, **base)]
    assert any("no model probability" in m for m in warns(hi))
    assert not any("no model probability" in m for m in warns(mod))
