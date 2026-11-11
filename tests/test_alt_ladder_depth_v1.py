"""Contract tests for scripts/research/analyze_alt_ladder_depth_v1.py.

The whole analysis rests on one premise: an alt ladder is monotone, so the
deepest cleared rung is well defined. These tests pin that premise, the
high-water/died bracket, and the survival arithmetic.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ladder_depth", Path(__file__).resolve().parents[1]
    / "scripts" / "research" / "analyze_alt_ladder_depth_v1.py")
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _ladder(player="P", market="rec_yards", book="fanduel", actual=40.0,
            thresholds=(9.5, 19.5, 29.5, 49.5),
            prices=(-400.0, -150.0, 120.0, 400.0), main=35.0, event="E1"):
    return pd.DataFrame([{
        "event_id": event, "player": player, "market": market, "book": book,
        "side": "OVER", "alt_threshold": t, "main_line": main,
        "offered_american": p, "actual": actual,
        "hit": int(actual > t), "pnl": 0.0, "p_side": 0.5,
    } for t, p in zip(thresholds, prices)])


# ---------------------------------------------------------------- dec()

@pytest.mark.parametrize("american,expected", [
    (100.0, 1.0), (-100.0, 1.0), (200.0, 2.0), (-200.0, 0.5), (-400.0, 0.25),
])
def test_dec_converts_american_to_decimal_profit(american, expected):
    assert mod.dec(american) == pytest.approx(expected)


# -------------------------------------------------- monotonicity premise

def test_monotonicity_holds_for_a_ladder_graded_from_one_actual():
    assert mod.check_monotone(_ladder(actual=40.0)) == 0


def test_monotonicity_violation_is_detected():
    """A hand-corrupted ladder: the hard rung hits while an easy one misses."""
    d = _ladder(actual=40.0)
    d.loc[d.alt_threshold == 19.5, "hit"] = 0  # easy rung forced to miss
    d.loc[d.alt_threshold == 49.5, "hit"] = 1  # hard rung forced to hit
    assert mod.check_monotone(d) == 1


def test_monotone_check_counts_ladders_not_rungs():
    bad = _ladder(player="A")
    bad.loc[bad.alt_threshold == 19.5, "hit"] = 0
    bad.loc[bad.alt_threshold == 49.5, "hit"] = 1
    assert mod.check_monotone(pd.concat([bad, _ladder(player="B")])) == 1


# ------------------------------------------------ high-water / died bracket

def test_high_water_mark_is_the_deepest_cleared_rung():
    L = mod.ladders(_ladder(actual=40.0))
    assert len(L) == 1
    row = L.iloc[0]
    assert row.hwm_thr == 29.5          # 29.5 < 40, 49.5 > 40
    assert row.hwm_price == 120.0
    assert row.n_hit == 3


def test_died_price_is_the_cheapest_missed_rung():
    row = mod.ladders(_ladder(actual=40.0)).iloc[0]
    assert row.died_thr == 49.5
    assert row.died_price == 400.0


def test_hwm_and_died_bracket_the_actual_result():
    row = mod.ladders(_ladder(actual=40.0)).iloc[0]
    assert row.hwm_thr < row.actual < row.died_thr


def test_a_ladder_that_clears_nothing_has_no_high_water_mark():
    row = mod.ladders(_ladder(actual=1.0)).iloc[0]
    assert np.isnan(row.hwm_price) and np.isnan(row.hwm_thr)
    assert row.cleared_nothing
    assert not row.cleared_200_or_better
    assert row.died_price == -400.0   # even the cheapest rung failed


def test_a_ladder_that_clears_everything_has_no_died_price():
    row = mod.ladders(_ladder(actual=500.0)).iloc[0]
    assert np.isnan(row.died_price)
    assert row.hwm_price == 400.0
    assert not row.cleared_nothing


def test_cleared_200_or_better_flag_tracks_the_hwm_price():
    """"Better than -200" means a HIGHER american price: -150 counts, -400 does not."""
    # tops out at -400, i.e. only ever cleared heavier juice than -200
    deep = mod.ladders(_ladder(actual=15.0, thresholds=(9.5, 19.5, 29.5),
                               prices=(-400.0, -150.0, 120.0))).iloc[0]
    assert deep.hwm_price == -400.0
    assert not deep.cleared_200_or_better
    # -150 IS better than -200, so clearing it must count
    r = mod.ladders(_ladder(actual=25.0, thresholds=(9.5, 19.5),
                            prices=(-400.0, -150.0))).iloc[0]
    assert r.hwm_price == -150.0 and r.cleared_200_or_better


def test_ladders_are_split_by_book():
    g = pd.concat([_ladder(book="fanduel"), _ladder(book="draftkings")])
    assert len(mod.ladders(g)) == 2


# ------------------------------------------------------------- survival

def test_survival_offered_counts_only_ladders_holding_a_qualifying_rung():
    # one ladder topping out at -150 offers nothing at +100 or better
    g = pd.concat([
        _ladder(player="A", actual=40.0),                       # has +120, +400
        _ladder(player="B", actual=25.0, thresholds=(9.5, 19.5),
                prices=(-400.0, -150.0)),                        # nothing >= +100
    ])
    s = mod.survival(g, [-400, 100]).set_index("level")
    assert s.loc[-400, "offered"] == 2
    assert s.loc[100, "offered"] == 1


def test_survival_cleared_requires_a_hitting_rung_at_or_above_the_level():
    g = _ladder(actual=40.0)   # clears 29.5 @ +120, misses 49.5 @ +400
    s = mod.survival(g, [100, 400]).set_index("level")
    assert s.loc[100, "cleared"] == 1
    assert s.loc[400, "cleared"] == 0


def test_survival_picks_the_cheapest_qualifying_rung():
    """At level -400 the pick must be the -400 rung, not the +400 one."""
    s = mod.survival(_ladder(actual=40.0), [-400]).iloc[0]
    assert s.pick_mean_price == -400.0
    assert s.pick_roi_pct == pytest.approx(25.0)   # hit at -400 -> +0.25u


def test_survival_roi_is_minus_one_hundred_when_the_pick_misses():
    s = mod.survival(_ladder(actual=40.0), [400]).iloc[0]
    assert s.pick_hit_pct == 0.0
    assert s.pick_roi_pct == pytest.approx(-100.0)


def test_survival_edge_is_hit_rate_minus_breakeven():
    s = mod.survival(_ladder(actual=40.0), [100]).iloc[0]
    # pick is +120 -> breakeven 100/220 = 45.45%, hit 100%
    assert s.pick_breakeven_pct == pytest.approx(45.5, abs=0.1)
    assert s.pick_edge_pts == pytest.approx(100.0 - 45.5, abs=0.1)


def test_survival_skips_levels_no_ladder_offers():
    assert 9999 not in mod.survival(_ladder(), [-400, 9999]).level.to_numpy()


# ------------------------------------------------------------ load_board

def test_load_board_rejects_under_rows(tmp_path):
    d = _ladder(); d.loc[0, "side"] = "UNDER"
    p = tmp_path / "b.csv"; d.to_csv(p, index=False)
    with pytest.raises(SystemExit, match="OVER"):
        mod.load_board(str(p))


def test_load_board_rejects_an_ungraded_board(tmp_path):
    d = _ladder().drop(columns=["pnl"])
    p = tmp_path / "b.csv"; d.to_csv(p, index=False)
    with pytest.raises(SystemExit, match="not a graded board"):
        mod.load_board(str(p))


def test_load_board_filters_to_one_book(tmp_path):
    g = pd.concat([_ladder(book="fanduel"), _ladder(book="draftkings")])
    p = tmp_path / "b.csv"; g.to_csv(p, index=False)
    assert set(mod.load_board(str(p), "fanduel").book) == {"fanduel"}
    with pytest.raises(SystemExit, match="no rows for book"):
        mod.load_board(str(p), "nosuchbook")


def test_load_board_drops_ungraded_rows(tmp_path):
    d = _ladder(); d.loc[3, "pnl"] = np.nan
    p = tmp_path / "b.csv"; d.to_csv(p, index=False)
    assert len(mod.load_board(str(p))) == 3


# ------------------------------------------- the committed week-9 board

@pytest.mark.parametrize("rel", [
    "data/market_track_record/graded/w9_2026_alt_board_graded_complete_v1.csv"])
def test_committed_week9_board_is_monotone(rel):
    """The published ladder findings are void if this ever regresses."""
    path = Path(__file__).resolve().parents[1] / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    g = mod.load_board(str(path))
    assert mod.check_monotone(g) == 0
    L = mod.ladders(g)
    assert len(L) == 966
    assert int(L.cleared_200_or_better.sum()) == 503
