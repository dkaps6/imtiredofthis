"""Tests for the alternate-line edge pricer.

No real alternate-line capture exists yet -- the code path only runs with live
odds -- so these exercise it end to end on synthetic offers, including the
cases that must refuse to produce a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

mod = pytest.importorskip("scripts.price_alt_line_offers_v1")


def _alt(rows):
    return pd.DataFrame(rows)


def _main(rows):
    return pd.DataFrame(rows)


def test_profit_and_ev_math():
    assert mod._profit_per_dollar(-400) == pytest.approx(0.25)
    assert mod._profit_per_dollar(+150) == pytest.approx(1.5)
    # a 60% shot at +150 is clearly positive: .6*1.5 - .4 = +0.5
    assert mod._ev(0.6, 150) == pytest.approx(0.5)
    # a 50% shot at -110 is the classic losing proposition
    assert mod._ev(0.5, -110) == pytest.approx(0.5 * (100 / 110) - 0.5)


def test_a_high_volume_over_at_half_the_line_prices_from_the_table():
    alt = _alt([{"event_id": "e1", "player": "Bijan Robinson", "book": "fd",
                 "market": "player_rush_reception_yds_alternate", "side": "OVER",
                 "line": 42.25, "price_american": -400}])
    main = _main([{"event_id": "e1", "player": "Bijan Robinson", "book": "fd",
                   "market": "player_rush_reception_yds", "line": 84.5}])
    out = mod.build(alt, main)
    r = out.iloc[0]
    assert r["unpriceable_reason"] == ""
    assert r["frac_of_line"] == pytest.approx(0.5)
    # Q3 is line > 71.5; the table's rush_rec_yards Q3 row at 0.50x
    assert r["clearance"] > 0.85
    assert r["ev_per_dollar"] == pytest.approx(
        r["clearance"] * 0.25 - (1 - r["clearance"]), abs=1e-9)
    assert r["ev_per_dollar"] > 0


def test_under_uses_the_complement_of_clearance():
    base = {"event_id": "e1", "player": "X", "book": "fd",
            "market": "player_rec_yards_alternate", "line": 30.0,
            "price_american": -110}
    # use a real alt key
    base["market"] = "player_reception_yds_alternate"
    alt = _alt([dict(base, side="OVER"), dict(base, side="UNDER")])
    main = _main([{"event_id": "e1", "player": "X", "book": "fd",
                   "market": "player_reception_yds", "line": 50.0}])
    out = mod.build(alt, main).set_index("side")
    assert out.loc["OVER", "p_side"] == pytest.approx(out.loc["OVER", "clearance"])
    assert out.loc["UNDER", "p_side"] == pytest.approx(1 - out.loc["UNDER", "clearance"])


def test_a_threshold_outside_the_table_support_refuses_to_price():
    """An alt ladder quotes far beyond what two seasons can speak to."""
    alt = _alt([{"event_id": "e1", "player": "X", "book": "fd",
                 "market": "player_reception_yds_alternate", "side": "OVER",
                 "line": 5.0, "price_american": -2000}])  # 0.1x the line
    main = _main([{"event_id": "e1", "player": "X", "book": "fd",
                   "market": "player_reception_yds", "line": 50.0}])
    out = mod.build(alt, main)
    assert out.iloc[0]["unpriceable_reason"] == "fraction_outside_table_support"
    assert not np.isfinite(out.iloc[0]["ev_per_dollar"])


def test_receptions_is_excluded_until_reparameterised():
    alt = _alt([{"event_id": "e1", "player": "X", "book": "fd",
                 "market": "player_receptions_alternate", "side": "OVER",
                 "line": 3.5, "price_american": -150}])
    main = _main([{"event_id": "e1", "player": "X", "book": "fd",
                   "market": "player_receptions", "line": 5.5}])
    out = mod.build(alt, main)
    assert out.iloc[0]["unpriceable_reason"] == "market_excluded_integer_parameterisation"


def test_a_missing_main_line_anchor_refuses_to_price():
    """The table is indexed on the sportsbook line; without it there is no anchor."""
    alt = _alt([{"event_id": "e1", "player": "Ghost", "book": "fd",
                 "market": "player_reception_yds_alternate", "side": "OVER",
                 "line": 30.0, "price_american": -200}])
    out = mod.build(alt, _main([{"event_id": "e1", "player": "Someone Else",
                                 "book": "fd", "market": "player_reception_yds",
                                 "line": 50.0}]))
    assert out.iloc[0]["unpriceable_reason"] == "no_main_line_anchor"


def test_the_model_projection_is_never_an_input():
    """The whole design point: p_over is miscalibrated by up to +0.448."""
    src = (mod.__file__ and open(mod.__file__, encoding="utf-8").read()) or ""
    for forbidden in ("mc_proj", "ml_proj", "state_proj", "ensemble_proj",
                      "model_proj", "_prob_over", "component_sd"):
        assert f'"{forbidden}"' not in src and f"'{forbidden}'" not in src
