"""The alt pricer must read the main-line artifact production actually writes.

Run 42094118645 paid for 738 alternate offers and then died on
KeyError: 'line'. outputs/props_raw.csv is the GROUPED artifact -- one row per
(event, market, player) with the per-book lines inside an offers_json blob and
no 'line' column at all. The pricer had only ever been exercised against the
older FLAT archive schema (data/market_track_record/raw_props/2026_wk08.csv).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.price_alt_line_offers_v1 import build, flatten_main

EVENT = "c75744790b7acf6260c78ba5e739c0e5"


def _grouped_main() -> pd.DataFrame:
    """The real production shape, as archived for 2026 week 9."""

    def offers(line: float) -> str:
        return json.dumps(
            [
                {"book": "draftkings", "book_title": "DraftKings", "side": "OVER",
                 "line": line, "price": -114},
                {"book": "draftkings", "book_title": "DraftKings", "side": "UNDER",
                 "line": line, "price": -110},
                {"book": "fanduel", "book_title": "FanDuel", "side": "OVER",
                 "line": line + 1.0, "price": -120},
            ]
        )

    return pd.DataFrame(
        [
            {"event_id": EVENT, "market": "player_reception_yds",
             "book_player_name": "Zay Flowers", "canonical_player_name": "Zay Flowers",
             "bookmaker_missing": 0, "offers_json": offers(54.5)},
            {"event_id": EVENT, "market": "player_reception_yds",
             "book_player_name": "Matt Hibner", "canonical_player_name": "Matt Hibner",
             "bookmaker_missing": 0, "offers_json": offers(14.5)},
        ]
    )


def _flat_main() -> pd.DataFrame:
    """The older archive shape, which must keep working."""
    return pd.DataFrame(
        [
            {"event_id": EVENT, "market": "player_reception_yds",
             "player": "Zay Flowers", "line": 54.5},
            {"event_id": EVENT, "market": "player_reception_yds",
             "player": "Zay Flowers", "line": 55.5},
        ]
    )


def _alt(player: str = "Zay Flowers") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_id": EVENT, "book": "draftkings",
             "market": "player_reception_yds_alternate", "player": player,
             "side": side, "line": 27.5, "price_american": price}
            for side, price in (("OVER", -175), ("UNDER", 140))
        ]
    )


def test_flatten_main_reads_offers_json():
    flat = flatten_main(_grouped_main())
    assert not flat.empty
    assert set(flat.columns) == {"event_id", "player", "market", "line"}
    # OVER rows only -- a one-sided book must not skew the median.
    assert sorted(flat.loc[flat["player"].eq("Zay Flowers"), "line"]) == [54.5, 55.5]


def test_flatten_main_passes_through_the_flat_schema():
    flat = flatten_main(_flat_main())
    assert len(flat) == 2
    assert flat["line"].tolist() == [54.5, 55.5]


def test_flatten_main_rejects_an_artifact_with_neither_shape():
    bad = pd.DataFrame([{"event_id": EVENT, "market": "player_receptions", "player": "X"}])
    try:
        flatten_main(bad)
    except RuntimeError as exc:
        assert "offers_json" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unusable main artifact must fail loudly")


def test_build_anchors_against_the_grouped_schema():
    out = build(_alt(), _grouped_main())
    assert len(out) == 2
    assert out["main_line"].notna().all()
    assert np.isclose(out["main_line"].iloc[0], 55.0)  # median of 54.5 / 55.5
    assert (out["unpriceable_reason"] == "").all()


def test_build_bridges_a_spelling_difference_between_the_two_calls():
    """The ladder and the main line are separate OddsAPI calls."""
    out = build(_alt("Matthew Hibner"), _grouped_main())
    assert out["main_line"].notna().all(), "the Matt/Matthew ladder lost its anchor"


def test_under_gets_a_real_conservative_haircut():
    """cleared_conservative is a one-sided DOWNWARD haircut on clearance.

    Taking 1 - cleared as the UNDER's conservative probability applies no
    haircut at all, so every UNDER outranks every OVER on ev_conservative --
    the column the board is sorted by.
    """
    out = build(_alt(), _grouped_main())
    under = out.loc[out["side"].eq("UNDER")].iloc[0]
    assert np.isfinite(under["p_side_conservative"])
    assert under["p_side_conservative"] < under["p_side"] - 1e-9
    assert under["ev_conservative"] < under["ev_per_dollar"] - 1e-9
    over = out.loc[out["side"].eq("OVER")].iloc[0]
    assert over["p_side_conservative"] <= over["p_side"] + 1e-9
