import pandas as pd

from scripts.research.audit_distribution_right_tail_asymmetry_v1 import _cell, classify


def _cell_fixture(season: int, market: str, upper: bool) -> pd.DataFrame:
    rows = []
    for i in range(250):
        rows.append({
            "season": season,
            "market": market,
            "game_id": f"g{i%50}",
            "actual": 1.0,
            "upper_break_10": int(upper and i % 5 == 0),
            "lower_break_10": int((not upper) and i % 5 == 0),
            "upper_break_05": int(upper and i % 10 == 0),
            "lower_break_05": int((not upper) and i % 10 == 0),
        })
    return pd.DataFrame(rows)


def test_cell_detects_clear_upper_tail_asymmetry():
    q = _cell_fixture(2024, "rec_yards", True)
    r = _cell(q, 2024, "rec_yards")
    assert r["support"] == "PASS"
    assert r["upper_minus_lower_q10q90"] > 0
    assert r["bootstrap_ci_low"] > 0
    assert r["cell_pass"] is True


def test_classify_requires_both_seasons():
    rows = []
    for season in (2024, 2025):
        for market in ("pass_yards", "rush_yards", "rec_yards", "receptions", "rush_rec_yards"):
            rows.append({
                "season": season,
                "market": market,
                "cell_pass": market == "rec_yards",
            })
    out = classify(pd.DataFrame(rows))
    assert out["disposition"] == "DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_SIGNAL_CONFIRMED"
    rec = [x for x in out["per_market"] if x["market"] == "rec_yards"][0]
    assert rec["disposition"] == "RIGHT_TAIL_ASYMMETRY_REPLICATED"
