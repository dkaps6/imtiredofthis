import numpy as np
import pandas as pd

from scripts.research import audit_specialist_mc_downstream_materiality_v1 as m


def _board(prob_over: float) -> pd.DataFrame:
    rows = []
    for book, line, over_odds, under_odds in [
        ("a", 100.5, -110, -110),
        ("b", 101.5, 105, -125),
    ]:
        for side, p, odds in [
            ("OVER", prob_over, over_odds),
            ("UNDER", 1.0 - prob_over, under_odds),
        ]:
            rows.append({
                "paid_row_id": len(rows),
                "season": 2026,
                "week": 3,
                "event_id": "g1",
                "player": "Player One",
                "player_clean_key": "playerone",
                "team": "AAA",
                "opponent": "BBB",
                "market": "pass_yards",
                "source_market": "player_pass_yds",
                "book": book,
                "book_title": book,
                "vegas_line": line,
                "vegas_odds": odds,
                "side": side,
                "fair_prob": p,
                "stage_ev_roi": m._ev_roi(p, odds),
            })
    return pd.DataFrame(rows)


def test_rank_corr_identical_is_one():
    a = pd.Series([1.0, 3.0, 2.0, 4.0])
    assert np.isclose(m._rank_corr(a, a), 1.0)


def test_compare_boards_detects_pricing_instability():
    left = _board(0.49)
    right = _board(0.58)
    summary, detail = m._compare_boards(
        left,
        right,
        {("g1", "playerone")},
        comparison="X",
        surface="SHAPE_ONLY_FIXED_FINAL_MEAN",
        kind="SPECIALIST",
    )
    assert summary["protected_side_rows"] == 4
    assert summary["max_abs_prob_delta"] > 0
    assert (
        summary["quote_preferred_side_flips"] > 0
        or summary["best_snapshot_bet_pass_flips"] > 0
        or summary["best_snapshot_identity_changes"] > 0
    )
    assert not detail.empty


def test_protected_sets_are_stage_specific():
    s = pd.DataFrame([
        {"event_id": "g1", "player_clean_key": "a", "te_protected": True, "wr_protected": False},
        {"event_id": "g1", "player_clean_key": "b", "te_protected": False, "wr_protected": True},
    ])
    out = m._protected_sets(s)
    assert out["TE_R5P_PROTECTED"] == {("g1", "a")}
    assert out["WR_R15_PROTECTED"] == {("g1", "b")}
