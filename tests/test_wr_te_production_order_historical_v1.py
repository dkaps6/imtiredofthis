import numpy as np
import pandas as pd
import pytest

from scripts.research.persist_wr_te_production_order_historical_v1 import (
    TE_FEATURES,
    WR_FEATURES,
    _authorized,
    apply_te_fold,
    apply_wr_fold,
)


def _params(features, season):
    return {
        "mean": np.zeros(len(features), dtype=float),
        "scale": np.ones(len(features), dtype=float),
        "coef": np.zeros(len(features), dtype=float),
        "intercept": 0.0,
        "test_season": season,
        "features": list(features),
    }


def _metrics():
    rows = [
        ("te_one", "TE", 0.10), ("te_two", "TE", 0.06),
        ("wr_one", "WR", 0.22), ("wr_two", "WR", 0.12), ("wr_three", "WR", 0.07),
        ("rb_one", "RB", 0.08),
    ]
    return pd.DataFrame([
        {
            "event_id": "2024_01_A_B", "season": 2024, "week": 1,
            "team": "A", "opponent": "B", "player": p,
            "player_clean_key": p, "position": pos,
            "entitlement_tgt_share": share, "entitlement_residual_share": 0.35,
            "market": "football_universe",
        }
        for p, pos, share in rows
    ])


def _snaps():
    players = ["te_one", "te_two", "wr_one", "wr_two", "wr_three", "rb_one"]
    rows = []
    for i, p in enumerate(players):
        for week in [15, 16, 17]:
            rows.append({
                "season": 2023, "week": week, "team": "A", "player_key": p,
                "offense_pct": 0.4 + 0.05 * i, "offense_snaps": 30 + i,
                "ordinal": 202300 + week,
            })
    return pd.DataFrame(rows)


def test_te_fold_conserves_te_room_and_non_te():
    before = _metrics()
    out, _, audit = apply_te_fold(before, snaps=_snaps(), params=_params(TE_FEATURES, 2024))
    te = before.position.eq("TE")
    assert out.loc[te, "entitlement_tgt_share"].sum() == pytest.approx(before.loc[te, "entitlement_tgt_share"].sum(), abs=1e-12)
    assert np.allclose(out.loc[~te, "entitlement_tgt_share"], before.loc[~te, "entitlement_tgt_share"], atol=0, rtol=0)
    assert audit["team_te_pool_max_abs_gap"] <= 1e-12


def test_wr_fold_preserves_anchor_secondary_pool_and_non_wr():
    before = _metrics()
    out, _, audit = apply_wr_fold(before, snaps=_snaps(), params=_params(WR_FEATURES, 2024))
    wr = before.position.eq("WR")
    anchor = before.loc[wr, "entitlement_tgt_share"].idxmax()
    secondaries = [i for i in before.index[wr] if i != anchor]
    assert out.loc[anchor, "entitlement_tgt_share"] == pytest.approx(before.loc[anchor, "entitlement_tgt_share"], abs=1e-12)
    assert out.loc[secondaries, "entitlement_tgt_share"].sum() == pytest.approx(before.loc[secondaries, "entitlement_tgt_share"].sum(), abs=1e-12)
    assert np.allclose(out.loc[~wr, "entitlement_tgt_share"], before.loc[~wr, "entitlement_tgt_share"], atol=0, rtol=0)
    assert audit["m38_wr1_anchor_max_abs_gap"] <= 1e-12
    assert audit["wr2plus_pool_max_abs_gap"] <= 1e-12


def test_authorized_scope_excludes_2025_wr():
    wr24 = pd.Series({"market": "rec_yards", "position": "WR"})
    te24 = pd.Series({"market": "receptions", "position": "TE"})
    wr25 = pd.Series({"market": "rush_rec_yards", "position": "WR"})
    te25 = pd.Series({"market": "rush_rec_yards", "position": "TE"})
    assert _authorized(wr24, 2024) == (True, "WR_R15_OOS_PRODUCTION_ORDER")
    assert _authorized(te24, 2024) == (True, "TE_R5P_OOS_PRODUCTION_ORDER")
    assert _authorized(wr25, 2025) == (False, "NO_WR_R15_OOS_AUTHORITY")
    assert _authorized(te25, 2025) == (True, "TE_R5P_OOS_PRODUCTION_ORDER")
