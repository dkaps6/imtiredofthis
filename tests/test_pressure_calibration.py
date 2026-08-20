import pandas as pd
import pytest

from scripts.backtest.run_pressure_calibration import VARIANTS, passing_ranking


def _summary(market="pass_yards"):
    return pd.DataFrame(
        [
            {"variant": name, "market": market, "mae": 62.0 + i / 100.0}
            for i, name in enumerate(VARIANTS)
        ]
    )


def test_passing_ranking_uses_canonical_pass_yards_market():
    ranked = passing_ranking(_summary())
    assert len(ranked) == len(VARIANTS)
    assert set(ranked["variant"]) == set(VARIANTS)
    assert ranked["market"].eq("pass_yards").all()


def test_passing_ranking_fails_loudly_if_market_name_drifts():
    with pytest.raises(RuntimeError, match="no pass_yards ranking"):
        passing_ranking(_summary("passing_yards"))


def test_passing_ranking_fails_if_a_variant_is_missing():
    summary = _summary().iloc[:-1].copy()
    with pytest.raises(RuntimeError, match="missing variants"):
        passing_ranking(summary)
