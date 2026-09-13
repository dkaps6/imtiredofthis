import numpy as np
import pandas as pd
import pytest

from scripts.research.diagnose_strong_gate_component_sd_v1 import diagnose


def _fixture(n=400, seed=7):
    rng = np.random.default_rng(seed)
    # component_sd deliberately ~0.3x the true residual spread, mirroring the
    # real repo finding, so the diagnostic's undersize_ratio should land well
    # below 1.0 in every quartile, not just the low end.
    true_resid_sd = 30.0
    component_sd = rng.uniform(2.0, 20.0, n)
    model_error = rng.normal(0, true_resid_sd, n)
    # STRONG fires almost always here by construction (mirrors the real gate
    # over-firing under a too-narrow Normal) -- the diagnostic doesn't need to
    # recompute the gate itself, just correlate the ratio with the fire rate.
    signal = np.where(rng.uniform(0, 1, n) < 0.90, "STRONG_EDGE", "NO_EDGE")
    return pd.DataFrame({
        "market": np.tile(["rec_yards", "rush_yards"], n // 2),
        "component_sd": component_sd,
        "model_error": model_error,
        "signal": signal,
    })


def test_diagnose_reports_undersize_ratio_below_one_in_every_quartile():
    detail = _fixture()
    out = diagnose(detail)
    assert set(out["market"]) == {"rec_yards", "rush_yards"}
    per_quartile = out.loc[out["quartile"] != "ALL"]
    assert len(per_quartile) > 0
    # The whole point of this diagnostic: even the highest component_sd
    # quartile should still be meaningfully undersized relative to the real
    # residual spread, not just the lowest one.
    assert (per_quartile["undersize_ratio"] < 1.0).all()


def test_diagnose_fails_closed_on_missing_columns():
    detail = pd.DataFrame({"market": ["rec_yards"], "component_sd": [5.0]})
    with pytest.raises(RuntimeError, match="missing required columns"):
        diagnose(detail)


def test_diagnose_all_row_matches_pooled_market_stats():
    detail = _fixture()
    out = diagnose(detail)
    all_row = out.loc[(out["market"] == "rec_yards") & (out["quartile"] == "ALL")].iloc[0]
    rec = detail.loc[detail.market.eq("rec_yards")]
    assert all_row["n"] == len(rec)
    assert all_row["empirical_resid_sd"] == pytest.approx(float(rec["model_error"].std()))
