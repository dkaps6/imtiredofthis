from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.rb_pricing_adapter_v1 import load_rb_context, lookup_rb_projection


def _context() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "season": 2026,
            "week": 1,
            "player": "D'Andre Swift",
            "player_clean_key": "dandreswift",
            "player_base_key": "dandreswift",
            "team": "CHI",
            "opponent": "CAR",
            "rb_synthesis_proj": 51.25,
            "rb_synthesis_route": "WEEK1_STACK_OVERRIDE",
            "rb_synthesis_version": "RB_P3_SYNTHESIS_V1",
            "rb_synthesis_applied": 1,
            "football_only_no_odds": 1,
            "sportsbook_inputs_used": 0,
            "rb_stack_implied_ypc": 4.1,
            "rb_ypc_fallback_used": 0,
        }
    ])


def test_lookup_matches_punctuation_normalized_identity():
    row = pd.Series({
        "season": 2026,
        "week": 1,
        "player": "D’Andre Swift",
        "player_clean_key": "D'Andre-Swift",
        "team": "chi",
        "opponent": "car",
    })
    got = lookup_rb_projection(row, _context())
    assert got["rb_synthesis_proj"] == pytest.approx(51.25)
    assert got["rb_synthesis_route"] == "WEEK1_STACK_OVERRIDE"
    assert got["rb_synthesis_applied"] == 1


def test_lookup_fails_closed_on_missing_identity():
    row = pd.Series({"season": 2026, "week": 1, "player": "Nobody", "team": "CHI", "opponent": "CAR"})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        lookup_rb_projection(row, _context())


def test_lookup_fails_closed_on_opponent_mismatch():
    row = pd.Series({"season": 2026, "week": 1, "player": "D'Andre Swift", "team": "CHI", "opponent": "GB"})
    with pytest.raises(RuntimeError, match="opponent mismatch"):
        lookup_rb_projection(row, _context())


def test_load_rejects_sportsbook_leakage(tmp_path: Path):
    bad = _context()
    bad.loc[:, "sportsbook_inputs_used"] = 1
    path = tmp_path / "rb.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="sportsbook leakage"):
        load_rb_context(path)


def test_load_rejects_duplicate_identity(tmp_path: Path):
    bad = pd.concat([_context(), _context()], ignore_index=True)
    path = tmp_path / "rb.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="duplicate promoted RB context identities"):
        load_rb_context(path)


def test_canonical_pricing_and_full_slate_are_wired_to_p3():
    pricing = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/full-slate.yml").read_text(encoding="utf-8")
    assert "load_rb_context" in pricing
    assert "lookup_rb_projection" in pricing
    assert "target_mean = rb_synthesis_proj" in pricing
    assert "promoted RB production pricing is currently locked to Week 1" in pricing
    assert "Build promoted RB P3 football-only context" in workflow
    assert "final RB model projection is not the promoted P3 synthesis mean" in workflow


def test_live_week1_context_prices_end_to_end_without_sportsbook_model_input():
    """In the promotion workflow this becomes the real end-to-end pricing gate.

    The early unit-test step runs before live Week-1 artifacts exist, so this
    test skips there.  The workflow's final full-suite pytest runs after the
    107-player no-odds P3 context, PlayerForm, ML and State artifacts exist; at
    that point it feeds synthetic downstream lines/odds to the *real* pricing
    function and requires every final rush-yard mean to remain exactly P3.
    """
    required = [
        Path("data/rb_rush_synthesis_context.csv"),
        Path("data/model_ml_diagnostics.csv"),
        Path("data/model_state_diagnostics.csv"),
        Path("data/player_form.csv"),
        Path("data/team_form.csv"),
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        pytest.skip("live Week-1 promotion artifacts not materialized in this test phase")

    from scripts.run_rb_week1_no_odds import build_internal_rb_metrics
    from scripts.run_pricing_v2 import price

    ctx = load_rb_context()
    metrics = build_internal_rb_metrics(2026, 1)
    metrics = metrics.loc[metrics["market"].astype(str).str.lower().eq("rush_yards")].copy()
    assert not metrics.empty
    assert metrics["team"].nunique() == 32

    # These are synthetic downstream pricing inputs only.  They are deliberately
    # constant and are never available to the P3 context/model construction.
    metrics["line"] = 50.5
    metrics["over_odds"] = -110
    metrics["under_odds"] = -110
    metrics["book"] = "SYNTHETIC_PROMOTION_GATE"
    metrics["book_title"] = "Synthetic Promotion Gate"
    metrics.to_csv("data/metrics_ready.csv", index=False)

    out = price(2026)
    rb = out.loc[out["market"].astype(str).str.lower().eq("rush_yards")].copy()
    assert not rb.empty
    assert rb["player_clean_key"].nunique() == ctx["player_clean_key"].nunique()
    assert rb["team"].nunique() == 32
    assert pd.to_numeric(rb["rb_synthesis_applied"], errors="coerce").eq(1).all()
    assert rb["rb_synthesis_version"].astype(str).eq("RB_P3_SYNTHESIS_V1").all()
    assert rb["rb_synthesis_route"].astype(str).eq("WEEK1_STACK_OVERRIDE").all()
    assert rb["ensemble_status"].astype(str).eq("calibrated").all()
    final = pd.to_numeric(rb["model_proj"], errors="coerce")
    p3 = pd.to_numeric(rb["rb_synthesis_proj"], errors="coerce")
    assert final.notna().all() and p3.notna().all()
    assert np.allclose(final, p3, rtol=0, atol=1e-8)
