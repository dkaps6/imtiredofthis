"""Guards for the MC dispersion calibration.

The thing that makes this safe to put on the pricing path is that it cannot move a
player's mean -- the promoted QB/RB syntheses depend on that -- so that is what the
guards check hardest, along with every fail-soft path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.modeling import mc_dispersion_calibration_v1 as M


def _draws(mean=60.0, sd=18.0, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, sd, n), 0.0, None)


def _cal(**k):
    return {"state": "current", "version": M.VERSION, "k": k}


def test_mean_is_preserved_exactly():
    """The invariant the promoted syntheses rely on (AGENTS.md section 7)."""
    for mean in (3.0, 12.0, 60.0, 280.0):
        d = _draws(mean=mean, sd=mean * 0.6, seed=1)
        out, meta = M.apply_dispersion_calibration(d, "rec_yards", _cal(rec_yards=1.64))
        assert meta["mc_dispersion_applied"] == 1
        assert out.mean() == pytest.approx(d.mean(), rel=1e-12), f"mean moved at mean={mean}"


def test_dispersion_actually_widens():
    d = _draws(seed=2)
    out, meta = M.apply_dispersion_calibration(d, "rush_yards", _cal(rush_yards=1.756))
    assert out.std(ddof=1) > d.std(ddof=1) * 1.3
    assert meta["mc_dispersion_sd_after"] > meta["mc_dispersion_sd_before"]


def test_naive_widen_then_clip_would_have_moved_the_mean():
    """Documents why the shipped transform restores the mean after clipping."""
    d = _draws(mean=4.0, sd=6.0, seed=3)          # low mean: clipping bites hard
    mu = d.mean()
    naive = np.clip(mu + 1.9 * (d - mu), 0.0, None)
    assert abs(naive.mean() - mu) / mu > 0.05, "fixture should show a real naive shift"
    out, _ = M.apply_dispersion_calibration(d, "rush_yards", _cal(rush_yards=1.9))
    assert out.mean() == pytest.approx(mu, rel=1e-12)


def test_non_negativity_is_preserved():
    out, _ = M.apply_dispersion_calibration(_draws(mean=5.0, sd=9.0, seed=4),
                                            "rec_yards", _cal(rec_yards=2.0))
    assert (out >= 0).all()


def test_uncalibrated_market_passes_through_unchanged_and_labelled():
    d = _draws(seed=5)
    out, meta = M.apply_dispersion_calibration(d, "anytime_td", _cal(rec_yards=1.64))
    assert np.array_equal(out, d)
    assert meta["mc_dispersion_applied"] == 0
    assert meta["mc_dispersion_state"] == "market_not_calibrated"
    assert meta["mc_dispersion_k"] == M.NEUTRAL_K


def test_missing_artifact_fails_soft_but_visible(tmp_path):
    cal = M.load_calibration(tmp_path / "nope.json")
    assert cal["state"] == "unavailable" and cal["k"] == {}
    d = _draws(seed=6)
    out, meta = M.apply_dispersion_calibration(d, "rec_yards", cal)
    assert np.array_equal(out, d), "a missing artifact must degrade to today's behaviour"
    assert meta["mc_dispersion_state"] == "unavailable"


def test_version_mismatch_is_refused(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": "SOMETHING_ELSE", "markets": {"rec_yards": {"k": 1.6}}}))
    cal = M.load_calibration(p)
    assert cal["state"] == "version_mismatch" and cal["k"] == {}


def test_out_of_range_k_is_rejected_not_applied(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": M.VERSION, "markets": {
        "rec_yards": {"k": 1.64}, "rush_yards": {"k": 99.0}, "receptions": {"k": -1.0}}}))
    cal = M.load_calibration(p)
    # no promotion_state in the fixture -> treated as a candidate, so active k is empty
    assert cal["k"] == {} and set(cal["k_candidate"]) == {"rec_yards"}
    assert set(cal["rejected"]) == {"rush_yards", "receptions"}


def test_shipped_artifact_is_loadable_and_in_range():
    cal = M.load_calibration()
    assert cal["state"] in ("current", "candidate_not_promoted"), cal.get("reason")
    ks = cal["k"] or cal.get("k_candidate") or {}
    assert ks, "no markets calibrated"
    for market, k in ks.items():
        assert M.MIN_K <= k <= M.MAX_K, f"{market} k={k} out of range"
    assert not cal.get("rejected"), f"shipped artifact has rejected entries: {cal['rejected']}"


def test_production_pricer_applies_it_before_integer_alignment():
    """Order matters: count markets must be re-aligned to integers AFTER widening."""
    src = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    i_disp = src.index("apply_dispersion_calibration(")
    i_align = src.index("align_prealigned_outcomes(\n            adjusted_outcomes")
    assert i_disp < i_align, "dispersion calibration must precede integer alignment"
    # and it must come after the football mean is aligned
    i_mean = src.index("adjusted_outcomes = base_outcomes * max(0.0, target_mean / mc_proj)")
    assert i_mean < i_disp, "dispersion calibration must follow the mean alignment"


def test_audit_columns_reach_the_priced_row():
    src = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    assert "**mc_dispersion_meta," in src, "audit columns are not emitted on the priced row"


# --- added after GPT audit 6472971619 ---

def test_candidate_artifact_is_inert_in_production():
    """Until the parity replay certifies k, the calibration must not change a price."""
    cal = M.load_calibration(live_pricing=True)
    assert cal["promotion_state"] == M.CANDIDATE_STATE, "artifact should still be a candidate"
    assert cal["k"] == {}, "a candidate must expose no active multipliers"
    assert cal["k_candidate"], "candidate k should still be visible for audit"
    d = _draws(seed=11)
    out, meta = M.apply_dispersion_calibration(d, "rec_yards", cal)
    assert np.array_equal(out, d)
    assert meta["mc_dispersion_applied"] == 0
    assert meta["mc_dispersion_state"] == "candidate_not_promoted"


def test_promoted_artifact_applies(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "CALIBRATION_REQUIRED_FOR_LIVE_PRICING", True)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": M.VERSION, "promotion_state": M.PROMOTED_STATE,
                             "markets": {"rec_yards": {"k": 1.64}}}))
    cal = M.load_calibration(p, live_pricing=True)
    assert cal["k"] == {"rec_yards": 1.64}
    d = _draws(seed=12)
    out, meta = M.apply_dispersion_calibration(d, "rec_yards", cal)
    assert meta["mc_dispersion_applied"] == 1
    assert out.mean() == pytest.approx(d.mean(), rel=1e-12)


def test_promoted_but_unusable_artifact_FAILS_CLOSED_for_live_pricing(tmp_path):
    """AGENTS.md section 4: a required production artifact must not silently revert."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": M.VERSION, "promotion_state": M.PROMOTED_STATE,
                             "markets": {"rec_yards": {"k": 99.0}}}))   # all rejected
    with pytest.raises(M.DispersionCalibrationUnavailable):
        M.load_calibration(p, live_pricing=True)
    # research use must still be able to inspect it without raising
    cal = M.load_calibration(p, live_pricing=False)
    assert cal["k"] == {} and cal["rejected"]


def test_a_promoted_artifact_alone_cannot_put_the_calibration_live(tmp_path):
    """Flipping the file without the code flag must NOT move a price, and must refuse."""
    assert M.CALIBRATION_REQUIRED_FOR_LIVE_PRICING is False
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": M.VERSION, "promotion_state": M.PROMOTED_STATE,
                             "markets": {"rec_yards": {"k": 1.64}}}))
    cal = M.load_calibration(p, live_pricing=False)
    assert cal["k"] == {}, "artifact alone must not activate k"
    d = _draws(seed=21)
    out, meta = M.apply_dispersion_calibration(d, "rec_yards", cal)
    assert np.array_equal(out, d) and meta["mc_dispersion_applied"] == 0
    with pytest.raises(M.DispersionCalibrationUnavailable, match="authorities disagree"):
        M.load_calibration(p, live_pricing=True)


def test_both_authorities_agreeing_activates_it(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "CALIBRATION_REQUIRED_FOR_LIVE_PRICING", True)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": M.VERSION, "promotion_state": M.PROMOTED_STATE,
                             "markets": {"rec_yards": {"k": 1.64}}}))
    cal = M.load_calibration(p, live_pricing=True)
    assert cal["k"] == {"rec_yards": 1.64}
    assert cal["artifact_promoted"] and cal["code_requires_calibration"]


def test_production_pricer_loads_with_live_pricing_true():
    src = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    assert "load_mc_dispersion_calibration(live_pricing=True)" in src


# --- GPT audit 6473505789 section 3: promotion authority must be independent ---

def test_promotion_authority_lives_in_code_not_only_in_the_artifact():
    """A file cannot make itself required: if it is gone, so is its own promotion flag."""
    assert hasattr(M, "CALIBRATION_REQUIRED_FOR_LIVE_PRICING")
    assert M.CALIBRATION_REQUIRED_FOR_LIVE_PRICING is False, "must stay False until promoted"


@pytest.mark.parametrize("failure", ["missing", "unreadable", "version_mismatch", "all_rejected"])
def test_required_calibration_fails_closed_on_every_failure_class(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(M, "CALIBRATION_REQUIRED_FOR_LIVE_PRICING", True)
    p = tmp_path / "c.json"
    if failure == "missing":
        pass
    elif failure == "unreadable":
        p.write_text("{ this is not json")
    elif failure == "version_mismatch":
        p.write_text(json.dumps({"version": "OLD", "promotion_state": M.PROMOTED_STATE,
                                 "markets": {"rec_yards": {"k": 1.64}}}))
    else:
        p.write_text(json.dumps({"version": M.VERSION, "promotion_state": M.PROMOTED_STATE,
                                 "markets": {"rec_yards": {"k": 99.0}}}))
    with pytest.raises(M.DispersionCalibrationUnavailable):
        M.load_calibration(p, live_pricing=True)


@pytest.mark.parametrize("failure", ["missing", "unreadable", "version_mismatch", "all_rejected"])
def test_the_same_failures_are_harmless_while_the_calibration_is_optional(tmp_path, failure):
    """With the code authority False, absence must never break a slate."""
    assert M.CALIBRATION_REQUIRED_FOR_LIVE_PRICING is False
    p = tmp_path / "c.json"
    if failure == "unreadable":
        p.write_text("{ nope")
    elif failure == "version_mismatch":
        p.write_text(json.dumps({"version": "OLD", "markets": {}}))
    elif failure == "all_rejected":
        p.write_text(json.dumps({"version": M.VERSION, "markets": {"rec_yards": {"k": 99.0}}}))
    cal = M.load_calibration(p, live_pricing=True)      # must not raise
    assert cal["k"] == {}


def test_code_and_artifact_authorities_must_agree(tmp_path, monkeypatch):
    """Promoted in code but still a candidate in the artifact is a contradiction."""
    monkeypatch.setattr(M, "CALIBRATION_REQUIRED_FOR_LIVE_PRICING", True)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": M.VERSION, "promotion_state": M.CANDIDATE_STATE,
                             "markets": {"rec_yards": {"k": 1.64}}}))
    with pytest.raises(M.DispersionCalibrationUnavailable, match="authorities disagree"):
        M.load_calibration(p, live_pricing=True)


def test_research_callers_never_trip_the_required_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "CALIBRATION_REQUIRED_FOR_LIVE_PRICING", True)
    cal = M.load_calibration(tmp_path / "absent.json", live_pricing=False)
    assert cal["state"] == "unavailable"
