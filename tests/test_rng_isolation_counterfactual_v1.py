import pandas as pd

from scripts.research.audit_rng_isolation_counterfactual_v1 import (
    _candidate_gate,
    _envelope,
)


def _row(surface, seed, p99p, p99e, betpass, identity, top10, top25, spearman=0.999, maxp=0.03):
    return {
        "surface": surface,
        "market_scope": "ALL_SUPPORTED",
        "seed": seed,
        "p99_abs_prob_delta": p99p,
        "p99_abs_ev_delta": p99e,
        "best_snapshot_bet_pass_flips": betpass,
        "best_snapshot_identity_changes": identity,
        "top10_turnover": top10,
        "top25_turnover": top25,
        "max_abs_prob_delta": maxp,
        "max_abs_ev_delta": 0.04,
        "quote_preferred_side_flips": 2,
        "quote_has_edge_pass_flips": 3,
        "best_snapshot_side_flips": 0,
        "mean_abs_best_ev_delta": 0.002,
        "max_abs_best_ev_delta": 0.03,
        "best_ev_spearman": spearman,
    }


def test_envelope_uses_maximum_across_all_preregistered_seeds(monkeypatch):
    import scripts.research.audit_rng_isolation_counterfactual_v1 as mod
    monkeypatch.setattr(mod, "ALT_SEEDS", [1, 2, 3])
    df = pd.DataFrame([
        _row("SHAPE_ONLY_FIXED_FINAL_MEAN", 1, .01, .02, 1, 0, 0, 1),
        _row("SHAPE_ONLY_FIXED_FINAL_MEAN", 2, .02, .01, 2, 1, 1, 0),
        _row("SHAPE_ONLY_FIXED_FINAL_MEAN", 3, .015, .03, 0, 0, 0, 2),
    ])
    env = _envelope(df, "SHAPE_ONLY_FIXED_FINAL_MEAN")
    assert env["p99_abs_prob_delta"] == .02
    assert env["p99_abs_ev_delta"] == .03
    assert env["best_snapshot_bet_pass_flips"] == 2
    assert env["best_snapshot_identity_changes"] == 1
    assert env["top10_turnover"] == 1
    assert env["top25_turnover"] == 2


def test_candidate_gate_passes_inside_envelope():
    surfaces = ["SHAPE_ONLY_FIXED_FINAL_MEAN", "FULL_DOWNSTREAM_PROPAGATION"]
    candidate = pd.DataFrame([
        _row(s, None, .01, .02, 1, 0, 0, 1, spearman=.995, maxp=.04)
        for s in surfaces
    ])
    env = {
        s: {
            "p99_abs_prob_delta": .02,
            "p99_abs_ev_delta": .03,
            "best_snapshot_bet_pass_flips": 2,
            "best_snapshot_identity_changes": 1,
            "top10_turnover": 1,
            "top25_turnover": 2,
            "max_abs_prob_delta": .06,
            "max_abs_ev_delta": .08,
            "quote_preferred_side_flips": 5,
            "quote_has_edge_pass_flips": 5,
            "best_snapshot_side_flips": 1,
            "mean_abs_best_ev_delta": .01,
            "max_abs_best_ev_delta": .05,
        }
        for s in surfaces
    }
    passed, detail = _candidate_gate(candidate, env)
    assert passed is True
    assert all(v["checks"]["surface_pass"] for v in detail.values())


def test_candidate_gate_fails_strict_probability_bound():
    surfaces = ["SHAPE_ONLY_FIXED_FINAL_MEAN", "FULL_DOWNSTREAM_PROPAGATION"]
    candidate = pd.DataFrame([
        _row(s, None, .01, .02, 1, 0, 0, 1, spearman=.995, maxp=.051)
        for s in surfaces
    ])
    env = {
        s: {
            "p99_abs_prob_delta": .02,
            "p99_abs_ev_delta": .03,
            "best_snapshot_bet_pass_flips": 2,
            "best_snapshot_identity_changes": 1,
            "top10_turnover": 1,
            "top25_turnover": 2,
            "max_abs_prob_delta": .06,
            "max_abs_ev_delta": .08,
            "quote_preferred_side_flips": 5,
            "quote_has_edge_pass_flips": 5,
            "best_snapshot_side_flips": 1,
            "mean_abs_best_ev_delta": .01,
            "max_abs_best_ev_delta": .05,
        }
        for s in surfaces
    }
    passed, detail = _candidate_gate(candidate, env)
    assert passed is False
    assert all(not v["checks"]["max_prob_le_5pct"] for v in detail.values())
