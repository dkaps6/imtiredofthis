import numpy as np
import pandas as pd
import pytest

from scripts.modeling.rush_pool_evidence_guard_v1 import ENV_VAR, VERSION, stable_rush_seed
from scripts.simulation_v2 import simulate


def _metrics(week=2):
    rows = [
        # player, position, role, rush share, target share, evidence
        ("Starter QB", "QB", "QB1", 0.18, 0.00, "prior+current"),
        ("Fallback QB", "QB", "QB2", 0.14, 0.00, "position_prior_only"),
        ("Lead RB", "RB", "RB1", 0.30, 0.10, "prior+current"),
        ("Second RB", "RB", "RB2", 0.20, 0.08, "prior_only"),
        ("Wideout", "WR", "WR1", 0.02, 0.30, "prior+current"),
        ("Tight End", "TE", "TE1", 0.01, 0.20, "prior+current"),
    ]
    out = []
    for player, pos, role, rush, tgt, state in rows:
        out.append({
            "event_id": "GAME1",
            "team": "AAA",
            "opponent": "BBB",
            "player": player,
            "player_clean_key": player.lower().replace(" ", ""),
            "position": pos,
            "role": role,
            "model_role": role,
            "season": 2026,
            "week": week,
            "rules_plays_est": 64.0,
            "rules_pass_rate": 0.58,
            "rules_tgt_share": tgt,
            "rules_rush_share": rush,
            "bayes_evidence_state": state,
            "rules_catch_rate": 0.67,
            "rules_ypt": 7.5,
            "rules_ypc": 4.3,
            "rules_ypa": 7.1,
            "rules_volatility_mult": 1.0,
            "offensive_td_rate": 0.10,
            "rz_share": 0.10,
            "team_wp": 0.50,
        })
    return pd.DataFrame(out)


def _run(monkeypatch, *, enabled, week=2):
    if enabled:
        monkeypatch.setenv(ENV_VAR, "1")
    else:
        monkeypatch.delenv(ENV_VAR, raising=False)
    trace = []
    result = simulate(_metrics(week=week), iterations=400, seed=12345, allocation_trace=trace)
    return result, pd.DataFrame(trace)


def test_dedicated_seed_is_stable_and_scoped():
    a = stable_rush_seed(simulation_seed=42, game="G", team="AAA")
    b = stable_rush_seed(simulation_seed=42, game="G", team="AAA")
    c = stable_rush_seed(simulation_seed=42, game="G", team="BBB")
    assert a == b
    assert a != c


def test_week2_guard_changes_only_rushing_outputs_and_preserves_base_rng(monkeypatch):
    base, bt = _run(monkeypatch, enabled=False, week=2)
    cand, ct = _run(monkeypatch, enabled=True, week=2)

    assert set(base.values) == set(cand.values)

    for key in base.values:
        market = key[2]
        if market in {"receptions", "rec_yards", "pass_yards", "anytime_td"}:
            assert np.array_equal(base.values[key], cand.values[key]), key

    assert any(
        not np.array_equal(base.values[key], cand.values[key])
        for key in base.values
        if key[2] == "rush_att"
    )

    # Candidate must preserve the canonical team-rush and target draws exactly.
    assert bt["team_rush_total_sha256"].nunique() == 1
    assert ct["team_rush_total_sha256"].nunique() == 1
    assert bt["team_rush_total_sha256"].iloc[0] == ct["team_rush_total_sha256"].iloc[0]

    assert bt["target_allocation_sha256"].nunique() == 1
    assert ct["target_allocation_sha256"].nunique() == 1
    assert bt["target_allocation_sha256"].iloc[0] == ct["target_allocation_sha256"].iloc[0]

    # Baseline carry allocation is still drawn from the canonical RNG in candidate mode.
    assert bt["final_carry_allocation_sha256"].iloc[0] == ct["baseline_carry_allocation_sha256"].iloc[0]
    assert ct["final_carry_allocation_sha256"].iloc[0] != ct["baseline_carry_allocation_sha256"].iloc[0]

    assert set(ct["rush_pool_evidence_guard_v1_version"]) == {VERSION}
    assert set(ct["rush_pool_evidence_guard_v1_enabled"]) == {1}
    assert set(ct["rush_pool_evidence_guard_v1_applied"]) == {1}


def test_week1_is_bit_exact_noop_even_when_enabled(monkeypatch):
    base, bt = _run(monkeypatch, enabled=False, week=1)
    cand, ct = _run(monkeypatch, enabled=True, week=1)

    assert set(base.values) == set(cand.values)
    for key in base.values:
        assert np.array_equal(base.values[key], cand.values[key]), key

    assert bt["final_carry_allocation_sha256"].iloc[0] == ct["final_carry_allocation_sha256"].iloc[0]
    assert set(ct["rush_pool_evidence_guard_v1_enabled"]) == {1}
    assert set(ct["rush_pool_evidence_guard_v1_applied"]) == {0}
    assert set(ct["rush_pool_evidence_guard_v1_reason"]) == {"week1_noop"}


def test_enabled_guard_fails_closed_without_evidence_state(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1")
    bad = _metrics(week=2).drop(columns=["bayes_evidence_state"])
    with pytest.raises(RuntimeError, match="bayes_evidence_state"):
        simulate(bad, iterations=10, seed=1)
