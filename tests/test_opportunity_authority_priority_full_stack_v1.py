import numpy as np
import pandas as pd
from unittest.mock import patch

from scripts.modeling.contracts import PlayerContext, TeamContext
from scripts.modeling import simulation_rules as sr
from scripts.simulation_v2 import simulate


def _frame():
    return pd.DataFrame([
        {
            "event_id": "AAA|BBB", "team": "AAA", "opponent": "BBB",
            "player": "RB One", "player_clean_key": "rbone", "position": "RB", "role": "RB1",
            "rules_plays_est": 64.0, "rules_pass_rate": 0.55,
            "rules_tgt_share": 0.10, "rules_rush_share": 0.50,
            "rules_ypt": 6.0, "rules_ypc": 4.2, "rules_catch_rate": 0.75,
        },
        {
            "event_id": "AAA|BBB", "team": "AAA", "opponent": "BBB",
            "player": "WR One", "player_clean_key": "wrone", "position": "WR", "role": "WR1",
            "rules_plays_est": 64.0, "rules_pass_rate": 0.55,
            "rules_tgt_share": 0.28, "rules_rush_share": 0.01,
            "rules_ypt": 8.0, "rules_ypc": 5.0, "rules_catch_rate": 0.64,
        },
    ])


def test_opportunity_trace_is_rng_neutral():
    frame = _frame()
    control = simulate(frame, iterations=300, seed=91)
    trace = []
    audited = simulate(frame, iterations=300, seed=91, allocation_trace=trace)

    assert trace
    by_key = {(r["event_id"], r["team"], r["player_clean_key"]): r for r in trace}
    assert by_key[("AAA|BBB", "AAA", "rbone")]["realized_multinomial_mean_carries"] >= 0
    assert by_key[("AAA|BBB", "AAA", "wrone")]["realized_multinomial_mean_targets"] >= 0

    assert set(control.values) == set(audited.values)
    for key in control.values:
        np.testing.assert_array_equal(control.values[key], audited.values[key])


def test_candidate_authority_keeps_efficiency_and_team_volume_exact(monkeypatch):
    off = TeamContext(team="AAA", season=2025, success_rate_off=0.45, plays_est=64.0)
    deff = TeamContext(team="BBB", season=2025, success_rate_def=0.45)
    players = [
        PlayerContext(
            player="RB One", team="AAA", opponent="BBB", season=2025, week=3,
            position="RB", role="RB1", offense=off, defense=deff,
            features={"rush_share": 0.52, "tgt_share": 0.10, "ypc": 4.2, "ypt": 6.0},
        ),
        PlayerContext(
            player="WR One", team="AAA", opponent="BBB", season=2025, week=3,
            position="WR", role="WR1", offense=off, defense=deff,
            features={"rush_share": 0.01, "tgt_share": 0.28, "ypt": 8.0},
        ),
    ]
    monkeypatch.setattr(sr, "load_model_contexts", lambda: ({}, players))
    metrics = pd.DataFrame([
        {
            "player": "RB One", "player_clean_key": "rbone", "team": "AAA", "position": "RB",
            "rush_share": 0.52, "tgt_share": 0.10,
            "bayes_rush_share": 0.34, "bayes_tgt_share": 0.08,
            "bayes_ypt": 6.0, "bayes_ypc": 4.2,
        },
        {
            "player": "WR One", "player_clean_key": "wrone", "team": "AAA", "position": "WR",
            "rush_share": 0.01, "tgt_share": 0.28,
            "bayes_rush_share": 0.02, "bayes_tgt_share": 0.20,
            "bayes_ypt": 8.0, "bayes_ypc": 5.0,
        },
    ])
    base = sr.apply_rules_to_metrics(metrics)
    cand = sr.apply_rules_to_metrics(
        metrics, opportunity_authority=sr.OPPORTUNITY_AUTHORITY_PLAYERFORM_FAST_STATE
    )
    assert cand.loc[0, "rules_rush_share"] == 0.52
    assert cand.loc[1, "rules_tgt_share"] == 0.28
    for col in (
        "rules_plays_est", "rules_pass_rate", "rules_ypt", "rules_ypc",
        "rules_ypa", "rules_catch_rate", "rules_volatility_mult",
        "rules_pass_eff_mult", "rules_rush_eff_mult",
    ):
        a = pd.to_numeric(base[col], errors="coerce")
        b = pd.to_numeric(cand[col], errors="coerce")
        np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)
