from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.research.run_all_player_all_position_replay_v1 as replay


def test_required_markets_are_individual_player_markets_only():
    assert replay._required_market("QB", "pass_yards")
    assert not replay._required_market("QB", "rec_yards")
    assert replay._required_market("RB", "rush_yards")
    assert replay._required_market("RB", "rec_yards")
    assert replay._required_market("RB", "receptions")
    assert replay._required_market("RB", "rush_rec_yards")
    assert replay._required_market("WR", "rec_yards")
    assert replay._required_market("TE", "receptions")
    assert not replay._required_market("WR", "pass_yards")


def test_depth_feature_is_strictly_prior_and_ignores_target_week():
    rows = []
    games = [
        (2025, 17, "g1", [1.0, 2.0, 3.0]),
        (2025, 18, "g2", [4.0, 5.0, 6.0]),
        (2026, 1, "g3", [7.0, 8.0, 9.0]),
        (2026, 2, "g4", [10.0, 11.0, 12.0]),
        # Must be ignored for a Week-3 replay target.
        (2026, 3, "g5", [1000.0, 1001.0, 1002.0]),
    ]
    for season, week, game_id, air in games:
        for value in air:
            rows.append({
                "season": season,
                "week": week,
                "game_id": game_id,
                "receiver_id": "R1",
                "air_yards": value,
            })
    events = pd.DataFrame(rows)
    feat = replay._depth_feature(events, "R1", week=3)
    assert feat is not None
    assert feat["prior_receiver_games"] == 4
    assert feat["prior_finite_air_targets"] == 12
    assert feat["feature_max_season"] == 2026
    assert feat["feature_max_week"] == 2
    expected = np.std(np.arange(1.0, 13.0), ddof=0)
    assert feat["prior8_target_depth_sd"] == pytest.approx(expected)


def test_depth_feature_fails_closed_below_support():
    rows = []
    for game_n, week in enumerate((15, 16, 17), start=1):
        for value in range(4):
            rows.append({
                "season": 2025,
                "week": week,
                "game_id": f"g{game_n}",
                "receiver_id": "R1",
                "air_yards": float(value + game_n),
            })
    assert replay._depth_feature(pd.DataFrame(rows), "R1", week=1) is None


def test_depth_scale_quantiles_are_reporting_only_and_cover_available_rows():
    frame = pd.DataFrame({
        "feature_available": [True] * 8 + [False],
        "depth_distribution_scale": [0.70, 0.80, 0.90, 0.95, 1.05, 1.10, 1.20, 1.30, 1.0],
        "absolute_point_error": np.arange(9, dtype=float),
        "baseline_crps": np.arange(9, dtype=float) + 1.0,
        "shadow_crps": np.arange(9, dtype=float) + 0.5,
        "crps_improvement": [0.5] * 9,
    })
    out = replay._add_depth_scale_quantiles(frame)
    assert out.loc[8, "depth_scale_quantile"] == "UNAVAILABLE"
    assert set(out.loc[:7, "depth_scale_quantile"]) == {"Q1_LOW", "Q2", "Q3", "Q4_HIGH"}
    summary = replay._depth_quantile_summary(out)
    assert sum(r["rows"] for r in summary) == 8
    assert all("mean_absolute_point_error" in r for r in summary)


def test_empirical_crps_is_zero_for_perfect_degenerate_distribution():
    draws = np.repeat(12.5, 100)
    assert replay._empirical_crps(draws, 12.5) == pytest.approx(0.0)


def test_qb_synthesis_writes_back_to_explicit_source_index(monkeypatch):
    frame = pd.DataFrame([
        {
            "market": "pass_yards", "position_family": "QB", "ensemble_proj": 250.0,
            "mc_proj": 245.0, "ml_proj": 255.0, "state_proj": 252.0,
            "week": 2, "team": "IND", "opponent": "HOU", "player_clean_key": "qb1",
        },
        {
            "market": "pass_yards", "position_family": "QB", "ensemble_proj": 200.0,
            "mc_proj": 195.0, "ml_proj": 205.0, "state_proj": 201.0,
            "week": 2, "team": "HOU", "opponent": "IND", "player_clean_key": "qb2",
        },
    ], index=[7, 11])

    def fake_history(q, team_weekly, player_logs):
        # Reproduce the production helper's index reset while retaining the
        # explicit source index column carried by the replay.
        return q.reset_index(drop=True)

    monkeypatch.setattr(replay, "add_history_features", fake_history)
    monkeypatch.setattr(replay, "load_qb_artifact", lambda: {"feature_contract": ["base_proj"]})
    monkeypatch.setattr(
        replay,
        "predict_qb_correction",
        lambda features, artifact=None: (float(features["base_proj"]) + 3.0, 3.0, "TEST_QB"),
    )

    out = replay._apply_qb_synthesis(
        frame,
        player_logs=pd.DataFrame(),
        team_weekly=pd.DataFrame(),
        controlled_map={},
    )
    assert out.loc[7, "projection_mean"] == pytest.approx(253.0)
    assert out.loc[11, "projection_mean"] == pytest.approx(203.0)
    assert out.loc[7, "qb_synthesis_applied"]
    assert out.loc[11, "qb_synthesis_applied"]


def test_week1_rb_combo_uses_p3_path_then_combo_ensemble(monkeypatch):
    frame = pd.DataFrame([
        {
            "event_id": "E1", "season": 2026, "week": 1, "team": "IND", "opponent": "HOU",
            "player": "Back One", "player_clean_key": "backone", "position_family": "RB",
            "market": "rush_yards", "projection_mean": 30.0, "ensemble_proj": 30.0,
            "mc_proj": 30.0, "ml_proj": 31.0, "state_proj": 32.0,
        },
        {
            "event_id": "E1", "season": 2026, "week": 1, "team": "IND", "opponent": "HOU",
            "player": "Back One", "player_clean_key": "backone", "position_family": "RB",
            "market": "rush_rec_yards", "projection_mean": 45.0, "ensemble_proj": 45.0,
            "mc_proj": 45.0, "ml_proj": 65.0, "state_proj": 66.0,
        },
    ])

    monkeypatch.setattr(replay, "_rb_p3_player_in_scope", lambda row, ctx: True)
    monkeypatch.setattr(
        replay,
        "lookup_rb_projection",
        lambda row, ctx: {
            "rb_synthesis_proj": 50.0,
            "rb_synthesis_route": "WEEK1_STACK_OVERRIDE",
            "rb_synthesis_version": "RB_P3_SYNTHESIS_V1",
            "rb_synthesis_applied": 1,
        },
    )
    monkeypatch.setattr(
        replay,
        "build_candidate_map",
        lambda metrics, sims, weights: ({}, {"week1_rows_changed": 0}),
    )

    def fake_lookup(sims, row, market):
        if market == "rush_yards":
            return np.array([10.0, 20.0])
        if market == "rec_yards":
            return np.array([5.0, 15.0])
        return np.array([15.0, 35.0])

    monkeypatch.setattr(replay, "lookup", fake_lookup)

    def fake_ensemble(component, weights=None):
        out = component.copy()
        # Raw rush mean=15 -> scaled to P3 50; raw receiving mean=10.
        # Therefore conserved combo MC must be 60 before the combo ensemble.
        assert float(out.iloc[0]["mc_proj"]) == pytest.approx(60.0)
        out["ensemble_proj"] = 70.0
        out["ensemble_status"] = "calibrated"
        return out

    monkeypatch.setattr(replay, "apply_ensemble", fake_ensemble)

    out = replay._apply_rb_authorities(
        frame,
        sims=object(),
        weights=pd.DataFrame({"market": ["rush_rec_yards"]}),
        rb_context=pd.DataFrame({"team": ["IND"]}),
    )
    rush = out.loc[out["market"].eq("rush_yards")].iloc[0]
    combo = out.loc[out["market"].eq("rush_rec_yards")].iloc[0]
    assert rush["projection_mean"] == pytest.approx(50.0)
    assert combo["mc_proj"] == pytest.approx(60.0)
    assert combo["projection_mean"] == pytest.approx(70.0)
    assert combo["rb_p3_applied"]
    assert combo["rb_p3_route"] == "WEEK1_P3_PATHWISE_CONSERVATION_THEN_COMBO_ENSEMBLE"


def test_rostered_player_missing_weekly_stats_is_verified_zero():
    frame = pd.DataFrame([{
        "team": "IND", "player": "Zero Player", "player_clean_key": replay._pkey("Zero Player"),
        "market": "rec_yards",
    }])
    universe = pd.DataFrame([{"team": "IND", "player": "Zero Player"}])
    logs = pd.DataFrame(columns=["season", "week", "player", "team"])
    out = replay._attach_actuals(frame, player_logs=logs, pregame_universe=universe, week=1)
    assert out.iloc[0]["actual"] == 0.0
    assert out.iloc[0]["actual_source"] == "PREGAME_ROSTER_NO_WEEKLY_STAT_ROW_ZERO"


def test_week1_p3_scope_is_exact_player_identity_not_team_only():
    key = replay.player_name_key("Ameer Abdullah", strip_suffix=True)
    context = pd.DataFrame([{
        "season": 2026,
        "week": 1,
        "team": "JAX",
        "player_base_key": replay.player_name_key("Bhayshul Tuten", strip_suffix=True),
    }])
    row = pd.Series({
        "season": 2026,
        "week": 1,
        "team": "JAX",
        "player": "Ameer Abdullah",
    })
    assert key != context.iloc[0]["player_base_key"]
    assert replay._rb_p3_player_in_scope(row, context) is False

    context2 = pd.concat([
        context,
        pd.DataFrame([{
            "season": 2026,
            "week": 1,
            "team": "JAX",
            "player_base_key": key,
        }]),
    ], ignore_index=True)
    assert replay._rb_p3_player_in_scope(row, context2) is True
