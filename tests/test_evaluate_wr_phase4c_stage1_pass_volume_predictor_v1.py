import numpy as np
import pandas as pd

from scripts.research.evaluate_wr_phase4c_stage1_pass_volume_predictor_v1 import (
    SCRIPT_FEATURES,
    build_script_feature_frame,
    crossfit_script_predictions,
    fit_downstream_models,
    build_sealed_2024_predictions,
    paired_cluster_bootstrap_ci,
    stage1_disposition,
)


def _toy_schedule():
    rows = []
    pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    for season in [2021, 2022, 2023, 2024]:
        for week in range(1, 19):
            total = 40.0 + (week % 5)
            for home, away in pairs:
                gid = f"{season}_{week:02d}_{away}_{home}"
                rows += [
                    dict(season=season, week=week, team=home, opponent=away, game_id=gid,
                         home_flag=1.0, team_rest=7.0, opp_rest=7.0, rest_diff=0.0, market_total=total),
                    dict(season=season, week=week, team=away, opponent=home, game_id=gid,
                         home_flag=0.0, team_rest=7.0, opp_rest=7.0, rest_diff=0.0, market_total=total),
                ]
    return pd.DataFrame(rows)


def _toy_weekly():
    rows = []
    teams = list("ABCDEFGH")
    for season in [2021, 2022, 2023, 2024]:
        for week in range(1, 19):
            for idx, team in enumerate(teams):
                rows.append(dict(
                    season=season, week=week, team=team,
                    plays_est=60 + idx * 0.5 + 0.2 * week,
                    dropback_rate=0.50 + 0.002 * week + idx * 0.001,
                ))
    return pd.DataFrame(rows)


def test_rolling_features_are_strictly_prior():
    x = build_script_feature_frame(_toy_weekly(), _toy_schedule())
    z = x.loc[(x.season == 2022) & (x.week == 5) & (x.team == "A")].iloc[0]
    assert z.team_source_max_ordinal < z.target_ordinal
    assert z.defense_team_source_max_ordinal < z.target_ordinal
    tw = _toy_weekly()
    tw.loc[(tw.season == 2022) & (tw.week == 5) & (tw.team == "A"), "plays_est"] = 9999
    y = build_script_feature_frame(tw, _toy_schedule())
    w = y.loc[(y.season == 2022) & (y.week == 5) & (y.team == "A")].iloc[0]
    assert np.isclose(z.team_plays_prior8, w.team_plays_prior8)
    assert np.isclose(z.team_dropback_rate_prior8, w.team_dropback_rate_prior8)


def test_script_crossfit_never_trains_on_prediction_season():
    x = build_script_feature_frame(_toy_weekly(), _toy_schedule())
    pred, audit = crossfit_script_predictions(x)
    assert set(pred.season.unique()) == {2023, 2024}
    assert (pred.script_fit_max_season < pred.season).all()
    by = {m["prediction_season"]: m["train_seasons"] for m in audit["models"]}
    assert by[2023] == [2022]
    assert by[2024] == [2022, 2023]


def _toy_phase4b_and_script():
    rows, sp, sched = [], [], []
    for season in [2023, 2024]:
        for i in range(500):
            week = (i % 18) + 1
            team = f"T{i:03d}"
            gid = f"{season}_{i//2:03d}"
            implied = 30 + (i % 3)
            script = 32 + (i % 20) * 0.2
            total = 40 + (i % 5)
            actual = implied + 0.4 + 0.03 * total + 0.02 * script
            rows.append(dict(
                season=season, week=week, team=team,
                implied_team_target_pool=implied, candidate_wr_room_mass=.5,
                actual_team_targets=actual, actual_wr_room_targets=actual * .5,
            ))
            sp.append(dict(
                season=season, week=week, team=team,
                pred_realized_pass_volume=script, naive_pass_volume=script - 1.0,
            ))
            sched.append(dict(
                season=season, week=week, team=team, market_total=total, game_id=gid,
            ))
    return pd.DataFrame(rows), pd.DataFrame(sp), pd.DataFrame(sched)


def test_downstream_nesting_and_sealed_predictions_have_no_actuals():
    l23, sp, sched = _toy_phase4b_and_script()
    train = l23[l23.season.eq(2023)].merge(sp, on=["season", "week", "team"]).merge(
        sched, on=["season", "week", "team"]
    )
    train["low_total_lt38"] = train.market_total.lt(38).astype(float)
    train["residual_target"] = train.actual_team_targets - train.implied_team_target_pool
    fit = fit_downstream_models(train)
    struct = l23[["season", "week", "team", "implied_team_target_pool", "candidate_wr_room_mass"]]
    sealed = build_sealed_2024_predictions(struct, sp, sched, fit)
    assert {"pool_A0", "pool_A", "pool_B", "pool_C"}.issubset(sealed.columns)
    assert "actual_team_targets" not in sealed.columns
    assert "actual_wr_room_targets" not in sealed.columns
    assert fit.model_b.n_features_in_ == 2
    assert fit.model_c.n_features_in_ == 3


def test_game_cluster_bootstrap_keeps_game_as_cluster_unit():
    x = pd.DataFrame({"game_id": ["g1", "g1", "g2", "g2"], "v": [1.0, 1.0, 3.0, 3.0]})
    out = paired_cluster_bootstrap_ci(x, "v", reps=500, seed=7)
    assert out["clusters"] == 2
    assert out["rows"] == 4
    assert out["cluster_unit"] == "actual_nfl_game_id"
    assert np.isclose(out["mean"], 2.0)


def test_disposition_is_strict_and_gate_complete():
    names = [
        "coverage_integrity", "script_blind_skill", "C_gt_B_material", "C_gt_A_material",
        "no_rmse_bias_tradeoff", "wr_room_translation_positive", "tail_mechanism_alignment",
    ]
    all_pass = {k: True for k in names}
    assert stage1_disposition(all_pass) == "QUALIFIED_PREDICTED_PASS_VOLUME_INCREMENT"
    for key in names:
        x = all_pass.copy()
        x[key] = False
        assert stage1_disposition(x) == "NO_ACTIONABLE_PREDICTED_PASS_VOLUME_INCREMENT"
