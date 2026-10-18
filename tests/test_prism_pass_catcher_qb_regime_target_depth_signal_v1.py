import pandas as pd
import scripts.research.evaluate_prism_pass_catcher_qb_regime_target_depth_signal_v1 as m


def test_bool_parser_handles_csv_strings():
    x = pd.Series(["True", "False", "1", "0"])
    assert m.bools(x).tolist() == [True, False, True, False]


def test_metrics_prefers_closer_candidate():
    x = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "teamweek": ["t1", "t2", "t3"],
        "actual_target_depth": [10.0, 12.0, 14.0],
        "baseline_target_depth": [5.0, 7.0, 9.0],
        "candidate_target_depth": [9.0, 11.0, 13.0],
        "baseline_abs_error": [5.0, 5.0, 5.0],
        "candidate_abs_error": [1.0, 1.0, 1.0],
        "error_delta": [-4.0, -4.0, -4.0],
        "actual_finite_air_targets": [3, 4, 5],
    })
    out = m.metrics(x)
    assert out["candidate_mae"] < out["baseline_mae"]
    assert out["candidate_rmse"] < out["baseline_rmse"]
    assert out["candidate_closer_rate"] == 1.0
    assert out["median_equal_player_mae_delta"] < 0


def test_cluster_bootstrap_negative_delta_stays_negative():
    x = pd.DataFrame({
        "cluster": ["a", "a", "b", "b", "c", "c"],
        "error_delta": [-1.0, -2.0, -1.5, -2.5, -0.5, -1.0],
    })
    lo, hi = m.bootstrap_cluster_ci(x, "cluster", reps=200, seed=1)
    assert lo < 0
    assert hi < 0
