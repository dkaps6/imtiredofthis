import numpy as np
import pandas as pd

from scripts.backtest.feature_coverage_audit import feature_coverage, passing_summary, passing_trace
from scripts.backtest.component_predictions import build_actual_rows


def _predictions():
    return pd.DataFrame([
        {
            "season": 2025, "week": 4, "player": "QB One", "player_clean_key": "qbone",
            "team": "IND", "opponent": "TEN", "position": "QB", "market": "pass_yards",
            "actual": 220.0, "actual_opportunities": 31.0, "mc_proj": 260.0,
            "ctx_tgt_share_available": 0, "ctx_rush_share_available": 1, "ctx_ypa_available": 1,
            "ctx_success_rate_available": 1, "ctx_pace_available": 1, "ctx_plays_available": 1,
            "ctx_proe_available": 1, "ctx_pressure_available": 1, "ctx_explosive_available": 1,
            "ctx_def_epa_available": 1, "ctx_coverage_scheme_available": 0,
            "ctx_box_rates_available": 0, "ctx_wr_cb_matchup_available": 0,
            "ctx_injury_available": 0, "ctx_weather_available": 0,
            "mc_projected_plays": 65.0, "mc_pass_rate": .58, "mc_expected_pass_attempts": 37.7,
            "mc_base_ypa": 7.0, "mc_bayes_ypa": 7.1, "mc_rules_ypa": 6.9,
            "mc_pass_eff_mult": .97, "mc_off_pressure_allowed": .25,
            "mc_def_pressure_generated": .31, "mc_pressure_mismatch": .06,
            "rules_applied": 1, "rules_role": "",
        },
        {
            "season": 2025, "week": 4, "player": "WR One", "player_clean_key": "wrone",
            "team": "IND", "opponent": "TEN", "position": "WR", "market": "rec_yards",
            "actual": 80.0, "actual_opportunities": 9.0, "mc_proj": 72.0,
            "ctx_tgt_share_available": 1, "ctx_rush_share_available": 0, "ctx_ypa_available": 0,
            "ctx_success_rate_available": 1, "ctx_pace_available": 1, "ctx_plays_available": 1,
            "ctx_proe_available": 1, "ctx_pressure_available": 1, "ctx_explosive_available": 1,
            "ctx_def_epa_available": 1, "ctx_coverage_scheme_available": 0,
            "ctx_box_rates_available": 0, "ctx_wr_cb_matchup_available": 0,
            "ctx_injury_available": 0, "ctx_weather_available": 0,
        },
    ])


def test_feature_coverage_reports_missing_optional_layers():
    out = feature_coverage(_predictions())
    cov = out.set_index("feature")
    assert cov.loc["team_pace", "coverage"] == 1.0
    assert cov.loc["coverage_scheme", "coverage"] == 0.0
    assert cov.loc["wr_cb_matchup", "status"] == "missing_or_neutral"


def test_passing_trace_preserves_mc_inputs_and_errors():
    trace = passing_trace(_predictions())
    assert len(trace) == 1
    row = trace.iloc[0]
    assert row["mc_error"] == 40.0
    assert row["actual_pass_attempts"] == 31.0
    assert np.isclose(row["mc_expected_pass_attempts"], 37.7)
    summary = passing_summary(trace)
    assert int(summary.iloc[0]["n"]) == 1
    assert summary.iloc[0]["bias"] == 40.0


def test_actual_rows_record_market_specific_opportunities():
    logs = pd.DataFrame([
        {
            "season": 2025, "week": 3, "player": "QB One", "player_clean_key": "qbone", "team": "IND",
            "pass_yards": 250, "pass_att": 33, "rush_yards": 12, "rushes": 4,
            "rec_yards": 0, "targets": 0, "receptions": 0,
        }
    ])
    out = build_actual_rows(logs, 2025, 3).set_index("market")
    assert out.loc["pass_yards", "actual_opportunities"] == 33.0
    assert out.loc["rush_yards", "actual_opportunities"] == 4.0
    assert out.loc["rush_rec_yards", "actual_opportunities"] == 4.0
