from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.research.run_player_opportunity_volume_vs_share_decomposition_v1 as d


def test_actual_volume_semantics_match_simulator():
    qb = pd.Series({"position_family": "QB", "opportunity_type": "pass_attempts"})
    wr = pd.Series({"position_family": "WR", "opportunity_type": "targets"})
    rb_t = pd.Series({"position_family": "RB", "opportunity_type": "targets"})
    rb_c = pd.Series({"position_family": "RB", "opportunity_type": "carries"})
    assert d._actual_volume_column(qb) == "actual_team_official_pass_attempts"
    assert d._actual_volume_column(wr) == "actual_team_dropbacks"
    assert d._actual_volume_column(rb_t) == "actual_team_dropbacks"
    assert d._actual_volume_column(rb_c) == "actual_team_non_dropback_plays"


def test_summary_distinguishes_team_volume_from_player_share():
    g = pd.DataFrame({
        "actual_opportunities": [8.0, 2.0],
        "model_expected": [4.0, 4.0],
        "actual_volume_diagnostic": [5.0, 3.0],
        "actual_share_diagnostic": [7.5, 2.5],
        "predicted_team_opportunity_mean": [20.0, 20.0],
        "actual_team_volume": [25.0, 15.0],
        "final_player_probability": [0.20, 0.20],
        "actual_player_share": [8/25, 2/15],
    })
    s = d._summary(g)
    assert s["model_mae"] == pytest.approx(3.0)
    assert s["actual_volume_mae"] == pytest.approx(2.0)
    assert s["actual_share_mae"] == pytest.approx(0.5)
    assert s["mae_improvement_actual_share"] > s["mae_improvement_actual_volume"]


def test_run_preserves_full_oracle_identity_and_model_expectation(monkeypatch, tmp_path: Path):
    rows = pd.DataFrame([
        {
            "season": 2026, "week": 1, "team": "IND", "player": "WR One",
            "player_clean_key": "wrone", "position_family": "WR",
            "opportunity_type": "targets", "actual_opportunities": 8.0,
            "actual_opportunity_bin": "06_08",
            "predicted_team_opportunity_mean": 40.0,
            "final_player_probability": 0.10,
            "predicted_opportunities": 4.02,
            "expected_opportunities_from_probability": 4.0,
            "sportsbook_inputs_used_upstream": False,
        },
        {
            "season": 2026, "week": 1, "team": "IND", "player": "RB One",
            "player_clean_key": "rbone", "position_family": "RB",
            "opportunity_type": "carries", "actual_opportunities": 12.0,
            "actual_opportunity_bin": "09_14",
            "predicted_team_opportunity_mean": 24.0,
            "final_player_probability": 0.25,
            "predicted_opportunities": 6.03,
            "expected_opportunities_from_probability": 6.0,
            "sportsbook_inputs_used_upstream": False,
        },
        {
            "season": 2026, "week": 1, "team": "IND", "player": "QB One",
            "player_clean_key": "qbone", "position_family": "QB",
            "opportunity_type": "pass_attempts", "actual_opportunities": 30.0,
            "actual_opportunity_bin": "21_30",
            "predicted_team_opportunity_mean": 35.0,
            "final_player_probability": 0.90,
            "predicted_opportunities": 31.5,
            "expected_opportunities_from_probability": 31.5,
            "sportsbook_inputs_used_upstream": False,
        },
    ])
    p = tmp_path / "rows.csv"
    rows.to_csv(p, index=False)

    team = pd.DataFrame([{
        "season": 2026, "week": 1, "team": "IND",
        "actual_team_official_pass_attempts": 32.0,
        "actual_team_dropbacks": 45.0,
        "actual_team_non_dropback_plays": 22.0,
        "actual_team_offensive_plays": 67.0,
    }])
    monkeypatch.setattr(d, "build_actual_team_volumes", lambda: team)

    payload = d.run(rows_path=p, out_dir=tmp_path / "out")
    assert payload["rows"] == 3
    assert payload["max_full_identity_gap"] <= 1e-10
    out = pd.read_csv(tmp_path / "out/player_opportunity_volume_share_rows.csv")
    wr = out.loc[out.player.eq("WR One")].iloc[0]
    rb = out.loc[out.player.eq("RB One")].iloc[0]
    qb = out.loc[out.player.eq("QB One")].iloc[0]
    assert wr.actual_team_volume == pytest.approx(45.0)
    assert rb.actual_team_volume == pytest.approx(22.0)
    assert qb.actual_team_volume == pytest.approx(32.0)
    assert wr.full_identity == pytest.approx(8.0)
    assert rb.full_identity == pytest.approx(12.0)
    assert qb.full_identity == pytest.approx(30.0)


def test_run_fails_if_expected_probability_identity_drifts(monkeypatch, tmp_path: Path):
    rows = pd.DataFrame([{
        "season": 2026, "week": 1, "team": "IND", "player": "WR One",
        "player_clean_key": "wrone", "position_family": "WR",
        "opportunity_type": "targets", "actual_opportunities": 2.0,
        "actual_opportunity_bin": "01_02",
        "predicted_team_opportunity_mean": 40.0,
        "final_player_probability": 0.10,
        "predicted_opportunities": 4.0,
        "expected_opportunities_from_probability": 4.5,
        "sportsbook_inputs_used_upstream": False,
    }])
    p = tmp_path / "bad.csv"
    rows.to_csv(p, index=False)
    with pytest.raises(RuntimeError, match="deterministic expectation parity failed"):
        d.run(rows_path=p, out_dir=tmp_path / "out")
