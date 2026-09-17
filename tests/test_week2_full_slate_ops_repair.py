from pathlib import Path

import pandas as pd
import pytest

from scripts.operations import validate_playerform_runtime_strict_prior_v1 as strict_prior
import scripts.run_team_form_context as team_context


TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC",
    "LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
]


def _write_playerform_audit_inputs(tmp_path: Path, *, log_week: int) -> None:
    pd.DataFrame([
        {"team": "BUF", "player_clean_key": "jamescook", "definitive_unavailable": 0},
    ]).to_csv(tmp_path / "availability.csv", index=False)
    pd.DataFrame([
        {"team": "BUF", "player_clean_key": "jamescook"},
    ]).to_csv(tmp_path / "form.csv", index=False)
    pd.DataFrame([
        {"player": "James Cook", "player_clean_key": "jamescook", "team": "BUF", "season": 2026, "week": log_week},
    ]).to_csv(tmp_path / "logs.csv", index=False)


def test_week2_strict_prior_allows_completed_week1(monkeypatch, tmp_path):
    _write_playerform_audit_inputs(tmp_path, log_week=1)
    monkeypatch.setattr(strict_prior, "AVAILABILITY", tmp_path / "availability.csv")
    monkeypatch.setattr(strict_prior, "PLAYER_FORM", tmp_path / "form.csv")
    monkeypatch.setattr(strict_prior, "GAME_LOGS", tmp_path / "logs.csv")

    result = strict_prior.validate(season=2026, target_week=2)
    assert result["same_season_prior_rows"] == 1
    assert result["illegal_current_or_future_rows"] == 0


def test_week2_strict_prior_rejects_week2_or_later(monkeypatch, tmp_path):
    _write_playerform_audit_inputs(tmp_path, log_week=2)
    monkeypatch.setattr(strict_prior, "AVAILABILITY", tmp_path / "availability.csv")
    monkeypatch.setattr(strict_prior, "PLAYER_FORM", tmp_path / "form.csv")
    monkeypatch.setattr(strict_prior, "GAME_LOGS", tmp_path / "logs.csv")

    with pytest.raises(RuntimeError, match="current/future target-week history"):
        strict_prior.validate(season=2026, target_week=2)


def _valid_core_teamform() -> pd.DataFrame:
    rows = []
    for team in TEAMS:
        rows.append({
            "team": team,
            "season": 2026,
            "def_pass_epa": -0.05,
            "def_rush_epa": -0.03,
            "def_sack_rate": 0.07,
            "pace": 29.0,
            "neutral_pace": 29.0,
            "proe": 0.01,
            "pass_rate_over_expected": 0.01,
            "rz_rate": 0.22,
            "ay_per_att": 7.5,
            "light_box_rate": 0.40,
            "heavy_box_rate": 0.20,
        })
    return pd.DataFrame(rows)


def test_teamform_compat_accepts_only_missing_legacy_coverage(monkeypatch, tmp_path):
    path = tmp_path / "team_form.csv"
    _valid_core_teamform().to_csv(path, index=False)
    monkeypatch.setattr(team_context, "TEAM_FORM_PATH", path)

    # No exception: core TeamForm is valid and legacy man/zone values are absent.
    team_context._validate_coverage_only_legacy_exit()


def test_teamform_compat_refuses_to_mask_noncoverage_failure(monkeypatch, tmp_path):
    path = tmp_path / "team_form.csv"
    df = _valid_core_teamform()
    df.loc[0, "def_pass_epa"] = pd.NA
    df.to_csv(path, index=False)
    monkeypatch.setattr(team_context, "TEAM_FORM_PATH", path)

    with pytest.raises(RuntimeError, match="Required team_form metrics missing"):
        team_context._validate_coverage_only_legacy_exit()
