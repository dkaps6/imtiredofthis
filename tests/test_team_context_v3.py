import numpy as np
import pandas as pd
import pytest

from scripts._opponent_map import CANON_TEAM_CODES
from scripts.modeling.context_bridge_v3 import build_team_contexts
from scripts.team_context_v3 import PROMOTED_FIELDS, build_team_context_v3


def _fixtures(*, season=2026, week=1, current_games=0, prior_games=8):
    teams = sorted(CANON_TEAM_CODES)
    tf_rows = []
    qb_rows = []
    cov_rows = []
    for i, team in enumerate(teams):
        tf_rows.append({
            "team": team,
            "season": season,
            "success_rate_off": 0.46 + (i % 5) * 0.005,
            "success_rate_def": 0.44 + (i % 4) * 0.005,
            "success_rate_diff": 0.02,
            "explosive_play_rate_allowed": 0.10,
            "pbp_feature_source": "prior_2025_pbp",
            "pbp_feature_season": 2025,
            "pbp_prior_used": 1,
            "proe": 99.0,
            "neutral_pace": 99.0,
            "def_rush_epa": -0.03,
            "neutral_pace_last5": 28.5,
            "seconds_per_play_last5": 28.5,
            "light_box_rate": 0.55,
            "heavy_box_rate": 0.22,
            "middle_open_rate": 0.48,
            "rz_rate": 0.15,
            "ay_per_att": 7.5,
        })
        q = {
            "team": team,
            "season": season,
            "week": week,
            "qb_context_history_games": current_games + prior_games,
            "qb_context_current_games": current_games,
            "qb_context_prior_games": prior_games,
            "qb_context_latest_season": 2025 if current_games == 0 else season,
            "qb_context_latest_week": 18 if current_games == 0 else max(1, week - 1),
            "qb_context_source_seasons": "2025" if current_games == 0 else "2025,2026",
        }
        promoted_values = {
            "true_proe": 0.03,
            "neutral_pace_true": 29.2,
            "pressure_rate_allowed": 0.24,
            "pressure_rate_generated": 0.31,
            "hit_sack_pressure_rate_allowed": 0.24,
            "hit_sack_pressure_rate_generated": 0.31,
            "pass_attempts_per_dropback": 0.90,
            "pass_rate_off": 0.58,
            "pass_rate_faced": 0.57,
            "def_pass_epa_allowed": 0.04,
            "def_pass_success_allowed": 0.49,
            "def_ypa_allowed": 7.1,
            "off_ypa": 7.3,
            "off_pass_epa": 0.08,
            "plays_est": 64.0,
        }
        q.update(promoted_values)
        qb_rows.append(q)
        cov_rows.append({"team": team, "man_rate": 0.35, "zone_rate": 0.65})
    return pd.DataFrame(tf_rows), pd.DataFrame(qb_rows), pd.DataFrame(cov_rows)


def test_team_context_v3_is_32_team_promoted_authority_with_provenance():
    tf, qb, cov = _fixtures()
    context, provenance = build_team_context_v3(2026, 1, team_form=tf, qb_context=qb, coverage=cov)

    assert len(context) == 32
    assert context["team"].nunique() == 32
    assert context["team_context_version"].eq("TEAM_CONTEXT_V3").all()
    assert np.isclose(context.loc[0, "true_proe"], 0.03)
    assert np.isclose(context.loc[0, "proe"], 0.03)  # promoted value overrides legacy 99.0
    assert np.isclose(context.loc[0, "neutral_pace"], 29.2)
    assert np.isclose(context.loc[0, "success_rate_off"], tf.sort_values("team").iloc[0]["success_rate_off"])
    assert context["success_rate_off"].between(0, 1).all()
    assert context["success_rate_def"].between(0, 1).all()
    assert context["coverage_man_rate"].eq(0.35).all()
    assert context["coverage_zone_rate"].eq(0.65).all()
    assert context["context_freshness_state"].eq("prior_only_preseason").all()

    for field in PROMOTED_FIELDS:
        p = provenance.loc[provenance["feature"].eq(field)]
        assert len(p) == 32
        assert p["source"].eq("M89_M90_PROMOTED_ROLLING_8").all()
        assert p["history_games"].eq(8).all()

    success_prov = provenance.loc[provenance["feature"].eq("success_rate_off")]
    assert success_prov["source_seasons"].eq("2025").all()
    assert success_prov["freshness_state"].eq("guarded_prior_pbp").all()


def test_team_context_v3_transitions_to_current_history_without_same_week_leakage_metadata():
    tf, qb, cov = _fixtures(week=5, current_games=4, prior_games=4)
    context, provenance = build_team_context_v3(2026, 5, team_form=tf, qb_context=qb, coverage=cov)
    assert context["context_current_games"].eq(4).all()
    assert context["context_prior_games"].eq(4).all()
    assert context["context_latest_week"].lt(5).all()
    assert context["context_freshness_state"].eq("current_season_history").all()
    promoted = provenance.loc[provenance["source"].eq("M89_M90_PROMOTED_ROLLING_8")]
    assert promoted["source_seasons"].eq("2025,2026").all()


def test_team_context_v3_fails_bad_history_accounting_and_bad_attempt_conversion():
    tf, qb, cov = _fixtures()
    qb.loc[0, "qb_context_history_games"] = 7
    with pytest.raises(RuntimeError, match="history accounting"):
        build_team_context_v3(2026, 1, team_form=tf, qb_context=qb, coverage=cov)

    tf, qb, cov = _fixtures()
    qb.loc[0, "pass_attempts_per_dropback"] = 1.03
    with pytest.raises(RuntimeError, match="attempt conversion"):
        build_team_context_v3(2026, 1, team_form=tf, qb_context=qb, coverage=cov)


def test_context_bridge_consumes_embedded_team_context_v3_fields():
    frame = pd.DataFrame([
        {
            "team": "LA", "season": 2026, "team_context_version": "TEAM_CONTEXT_V3",
            "success_rate_off": 0.51, "success_rate_def": 0.47,
            "hit_sack_pressure_rate_generated": 0.34,
            "hit_sack_pressure_rate_allowed": 0.26,
            "neutral_pace_true": 28.0, "plays_est": 64.0, "true_proe": 0.03,
            "coverage_man_rate": 0.41, "coverage_zone_rate": 0.59,
            "def_pass_epa_allowed": 0.04,
        },
        {
            "team": "SF", "season": 2026, "team_context_version": "TEAM_CONTEXT_V3",
            "success_rate_off": 0.50, "success_rate_def": 0.45,
            "hit_sack_pressure_rate_generated": 0.31,
            "hit_sack_pressure_rate_allowed": 0.25,
            "neutral_pace_true": 29.0, "plays_est": 63.0, "true_proe": 0.01,
            "coverage_man_rate": 0.35, "coverage_zone_rate": 0.65,
            "def_pass_epa_allowed": 0.02,
        },
    ])
    teams = build_team_contexts(frame)
    assert set(teams) == {"LAR", "SF"}
    assert teams["LAR"].pressure_rate_generated == 0.34
    assert teams["LAR"].coverage_man_rate == 0.41
    assert teams["SF"].coverage_zone_rate == 0.65
    assert teams["LAR"].proe == 0.03
