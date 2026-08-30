import numpy as np
import pandas as pd

from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
import scripts.run_team_form_context as team_ctx


def test_promoted_qb_ensemble_weights_are_committed_and_non_mc_only():
    weights = load_weights()
    qb = weights.loc[weights["market"].astype(str).str.lower().eq("pass_yards")]
    assert len(qb) == 1
    row = qb.iloc[0]
    vals = np.array([row.mc_weight, row.ml_weight, row.state_weight], dtype=float)
    assert np.all(vals > 0)
    assert np.isclose(vals.sum(), 1.0)
    assert int(row.calibration_rows) == 474

    frame = pd.DataFrame([{
        "market": "pass_yards",
        "mc_proj": 260.0,
        "ml_proj": 250.0,
        "state_proj": 240.0,
    }])
    out = apply_ensemble(frame, weights=weights).iloc[0]
    assert out.ensemble_status == "calibrated"
    assert out.ensemble_method == "promoted_nonnegative_oos_linear_blend_v2"
    assert out.ensemble_weight_mc > 0
    assert out.ensemble_weight_ml > 0
    assert out.ensemble_weight_state > 0


def test_runtime_team_context_repair_uses_only_prior_weeks_of_active_season(tmp_path, monkeypatch):
    team_form = tmp_path / "team_form.csv"
    pd.DataFrame({"team": ["BUF", "NYJ"], "season": [2026, 2026]}).to_csv(team_form, index=False)

    # Week 1 is eligible for a Week 2 target. Week 2 rows intentionally reverse
    # the signals and must not contaminate the pregame context.
    pbp = pd.DataFrame([
        {"season_type": "REG", "week": 1, "posteam": "BUF", "defteam": "NYJ", "play_type": "pass", "epa": 0.5, "yards_gained": 20},
        {"season_type": "REG", "week": 1, "posteam": "NYJ", "defteam": "BUF", "play_type": "run", "epa": -0.5, "yards_gained": 5},
        {"season_type": "REG", "week": 2, "posteam": "BUF", "defteam": "NYJ", "play_type": "pass", "epa": -2.0, "yards_gained": 0},
        {"season_type": "REG", "week": 2, "posteam": "NYJ", "defteam": "BUF", "play_type": "run", "epa": 2.0, "yards_gained": 40},
    ])

    monkeypatch.setattr(team_ctx, "TEAM_FORM_PATH", team_form)
    monkeypatch.setattr(team_ctx.make_team_form, "load_pbp", lambda season: pbp.copy())

    state = {"pbp_feature_season": 2026, "used_prior": False}
    team_ctx._repair_success_explosive_context(2026, 2, state)
    got = pd.read_csv(team_form).set_index("team")

    assert np.isclose(got.loc["BUF", "success_rate_off"], 1.0)
    assert np.isclose(got.loc["NYJ", "success_rate_off"], 0.0)
    assert np.isclose(got.loc["BUF", "success_rate_def"], 0.0)
    assert np.isclose(got.loc["NYJ", "success_rate_def"], 1.0)
    assert np.isclose(got.loc["BUF", "explosive_play_rate_allowed"], 0.0)
    assert np.isclose(got.loc["NYJ", "explosive_play_rate_allowed"], 1.0)
    assert got["success_explosive_source_season"].eq(2026).all()
    assert got["success_explosive_target_week"].eq(2).all()
