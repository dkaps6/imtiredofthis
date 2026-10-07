from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.research.run_rb_receiving_share_state_coverage_v1 as audit


def test_normalize_room_preserves_missing_vs_real_zero():
    g = pd.DataFrame(
        {"x": [0.8, 0.2, np.nan]},
        index=[10, 11, 12],
    )
    out = audit._normalize_room(g, "x")
    assert out.loc[10] == pytest.approx(0.8)
    assert out.loc[11] == pytest.approx(0.2)
    # Missing player history contributes zero only after at least one real room
    # value exists; an all-missing room remains unscoreable.
    assert out.loc[12] == pytest.approx(0.0)

    all_missing = pd.DataFrame({"x": [np.nan, np.nan]})
    out2 = audit._normalize_room(all_missing, "x")
    assert out2.isna().all()


def test_raw_room_state_can_outperform_current_model_without_fit(monkeypatch, tmp_path: Path):
    parent = pd.DataFrame([
        {
            "season": 2026, "week": 1, "event_id": "G1", "team": "IND",
            "player": "Back A", "player_clean_key": "backa",
            "position_family": "RB", "opportunity_type": "targets",
            "actual_opportunities": 8.0, "final_player_probability": 0.10,
            "actual_player_share": 0.20, "sportsbook_inputs_used_upstream": False,
        },
        {
            "season": 2026, "week": 1, "event_id": "G1", "team": "IND",
            "player": "Back B", "player_clean_key": "backb",
            "position_family": "RB", "opportunity_type": "targets",
            "actual_opportunities": 2.0, "final_player_probability": 0.10,
            "actual_player_share": 0.05, "sportsbook_inputs_used_upstream": False,
        },
    ])
    rows = tmp_path / "rows.csv"
    parent.to_csv(rows, index=False)

    monkeypatch.setattr(audit, "identity_atlas", lambda *_: (pd.DataFrame(), pd.DataFrame()))

    def fake_snapshot(q, states, prev):
        vals = {
            "backa": {
                "prior_rb_room_share": 0.8,
                "last8_rb_room_share": 0.8,
                "prev_season_rb_room_share": 0.8,
                "same_team_prior_rb_room_share": 0.8,
                "prior_targets_pg": 4.0,
                "last8_targets_pg": 4.5,
                "prev_season_targets_pg": 4.0,
                "same_team_prior_targets_pg": 4.0,
                "prior_target_share": 0.12,
                "last8_target_share": 0.13,
                "prev_season_target_share": 0.12,
                "prior_games": 20.0,
                "same_team_prior_games": 20.0,
                "prev_season_games": 17.0,
            },
            "backb": {
                "prior_rb_room_share": 0.2,
                "last8_rb_room_share": 0.2,
                "prev_season_rb_room_share": 0.2,
                "same_team_prior_rb_room_share": 0.2,
                "prior_targets_pg": 1.0,
                "last8_targets_pg": 1.0,
                "prev_season_targets_pg": 1.0,
                "same_team_prior_targets_pg": 1.0,
                "prior_target_share": 0.03,
                "last8_target_share": 0.03,
                "prev_season_target_share": 0.03,
                "prior_games": 20.0,
                "same_team_prior_games": 20.0,
                "prev_season_games": 17.0,
            },
        }
        out = q.copy()
        for col in [
            *audit.ALL_FIELDS,
            "prior_games", "same_team_prior_games", "prev_season_games",
        ]:
            out[col] = [vals[k][col] for k in out["player_clean_key"]]
        return out

    monkeypatch.setattr(audit, "_snapshot_queries", fake_snapshot)
    payload = audit.run(rows_path=rows, out_dir=tmp_path / "out", repo_root=Path("."))

    assert payload["rows"] == 2
    assert payload["current_model_room_share_mae"] == pytest.approx(0.3)
    proxy = pd.read_csv(tmp_path / "out/rb_receiving_share_state_proxy_scoreboard.csv")
    prior = proxy.loc[proxy.field.eq("prior_rb_room_share")].iloc[0]
    assert prior.proxy_mae == pytest.approx(0.0)
    assert prior.mae_improvement_current_minus_proxy == pytest.approx(0.3)
    assert payload["parameters_fit"] == 0
    assert payload["sportsbook_inputs_used"] is False


def test_source_consumption_audit_separates_generic_path_from_specialists():
    x = audit._source_consumption_audit(Path("."))
    assert set(audit.ALL_FIELDS).issubset(set(x.field))
    # Rich RB receiving-identity fields belong to the specialist code, not the
    # generic Bayes/rules/entitlement path.
    prior_room = x.loc[x.field.eq("prior_rb_room_share")].iloc[0]
    assert not bool(prior_room.generic_bayesian_mentions)
    assert not bool(prior_room.generic_rules_mentions)
    assert not bool(prior_room.generic_entitlement_mentions)
    assert bool(prior_room.rb_r26_mentions)
