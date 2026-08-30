import pandas as pd

import scripts.enrich_player_scoring_v2 as scoring


def test_scoring_enrichment_preserves_playerform_prior_current_columns(tmp_path, monkeypatch):
    identity = "gsis:00-0099000"
    form = pd.DataFrame([{
        "player": "Alpha WR", "player_clean_key": "alphawr", "player_identity_key": identity,
        "player_id": "00-0099000", "team": "IND", "season": 2026,
        "week": 3, "position": "WR", "role": "LWR", "prior_games": 17, "current_games": 2,
        "tgt_share": .30, "tgt_share_prior": .28, "tgt_share_current": .34,
        "rush_share": .01, "rush_share_prior": .01, "rush_share_current": .01,
        "route_rate": .89, "route_rate_prior": .88, "route_rate_current": .91,
        "yprr": 2.2, "yprr_prior": 2.1, "yprr_current": 2.4,
        "ypt": 9.3, "ypt_prior": 9.0, "ypt_current": 10.0,
        "ypc": 5.3, "ypc_prior": 5.0, "ypc_current": 6.0,
        "ypa": None, "ypa_prior": None, "ypa_current": None,
        "receptions_per_target": .67, "receptions_per_target_prior": .66, "receptions_per_target_current": .70,
    }])
    logs = pd.DataFrame([
        {
            "season": 2025, "week": 18, "player": "Alpha WR", "player_clean_key": "alphawr",
            "player_identity_key": identity, "player_id": "00-0099000", "team": "IND",
            "position": "WR", "identity_full_name_key": "alphawr", "identity_base_name_key": "alphawr",
        },
        {
            "season": 2026, "week": 1, "player": "Alpha WR", "player_clean_key": "alphawr",
            "player_identity_key": identity, "player_id": "00-0099000", "team": "IND",
            "position": "WR", "identity_full_name_key": "alphawr", "identity_base_name_key": "alphawr",
        },
        {
            "season": 2026, "week": 2, "player": "Alpha WR", "player_clean_key": "alphawr",
            "player_identity_key": identity, "player_id": "00-0099000", "team": "IND",
            "position": "WR", "identity_full_name_key": "alphawr", "identity_base_name_key": "alphawr",
        },
    ])
    form.to_csv(tmp_path / "player_form_consensus.csv", index=False)
    logs.to_csv(tmp_path / "player_game_logs.csv", index=False)

    prior_scoring = pd.DataFrame([{
        "team": "IND", "player_identity_key": identity, "rz_tgt_share": .25,
        "rz_carry_share": .00, "offensive_tds": 8, "season": 2025,
    }])
    current_scoring = pd.DataFrame([{
        "team": "IND", "player_identity_key": identity, "rz_tgt_share": .30,
        "rz_carry_share": .00, "offensive_tds": 1, "season": 2026,
    }])

    monkeypatch.setattr(scoring, "DATA", tmp_path)
    monkeypatch.setattr(
        scoring,
        "_season_scoring",
        lambda season, before_week=None, registry=None: (
            prior_scoring.copy() if season == 2025 else current_scoring.copy()
        ),
    )

    out = scoring.enrich(2026, 2025, 3)
    for col in (
        "tgt_share_prior", "tgt_share_current", "rush_share_prior", "rush_share_current",
        "ypt_prior", "ypt_current", "receptions_per_target_prior", "receptions_per_target_current",
    ):
        assert col in out.columns
    assert out.loc[0, "player_identity_key"] == identity
    assert out.loc[0, "tgt_share_prior"] == .28
    assert out.loc[0, "tgt_share_current"] == .34
    assert "rz_tgt_share_prior" not in out.columns
    assert "offensive_tds_current" not in out.columns