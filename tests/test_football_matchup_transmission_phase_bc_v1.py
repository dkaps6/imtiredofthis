import numpy as np
import pandas as pd

import scripts.research.audit_football_matchup_transmission_phase_bc_v1 as m


def test_source_parity_and_closed_family_guards_are_frozen():
    sources = m._source_matrix().set_index("feature")
    assert sources.loc["def_rush_epa", "parity_status"] == "EXACT_LIVE_SEMANTICS"
    assert sources.loc["dl_stuff_rate", "parity_status"] == "SOURCE_BLOCKED"
    assert sources.loc["dl_ybc_per_rush", "parity_status"] == "SOURCE_BLOCKED"
    assert sources.loc["outside_ypt_allowed", "parity_status"] == "SOURCE_BLOCKED"
    assert sources.loc["slot_ypt_allowed", "parity_status"] == "SOURCE_BLOCKED"
    assert sources.loc["wr_ypt_allowed", "parity_status"] == "PARITY_UNPROVEN_DIAGNOSTIC_ONLY"

    specs = m._feature_specs()
    rb_phase_c = [
        s for s in specs
        if s.phase == "C" and s.cohort in {"RB_RUSH", "RB_RUSH_REC"} and s.feature == "def_rush_epa"
    ]
    assert rb_phase_c
    assert all(s.closed_family == "M95A_M95B_CLOSED_OVERLAP" for s in rb_phase_c)

    qb_specs = [s for s in specs if s.cohort == "QB_PASS_CONTROL"]
    assert qb_specs
    assert all(s.closed_family == "M56_M83_CONTROL_ONLY" for s in qb_specs)


def test_player_opportunity_blend_excludes_target_week():
    logs = pd.DataFrame(
        [
            {
                "season": 2023,
                "week": 18,
                "player_identity_key": "p1",
                "targets": 20.0,
                "team_targets": 100.0,
                "rushes": 30.0,
                "team_rushes": 100.0,
            },
            {
                "season": 2024,
                "week": 1,
                "player_identity_key": "p1",
                "targets": 40.0,
                "team_targets": 100.0,
                "rushes": 50.0,
                "team_rushes": 100.0,
            },
            {
                # Target-week usage must not enter the Week-2 feature.
                "season": 2024,
                "week": 2,
                "player_identity_key": "p1",
                "targets": 99.0,
                "team_targets": 100.0,
                "rushes": 99.0,
                "team_rushes": 100.0,
            },
        ]
    )
    out = m.build_player_opportunity_pregame(logs)
    w2 = out.loc[
        out["season"].eq(2024)
        & out["week"].eq(2)
        & out["player_identity_key"].eq("p1")
    ].iloc[0]

    # Production-style current weight = 1 / (1 + 4) = 0.2.
    # target share = .8*.20 + .2*.40 = .24
    # rush share   = .8*.30 + .2*.50 = .34
    assert np.isclose(float(w2["target_share"]), 0.24)
    assert np.isclose(float(w2["rush_share"]), 0.34)
    assert int(w2["current_games"]) == 1


def test_exact_rush_epa_uses_only_completed_prior_weeks(monkeypatch):
    pbp = pd.DataFrame(
        [
            {"season_type": "REG", "week": 1, "defteam": "NO", "epa": 0.10, "rush": True},
            {"season_type": "REG", "week": 1, "defteam": "NO", "epa": 0.30, "rush": True},
            # Target week is intentionally extreme and must not enter Week-2 pregame context.
            {"season_type": "REG", "week": 2, "defteam": "NO", "epa": 10.0, "rush": True},
            {"season_type": "REG", "week": 1, "defteam": "ATL", "epa": -0.20, "rush": True},
        ]
    )
    monkeypatch.setattr(m, "get_pbp", lambda season, min_rows=1: pbp.copy())

    out = m.build_exact_rush_epa_pregame([2024])
    no_w2 = out.loc[
        out["season"].eq(2024)
        & out["week"].eq(2)
        & out["team"].eq("NO")
    ].iloc[0]
    assert np.isclose(float(no_w2["def_rush_epa"]), 0.20)
    assert int(no_w2["def_rush_epa_plays"]) == 2


def test_dual_cluster_support_requires_positive_signal():
    rows = []
    for game in range(60):
        for j in range(4):
            value = float(game * 4 + j)
            rows.append(
                {
                    "season": 2024,
                    "residual": value,
                    "weakness_z": value,
                    "game_id": f"g{game}",
                    "player_identity_key": f"p{(game * 4 + j) % 40}",
                }
            )
    q = pd.DataFrame(rows)
    rec = m.evaluate_cell(q, "weakness_z", 2024, seed_offset=7)
    assert rec["support"] is True
    assert rec["rho"] > 0.99
    assert rec["game_ci_low"] > 0
    assert rec["player_ci_low"] > 0
    assert rec["directional_cluster_support"] is True


def test_candidate_gate_cannot_promote_closed_or_parity_unproven_signal(monkeypatch):
    def fake_eval(q, signal_col, season, seed_offset):
        return {
            "season": season,
            "rows": 500,
            "games": 100,
            "players": 100,
            "support": True,
            "rho": 0.2,
            "game_ci_low": 0.05,
            "game_ci_high": 0.35,
            "player_ci_low": 0.03,
            "player_ci_high": 0.34,
            "directional_cluster_support": True,
        }

    monkeypatch.setattr(m, "evaluate_cell", fake_eval)

    base = pd.DataFrame(
        [
            {
                "season": season,
                "week": 2,
                "team": "ATL",
                "opponent": "NO",
                "residual": 1.0,
                "game_id": f"{season}_g",
                "player_identity_key": "p1",
            }
            for season in (2024, 2025)
        ]
    )
    team_features = pd.DataFrame(
        [
            {
                "season": season,
                "week": 2,
                "team": "ATL",
                "opponent": "NO",
                "def_rush_epa": 0.1,
                "def_rush_epa__z": 1.0,
                "def_wr_ypt_allowed": 8.0,
                "def_wr_ypt_allowed__z": 1.0,
            }
            for season in (2024, 2025)
        ]
    )
    opportunity = pd.DataFrame(
        [
            {
                "season": season,
                "week": 2,
                "player_identity_key": "p1",
                "rush_share": 0.5,
                "target_share": 0.3,
            }
            for season in (2024, 2025)
        ]
    )
    specs = [
        m.FeatureSpec(
            "RB_RUSH",
            "def_rush_epa",
            "run_defense",
            +1,
            "C",
            "EXACT_LIVE_SEMANTICS",
            "M95A_M95B_CLOSED_OVERLAP",
            "rush_share",
        ),
        m.FeatureSpec(
            "WR_REC",
            "def_wr_ypt_allowed",
            "position_receiving_defense",
            +1,
            "C",
            "PARITY_UNPROVEN_DIAGNOSTIC_ONLY",
            "",
            "target_share",
        ),
    ]
    _, summary = m.evaluate_specs(
        {"RB_RUSH": base.copy(), "WR_REC": base.copy()},
        team_features,
        opportunity,
        specs,
    )
    rb = summary.loc[summary["cohort"].eq("RB_RUSH")].iloc[0]
    wr = summary.loc[summary["cohort"].eq("WR_REC")].iloc[0]
    assert bool(rb["replicated"]) is True
    assert bool(rb["integration_candidate_eligible"]) is False
    assert rb["disposition"] == "REPLICATED_BUT_CLOSED_PRIOR_FAMILY"
    assert bool(wr["replicated"]) is True
    assert bool(wr["integration_candidate_eligible"]) is False
    assert wr["disposition"] == "REPLICATED_DIAGNOSTIC_SOURCE_PARITY_BLOCKED"
