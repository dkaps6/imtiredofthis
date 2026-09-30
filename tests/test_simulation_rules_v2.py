import numpy as np
import pandas as pd

from scripts.modeling.contracts import PlayerContext, TeamContext
from scripts.modeling import simulation_rules
from scripts.simulation_v2 import lookup, simulate


def _team(team, **kw):
    base = dict(
        team=team, season=2026, success_rate_off=0.50, success_rate_def=0.47,
        pressure_rate_generated=0.30, pressure_rate_allowed=0.25,
        neutral_pace=27.0, neutral_pace_last5=27.0, sec_per_play_last5=27.0,
        plays_est=64.0, proe=0.0, explosive_play_rate_allowed=0.10,
        coverage_man_rate=0.30, coverage_zone_rate=0.50, middle_open_rate=0.30,
        light_box_rate=0.30, heavy_box_rate=0.30, def_pass_epa=0.0, def_rush_epa=0.0,
    )
    base.update(kw)
    return TeamContext(**base)


def test_rule_adapter_applies_game_script_role_and_pressure(monkeypatch):
    ind = _team("IND", success_rate_off=0.55, pressure_rate_allowed=0.20, proe=0.05)
    hou = _team("HOU", success_rate_def=0.43, pressure_rate_generated=0.35,
                coverage_man_rate=0.55, coverage_zone_rate=0.30)
    wr1 = PlayerContext("Alpha Receiver", "IND", "HOU", 2026, 1, "LWR", "WR1", "g1",
                        {"tgt_share": 0.30, "ypt": 9.0}, ind, hou)
    wr2 = PlayerContext("Beta Receiver", "IND", "HOU", 2026, 1, "RWR", "WR1", "g1",
                        {"tgt_share": 0.20, "ypt": 8.0}, ind, hou)
    monkeypatch.setattr(simulation_rules, "load_model_contexts", lambda: ({"IND": ind, "HOU": hou}, [wr1, wr2]))

    metrics = pd.DataFrame([
        {"player": "Alpha Receiver", "team": "IND", "opponent": "HOU", "position": "WR", "tgt_share": 0.30, "ypt": 9.0},
        {"player": "Beta Receiver", "team": "IND", "opponent": "HOU", "position": "WR", "tgt_share": 0.20, "ypt": 8.0},
    ])
    out = simulation_rules.apply_rules_to_metrics(metrics)
    assert out["rules_applied"].tolist() == [1, 1]
    assert out.iloc[0]["rules_role"] == "WR1"
    assert out.iloc[1]["rules_role"] == "WR1_5"
    assert out.iloc[0]["rules_pass_eff_mult"] < 1.0
    assert out.iloc[1]["rules_tgt_share"] > 0.20
    assert out.iloc[0]["rules_plays_est"] >= 50.0


def test_alpha_injury_redistribution_preserves_legacy_60_30_10(monkeypatch):
    ind = _team("IND")
    hou = _team("HOU")
    players = [
        PlayerContext("Alpha", "IND", "HOU", 2026, 1, "LWR", "WR1", "g1", {"tgt_share": 0.30, "injury_status": "OUT"}, ind, hou),
        PlayerContext("WR2", "IND", "HOU", 2026, 1, "RWR", "WR1", "g1", {"tgt_share": 0.20}, ind, hou),
        PlayerContext("Slot", "IND", "HOU", 2026, 1, "SWR", "SWR1", "g1", {"tgt_share": 0.20}, ind, hou),
        PlayerContext("Back", "IND", "HOU", 2026, 1, "RB", "RB1", "g1", {"tgt_share": 0.10}, ind, hou),
    ]
    monkeypatch.setattr(simulation_rules, "load_model_contexts", lambda: ({"IND": ind, "HOU": hou}, players))
    metrics = pd.DataFrame([{"player": p.player, "team": "IND", "opponent": "HOU", "position": p.position, "tgt_share": p.features["tgt_share"]} for p in players])
    out = simulation_rules.apply_rules_to_metrics(metrics).set_index("player")
    assert np.isclose(out.loc["Alpha", "rules_tgt_share"], 0.15)
    assert out.loc["WR2", "rules_tgt_share"] > 0.20
    assert out.loc["Slot", "rules_tgt_share"] > 0.20
    assert out.loc["Back", "rules_tgt_share"] > 0.10
    assert int(out["rules_injury_redistribution"].sum()) == 4


def test_simulation_prefers_rule_adjusted_inputs():
    base = pd.DataFrame([{
        "event_id": "g1", "player": "QB Test", "player_clean_key": "qbtest",
        "team": "IND", "opponent": "HOU", "position": "QB", "role": "QB1",
        "plays_est": 60.0, "proe": 0.0, "ypa": 7.0, "rush_share": 0.0, "tgt_share": 0.0,
    }])
    adjusted = base.copy()
    adjusted["rules_plays_est"] = 72.0
    adjusted["rules_pass_rate"] = 0.68
    adjusted["rules_ypa"] = 8.0
    adjusted["rules_volatility_mult"] = 1.0
    a = simulate(base, iterations=4000, seed=7)
    b = simulate(adjusted, iterations=4000, seed=7)
    base_yards = lookup(a, base.iloc[0], "pass_yards")
    adj_yards = lookup(b, adjusted.iloc[0], "pass_yards")
    assert float(np.mean(adj_yards)) > float(np.mean(base_yards))


def test_full_bayesian_authority_removes_sportsbook_shape_from_injury_redistribution(monkeypatch):
    ind = _team("IND")
    hou = _team("HOU")
    players = [
        PlayerContext("Alpha", "IND", "HOU", 2026, 3, "LWR", "WR1", "g1", {"tgt_share": 0.30, "injury_status": "OUT"}, ind, hou),
        PlayerContext("WR2", "IND", "HOU", 2026, 3, "RWR", "WR2", "g1", {"tgt_share": 0.20}, ind, hou),
        PlayerContext("Slot", "IND", "HOU", 2026, 3, "SWR", "WR3", "g1", {"tgt_share": 0.18}, ind, hou),
        PlayerContext("Tight End", "IND", "HOU", 2026, 3, "TE", "TE1", "g1", {"tgt_share": 0.12}, ind, hou),
        PlayerContext("Back", "IND", "HOU", 2026, 3, "RB", "RB1", "g1", {"tgt_share": 0.10}, ind, hou),
    ]
    monkeypatch.setattr(simulation_rules, "load_model_contexts", lambda: ({"IND": ind, "HOU": hou}, players))

    bayes = pd.DataFrame([
        {"player": "Alpha", "player_clean_key": "alpha", "team": "IND", "bayes_tgt_share": 0.24},
        {"player": "WR2", "player_clean_key": "wr2", "team": "IND", "bayes_tgt_share": 0.21},
        {"player": "Slot", "player_clean_key": "slot", "team": "IND", "bayes_tgt_share": 0.17},
        {"player": "Tight End", "player_clean_key": "tightend", "team": "IND", "bayes_tgt_share": 0.11},
        {"player": "Back", "player_clean_key": "back", "team": "IND", "bayes_tgt_share": 0.09},
    ])
    full = pd.DataFrame([
        {"player": r["player"], "player_clean_key": r["player_clean_key"], "team": "IND", "opponent": "HOU",
         "position": next(p.position for p in players if p.player == r["player"]), "bayes_tgt_share": r["bayes_tgt_share"]}
        for r in bayes.to_dict("records")
    ])
    partial = full.loc[~full["player"].eq("Alpha")].copy()

    full_out = simulation_rules.apply_rules_to_metrics(full, bayes_baseline=bayes).set_index("player")
    partial_out = simulation_rules.apply_rules_to_metrics(partial, bayes_baseline=bayes).set_index("player")
    recipients = ["WR2", "Slot", "Tight End", "Back"]
    assert np.allclose(
        full_out.loc[recipients, "rules_tgt_share"].to_numpy(float),
        partial_out.loc[recipients, "rules_tgt_share"].to_numpy(float),
        rtol=0,
        atol=1e-12,
    )

    legacy_partial = simulation_rules.apply_rules_to_metrics(partial).set_index("player")
    assert not np.allclose(
        partial_out.loc[recipients, "rules_tgt_share"].to_numpy(float),
        legacy_partial.loc[recipients, "rules_tgt_share"].to_numpy(float),
        rtol=0,
        atol=1e-12,
    )



def test_wr_cb_assignment_flags_no_longer_modify_rules_ypt_or_target_share(monkeypatch):
    ind = _team("IND")
    hou = _team("HOU", coverage_man_rate=0.30, coverage_zone_rate=0.65)

    def run(primary_cb: str):
        wr = PlayerContext(
            "Alpha Receiver", "IND", "HOU", 2026, 4, "LWR", "WR1", "g1",
            {
                "tgt_share": 0.25,
                "ypt": 10.0,
                "matchup_available": 1,
                "team_coverage_available": 1,
                "primary_cb": primary_cb,
            },
            ind, hou,
        )
        monkeypatch.setattr(
            simulation_rules,
            "load_model_contexts",
            lambda: ({"IND": ind, "HOU": hou}, [wr]),
        )
        metrics = pd.DataFrame([{
            "player": "Alpha Receiver",
            "team": "IND",
            "opponent": "HOU",
            "position": "WR",
            "tgt_share": 0.25,
            "ypt": 10.0,
        }])
        return simulation_rules.apply_rules_to_metrics(metrics).iloc[0]

    shadow = run("Corner One")
    no_shadow = run("")

    assert shadow["rules_tgt_share"] == no_shadow["rules_tgt_share"]
    assert shadow["rules_ypt"] == no_shadow["rules_ypt"]
