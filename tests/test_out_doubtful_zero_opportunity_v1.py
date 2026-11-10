"""A player the league has ruled OUT must receive zero modeled opportunity.

Evidence: PRISM_D1_W9_HOLDOUT_RESULTS_V1. On 2026 weeks 2-8, OUT retention is
0.0000 over 364 cases with 0.0% played and DOUBTFUL 0.0000 over 23 cases with
0.0% played. With the previous 0.50 multiplier production projected a mean
17.74 receiving and 8.55 rushing yards for 24 players publicly ruled out, all of
whom recorded 0.0.

Promoted as a correctness repair, not an edge claim: its measured betting impact
was indistinguishable from noise (12/18 disagreements, p=0.238, and only 2 of
those 18 involved a designated player).
"""
import numpy as np
import pandas as pd
import pytest

from scripts.modeling import simulation_rules
from scripts.modeling.simulation_rules import OUT_DOUBTFUL_OPPORTUNITY_RETENTION
from tests.test_simulation_rules_v2 import _team
from scripts.modeling.context_bridge import PlayerContext


TEAM, OPP = "IND", "HOU"


def _p(name, pos, role, feats):
    """PlayerContext is a frozen dataclass, so contexts are passed at build time."""
    return PlayerContext(name, TEAM, OPP, 2026, 9, pos, role, "g1", feats,
                         _team(TEAM), _team(OPP))


def _run(players, monkeypatch):
    t, o = _team(TEAM), _team(OPP)
    monkeypatch.setattr(simulation_rules, "load_model_contexts",
                        lambda: ({TEAM: t, OPP: o}, players))
    metrics = pd.DataFrame([{
        "player": p.player, "team": TEAM, "opponent": OPP, "position": p.position,
        "tgt_share": p.features.get("tgt_share", 0.0),
        "rush_share": p.features.get("rush_share", 0.0),
    } for p in players])
    return simulation_rules.apply_rules_to_metrics(metrics).set_index("player")


def test_retention_constant_is_zero():
    assert OUT_DOUBTFUL_OPPORTUNITY_RETENTION == 0.0


@pytest.mark.parametrize("status", ["Out", "OUT", "out", "Doubtful", "IR", "PUP"])
def test_out_player_receives_zero_target_and_rush_share(monkeypatch, status):
    """The core promotion. A non-alpha designated player gets nothing."""
    players = [
        _p("Alpha", "LWR", "WR1", {"tgt_share": 0.28}),
        _p("Hurt Guy", "RWR", "WR2", {"tgt_share": 0.20, "rush_share": 0.15,
                                      "injury_status": status}),
        _p("Healthy", "SWR", "WR3", {"tgt_share": 0.18}),
    ]
    out = _run(players, monkeypatch)
    assert out.loc["Hurt Guy", "rules_tgt_share"] == pytest.approx(0.0), status
    assert out.loc["Hurt Guy", "rules_rush_share"] == pytest.approx(0.0), status


def test_designation_field_also_triggers_it(monkeypatch):
    """context_bridge maps injuries.csv `designation` as well as `status`."""
    players = [
        _p("Alpha", "LWR", "WR1", {"tgt_share": 0.28}),
        _p("Hurt Guy", "RWR", "WR2", {"tgt_share": 0.20, "injury_designation": "Out"}),
    ]
    out = _run(players, monkeypatch)
    assert out.loc["Hurt Guy", "rules_tgt_share"] == pytest.approx(0.0)


def test_healthy_players_keep_their_opportunity(monkeypatch):
    players = [
        _p("Alpha", "LWR", "WR1", {"tgt_share": 0.28}),
        _p("Hurt Guy", "RWR", "WR2", {"tgt_share": 0.20, "injury_status": "Out"}),
        _p("Healthy", "SWR", "WR3", {"tgt_share": 0.18}),
    ]
    out = _run(players, monkeypatch)
    assert out.loc["Healthy", "rules_tgt_share"] > 0.0
    assert out.loc["Alpha", "rules_tgt_share"] > 0.0


def test_questionable_is_deliberately_untouched(monkeypatch):
    """QUESTIONABLE retention is NOT promoted: n=21 and club usage is inconsistent.

    If this starts failing because someone added QUESTIONABLE to
    ``_injury_limited``, that is a promotion and needs its own preregistration.
    """
    players = [
        _p("Alpha", "LWR", "WR1", {"tgt_share": 0.28}),
        _p("Maybe", "RWR", "WR2", {"tgt_share": 0.20, "injury_status": "Questionable"}),
    ]
    out = _run(players, monkeypatch)
    assert out.loc["Maybe", "rules_tgt_share"] > 0.0


def test_no_designation_is_untouched(monkeypatch):
    players = [
        _p("Alpha", "LWR", "WR1", {"tgt_share": 0.28}),
        _p("Fine", "RWR", "WR2", {"tgt_share": 0.20}),
    ]
    out = _run(players, monkeypatch)
    assert out.loc["Fine", "rules_tgt_share"] > 0.0


def test_out_alpha_still_uses_the_legacy_redistribution_rule(monkeypatch):
    """KNOWN INCONSISTENCY, deliberately not changed here.

    An OUT **WR1 alpha** goes through ``_injury_target_overrides`` instead, which
    keeps half his share and redistributes the other half 60/30/10. So an OUT
    WR1 retains opportunity while an OUT WR2 now gets zero.

    That legacy rule is a separately promoted behaviour with its own test
    (``test_alpha_injury_redistribution_preserves_legacy_60_30_10``) and it
    REDISTRIBUTES, which is the D1 study's arm A3 — the arm that lost half of
    A2's gain with a CI crossing zero. Replacing it is a real change that needs
    its own preregistration, not a bundled edit. This test pins the
    inconsistency so it stays visible instead of being discovered later.
    """
    players = [
        _p("Alpha", "LWR", "WR1", {"tgt_share": 0.30, "injury_status": "OUT"}),
        _p("WR2", "RWR", "WR1_5", {"tgt_share": 0.20}),
        _p("Slot", "SWR", "SLOT", {"tgt_share": 0.20}),
        _p("Back", "RB", "RB1", {"tgt_share": 0.10}),
    ]
    out = _run(players, monkeypatch)
    alpha = float(out.loc["Alpha", "rules_tgt_share"])
    assert alpha > 0.0, "legacy alpha rule changed without a preregistration"
    assert int(out["rules_injury_redistribution"].sum()) >= 1
