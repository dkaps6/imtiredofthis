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


def test_out_alpha_is_zeroed_and_his_share_is_redistributed(monkeypatch):
    """The alpha path and the general path now agree that OUT means zero.

    They differ in what happens to the freed share, and that difference is
    deliberate and measured. A non-alpha OUT player's share falls into
    ``_allocate_counts``'s residual bucket and is discarded. An OUT alpha's
    share is handed to his teammates, because team pass attempts come from
    ``binomial(plays, pass_rate)`` and do not depend on shares -- the
    quarterback throws the same number of passes either way. Across 14
    OUT/DOUBTFUL alpha-WR vacancies in 2026 weeks 3-9, 100.0% of the vacated
    aggregate target share landed on teammates (WR 73.7%, TE 22.5%, RB 4.9%).
    """
    players = [
        _p("Alpha", "LWR", "WR1", {"tgt_share": 0.30, "injury_status": "OUT"}),
        _p("WR2", "RWR", "WR1_5", {"tgt_share": 0.20}),
        _p("Slot", "SWR", "SLOT", {"tgt_share": 0.20}),
        _p("Back", "RB", "RB1", {"tgt_share": 0.10}),
    ]
    out = _run(players, monkeypatch)
    assert float(out.loc["Alpha", "rules_tgt_share"]) == pytest.approx(0.0)
    assert float(out["rules_tgt_share"].sum()) == pytest.approx(0.80), "team mass not conserved"
    assert int(out["rules_injury_redistribution"].sum()) >= 1


def test_alpha_vacancy_is_redistributed_not_discarded(monkeypatch):
    """Guards the distinction from the non-alpha path.

    If someone makes the alpha path discard like the general path, team target
    mass drops and the model projects a team that simply throws fewer
    completions to anybody. That is the failure this pins.
    """
    players = [
        _p("Alpha", "LWR", "WR1", {"tgt_share": 0.30, "injury_status": "OUT"}),
        _p("WR2", "RWR", "WR1_5", {"tgt_share": 0.20}),
        _p("Slot", "SWR", "SLOT", {"tgt_share": 0.20}),
        _p("Back", "RB", "RB1", {"tgt_share": 0.10}),
    ]
    out = _run(players, monkeypatch)
    teammates = float(out.loc[["WR2", "Slot", "Back"], "rules_tgt_share"].sum())
    assert teammates == pytest.approx(0.50 + 0.30), "vacated share was discarded, not redistributed"


def test_alpha_redistribution_fraction_is_total():
    from scripts.modeling.simulation_rules import ALPHA_VACANCY_REDISTRIBUTED_FRACTION
    assert ALPHA_VACANCY_REDISTRIBUTED_FRACTION == 1.00
