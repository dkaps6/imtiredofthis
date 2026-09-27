import pandas as pd
import pytest

from scripts.modeling.contracts import PlayerContext, TeamContext
from scripts.modeling import simulation_rules as sr


def _contexts():
    off = TeamContext(team="AAA", season=2025, success_rate_off=0.45, plays_est=64.0)
    deff = TeamContext(team="BBB", season=2025, success_rate_def=0.45)
    players = [
        PlayerContext(
            player="RB One", team="AAA", opponent="BBB", season=2025, week=3,
            position="RB", role="RB1", offense=off, defense=deff,
            features={"rush_share": 0.52, "tgt_share": 0.10, "ypc": 4.2, "ypt": 6.0, "receptions_per_target": 0.75},
        ),
        PlayerContext(
            player="WR One", team="AAA", opponent="BBB", season=2025, week=3,
            position="WR", role="WR1", offense=off, defense=deff,
            features={"rush_share": 0.01, "tgt_share": 0.28, "ypt": 8.0, "receptions_per_target": 0.64},
        ),
        PlayerContext(
            player="TE One", team="AAA", opponent="BBB", season=2025, week=3,
            position="TE", role="TE1", offense=off, defense=deff,
            features={"rush_share": 0.00, "tgt_share": 0.18, "ypt": 7.0, "receptions_per_target": 0.68},
        ),
        PlayerContext(
            player="QB One", team="AAA", opponent="BBB", season=2025, week=3,
            position="QB", role="QB1", offense=off, defense=deff,
            features={"rush_share": 0.12, "tgt_share": 0.0, "ypa": 7.1},
        ),
    ]
    return off, deff, players


def _metrics():
    return pd.DataFrame([
        {"player":"RB One","player_clean_key":"rbone","team":"AAA","position":"RB","tgt_share":0.10,"rush_share":0.52,
         "bayes_tgt_share":0.08,"bayes_rush_share":0.34,"bayes_ypt":6.0,"bayes_ypc":4.2,"bayes_receptions_per_target":0.75},
        {"player":"WR One","player_clean_key":"wrone","team":"AAA","position":"WR","tgt_share":0.28,"rush_share":0.01,
         "bayes_tgt_share":0.20,"bayes_rush_share":0.02,"bayes_ypt":8.0,"bayes_ypc":5.0,"bayes_receptions_per_target":0.64},
        {"player":"TE One","player_clean_key":"teone","team":"AAA","position":"TE","tgt_share":0.18,"rush_share":0.0,
         "bayes_tgt_share":0.12,"bayes_rush_share":0.0,"bayes_ypt":7.0,"bayes_ypc":4.0,"bayes_receptions_per_target":0.68},
        {"player":"QB One","player_clean_key":"qbone","team":"AAA","position":"QB","tgt_share":0.0,"rush_share":0.12,
         "bayes_tgt_share":0.0,"bayes_rush_share":0.09,"bayes_ypa":7.1},
    ])


def test_default_authority_remains_bayesian(monkeypatch):
    _, _, players = _contexts()
    monkeypatch.setattr(sr, "load_model_contexts", lambda: ({}, players))
    out = sr.apply_rules_to_metrics(_metrics())
    x = out.set_index("player")
    assert x.loc["RB One", "rules_rush_share"] == pytest.approx(0.34)
    assert x.loc["WR One", "rules_tgt_share"] == pytest.approx(0.20)
    assert x.loc["TE One", "rules_tgt_share"] == pytest.approx(0.12)
    assert x.loc["QB One", "rules_rush_share"] == pytest.approx(0.09)


def test_fast_state_authority_changes_only_frozen_opportunity_cells(monkeypatch):
    _, _, players = _contexts()
    monkeypatch.setattr(sr, "load_model_contexts", lambda: ({}, players))
    base = sr.apply_rules_to_metrics(_metrics())
    cand = sr.apply_rules_to_metrics(
        _metrics(),
        opportunity_authority=sr.OPPORTUNITY_AUTHORITY_PLAYERFORM_FAST_STATE,
    )
    b = base.set_index("player")
    c = cand.set_index("player")

    assert c.loc["RB One", "rules_rush_share"] == pytest.approx(0.52)
    assert c.loc["WR One", "rules_tgt_share"] == pytest.approx(0.28)
    assert c.loc["TE One", "rules_tgt_share"] == pytest.approx(0.18)

    # Unfrozen opportunity cells retain baseline Bayes authority.
    assert c.loc["RB One", "rules_tgt_share"] == pytest.approx(b.loc["RB One", "rules_tgt_share"])
    assert c.loc["QB One", "rules_rush_share"] == pytest.approx(b.loc["QB One", "rules_rush_share"])

    # Efficiency semantics are invariant.
    for col in ("rules_ypt", "rules_ypc", "rules_ypa", "rules_catch_rate", "rules_pass_rate", "rules_plays_est"):
        pd.testing.assert_series_equal(
            b[col].reset_index(drop=True),
            c[col].reset_index(drop=True),
            check_names=False,
        )


def test_unknown_authority_fails_closed(monkeypatch):
    _, _, players = _contexts()
    monkeypatch.setattr(sr, "load_model_contexts", lambda: ({}, players))
    with pytest.raises(RuntimeError, match="unsupported opportunity authority"):
        sr.apply_rules_to_metrics(_metrics(), opportunity_authority="made_up")
