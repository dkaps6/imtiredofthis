"""Tests for the two model fixes promoted 2026-11-24.

RB_CARRY_SHARE_RECENCY_V2 (scripts/modeling/rb_carry_recency_v2.py, used by scripts/player_form_v2.py) and
RB_RECEPTIONS_OPPONENT_ADJ_V1 (scripts/modeling/rb_receptions_opponent_adj_v1.py).
"""
import numpy as np
import pandas as pd
import pytest

import scripts.player_form_v2 as pf
import scripts.modeling.rb_receptions_opponent_adj_v1 as adj


# ---------------------------------------------------------------- recency (V2)
import scripts.modeling.rb_carry_recency_v2 as rec


def _games(rows):
    # (key, team, week, offense_pct, carries, team_carries)
    return pd.DataFrame(rows, columns=["key", "team", "week", "offense_pct", "carries", "team_carries"])


def _tw(team, weeks):
    return pd.DataFrame({"team": team, "week": list(weeks)})


def _inj(rows):
    return pd.DataFrame(rows, columns=["key", "week", "report_status", "practice_status"])


def _states(labeled, key):
    return dict(zip(labeled.loc[labeled.key.eq(key), "week"], labeled.loc[labeled.key.eq(key), "game_state"]))


def test_game_states_follow_the_anomaly_gate():
    g = _games(
        [("inj", "KC", w, p, 10, 25) for w, p in [(1, .6), (2, .6), (3, .6), (4, .25), (5, .6)]]
        + [("exit", "KC", w, p, 10, 25) for w, p in [(1, .6), (2, .6), (3, .2), (4, .6)]]
        + [("bench", "KC", w, p, 10, 25) for w, p in [(1, .6), (2, .6), (3, .2), (4, .6)]]
        + [("spike", "KC", w, p, 10, 25) for w, p in [(1, .15), (2, .15), (3, .7), (4, .15)]]
        + [("ret", "KC", w, p, 10, 25) for w, p in [(1, .6), (2, .6), (5, .3), (6, .6)]]
        + [("back_ok", "KC", w, p, 10, 25) for w, p in [(1, .6), (2, .6), (5, .6), (6, .6)]]
    )
    inj = _inj([("inj", 4, "Questionable", None), ("exit", 4, None, "Did Not Participate In Practice")])
    lab = rec.label_games(g, _tw("KC", range(1, 7)), inj)
    assert _states(lab, "inj")[4] == "INJURY_LIMITED"
    assert _states(lab, "exit")[3] == "PARTIAL_GAME"          # big drop, on next week's report
    assert _states(lab, "bench")[3] == "UNRESOLVED_CONTEXT"   # big drop, no injury record
    assert _states(lab, "spike")[3] == "UNRESOLVED_CONTEXT"   # teammate absence / exit not resolved
    assert _states(lab, "ret")[5] == "RETURN_FROM_INJURY"     # missed weeks 3-4, limited on return
    assert set(_states(lab, "back_ok").values()) == {rec.HEALTHY}  # returned at his normal share


def test_persistent_new_role_is_not_an_anomaly():
    g = _games([("rb", "KC", w, p, 10, 25) for w, p in [(1, .7), (2, .7), (3, .3), (4, .3), (5, .3)]])
    lab = rec.label_games(g, _tw("KC", range(1, 6)), _inj([]))
    # the three games at the new level judge clean, so the window can use them
    assert [_states(lab, "rb")[w] for w in (3, 4, 5)] == [rec.HEALTHY] * 3
    win = rec.last3(lab, pd.DataFrame({"key": ["rb"], "team": ["KC"]}))
    assert bool(win.loc["rb", "window_clean"])


def test_zero_touch_snap_games_count_in_the_window():
    # review finding: PlayerForm logs drop zero-touch games; the window must still count them
    snaps = pd.DataFrame([dict(pfr_player_id="p1", team="KC", week=w, offense_snaps=20, offense_pct=.3) for w in range(1, 7)])
    logs = pd.DataFrame([dict(key="rb1", team="KC", week=w, rushes=10, team_rushes=25) for w in range(1, 5)]
                        + [dict(key="rbX", team="KC", week=w, rushes=25, team_rushes=25) for w in (5, 6)])
    games = rec.snap_games(snaps, {"p1": "rb1"}, logs, 7)
    assert games.set_index("week").carries.to_dict() == {1: 10, 2: 10, 3: 10, 4: 10, 5: 0, 6: 0}
    win = rec.last3(rec.label_games(games, logs[["team", "week"]].drop_duplicates(), _inj([])),
                    pd.DataFrame({"key": ["rb1"], "team": ["KC"]}))
    assert win.loc["rb1", "window_weeks"] == "4|5|6"
    assert win.loc["rb1", "rush_share_last3"] == pytest.approx(10 / 75)


def test_snap_team_codes_are_canonicalised():
    snaps = pd.DataFrame([dict(pfr_player_id="p1", team="LA", week=1, offense_snaps=20, offense_pct=.5)])
    logs = pd.DataFrame([dict(key="rb1", team="LAR", week=1, rushes=10, team_rushes=25)])
    assert len(rec.snap_games(snaps, {"p1": "rb1"}, logs, 2)) == 1


def _form(rows):
    return pd.DataFrame(rows, columns=["player_identity_key", "team", "position", "current_games", "rush_share_current"])


def _win(rows):
    return pd.DataFrame(rows, columns=["key", "rush_share_last3", "window_games", "window_states", "window_weeks",
                                       "window_clean"]).set_index("key")


def test_apply_changes_only_clean_rb_windows():
    form = _form([("clean", "KC", "RB", 5, .40), ("dirty", "KC", "RB", 5, .40), ("wr", "KC", "WR", 5, .04),
                  ("thin", "KC", "RB", 2, .40), ("fb", "KC", "FB", 4, .10)])
    win = _win([("clean", .60, 3, "HEALTHY_FULL|HEALTHY_FULL|HEALTHY_FULL", "3|4|5", True),
                ("dirty", .10, 3, "HEALTHY_FULL|INJURY_LIMITED|HEALTHY_FULL", "3|4|5", False),
                ("wr", .50, 3, "HEALTHY_FULL|HEALTHY_FULL|HEALTHY_FULL", "2|3|4", True),
                ("fb", .20, 3, "HEALTHY_FULL|HEALTHY_FULL|HEALTHY_FULL", "2|3|4", True)])
    out = rec.apply(form, win).set_index("player_identity_key")
    assert out.loc["clean", "rush_share_current"] == pytest.approx(.15 * .40 + .85 * .60)
    assert out.loc["fb", "rush_share_current"] == pytest.approx(.15 * .10 + .85 * .20)
    for k, v in (("dirty", .40), ("wr", .04), ("thin", .40)):
        assert out.loc[k, "rush_share_current"] == pytest.approx(v)
    assert out.rb_carry_recency_status.to_dict() == {
        "clean": "APPLIED", "dirty": "WITHHELD_ANOMALY", "wr": "NOT_RB", "thin": "FEWER_THAN_3_GAMES", "fb": "APPLIED"}
    assert out.loc["dirty", "rb_carry_recency_window_states"] == "HEALTHY_FULL|INJURY_LIMITED|HEALTHY_FULL"
    assert out.loc["clean", "rush_share_current_season"] == pytest.approx(.40)
    assert set(out.rb_carry_recency_version) == {rec.VERSION}


def test_unavailable_sources_change_nothing_and_say_why():
    form = _form([("rb", "KC", "RB", 5, .40)])
    out = rec.apply(form, None, unavailable_reason="nflreadpy:ConnectionError")
    assert out.rush_share_current.iat[0] == pytest.approx(.40)
    assert out.rb_carry_recency_status.iat[0] == "SOURCE_UNAVAILABLE:nflreadpy:ConnectionError"


def test_only_games_with_the_slate_team_count():
    g = _games([("rb3", "NYJ", w, .6, 20, 25) for w in (1, 2, 3)] + [("rb3", "KC", w, .6, 5, 25) for w in (4, 5)])
    tw = pd.concat([_tw("NYJ", range(1, 6)), _tw("KC", range(1, 6))])
    win = rec.last3(rec.label_games(g, tw, _inj([])), pd.DataFrame({"key": ["rb3"], "team": ["KC"]}))
    assert win.loc["rb3", "window_games"] == 2 and np.isnan(win.loc["rb3", "rush_share_last3"])


def _sources(snap_weeks):
    snaps = pd.DataFrame([dict(pfr_player_id="p1", team="KC", week=w, offense_snaps=30, offense_pct=.6, game_type="REG")
                          for w in snap_weeks])
    players = pd.DataFrame({"gsis_id": ["g1"], "pfr_id": ["p1"]})
    inj = pd.DataFrame(columns=["gsis_id", "week", "report_status", "practice_status"])
    logs = pd.DataFrame([dict(player_id="g1", player_identity_key="rb1", team="KC", week=w, rushes=12, team_rushes=25)
                         for w in range(1, 6)])
    form = _form([("rb1", "KC", "RB", 5, 12 / 25)])
    return snaps, players, inj, logs, form


def test_production_windows_end_to_end_and_incomplete_snap_source():
    snaps, players, inj, logs, form = _sources(range(1, 6))
    win, reason = pf.build_rb_recency_windows(snaps, players, inj, logs, logs, form, 6)
    out = rec.apply(form, win, unavailable_reason=reason)
    assert reason == "" and out.rb_carry_recency_status.iat[0] == "APPLIED"
    snaps, players, inj, logs, form = _sources([1, 2, 3, 4])          # week 5 snaps not published yet
    win, reason = pf.build_rb_recency_windows(snaps, players, inj, logs, logs, form, 6)
    out = rec.apply(form, win, unavailable_reason=reason)
    assert out.rb_carry_recency_status.iat[0] == "WITHHELD_SOURCE_INCOMPLETE"
    assert out.rush_share_current.iat[0] == pytest.approx(12 / 25)


def test_recency_kill_switch(monkeypatch):
    monkeypatch.setenv("RB_CARRY_SHARE_RECENCY", "0")
    win, reason = pf.rb_recency_windows(2026, 6, pd.DataFrame(), pd.DataFrame({"x": [1]}), pd.DataFrame())
    assert win is None and reason == "KILL_SWITCH"


# ---------------------------------------------------------------- opponent adjustment
def _stats():
    rows = []
    # weeks 1-3; defense AAA allows 6 RB catches a game, BBB allows 2, and a week-3 game
    # (the slate week) that must NOT be counted
    for w in (1, 2):
        rows += [dict(season=2026, week=w, team="XA", position="RB", receptions=6),
                 dict(season=2026, week=w, team="XB", position="RB", receptions=2),
                 dict(season=2026, week=w, team="XA", position="WR", receptions=9)]
    rows += [dict(season=2026, week=3, team="XA", position="RB", receptions=50)]
    return pd.DataFrame(rows)


def _opps():
    return pd.DataFrame([dict(season=2026, week=w, team=t, opponent=o)
                         for w in (1, 2, 3) for t, o in (("XA", "AAA"), ("XB", "BBB"))])


def test_context_is_strictly_prior_and_shrunk():
    ctx = adj.build_context(_stats(), _opps(), 2026, 3).set_index("defense")
    assert ctx.loc["AAA", "allowed_pg"] == pytest.approx(6.0)      # week-3 game excluded, WR excluded
    assert ctx.loc["BBB", "allowed_pg"] == pytest.approx(2.0)
    league = 4.0
    shrink = 2 / (2 + adj.SHRINK_K)
    assert ctx.loc["AAA", "D"] == pytest.approx(shrink * (6 / league - 1))
    assert ctx.loc["BBB", "D"] == pytest.approx(shrink * (2 / league - 1))
    assert set(ctx.version) == {adj.VERSION}


def test_factor_applies_only_to_rb_receptions_after_week_one():
    ctx = {"AAA": 0.2}
    assert adj.factor("receptions", "RB", "AAA", 5, ctx) == (pytest.approx(1 + adj.B * 0.2), 0.2, 1)
    assert adj.factor("rec_yards", "RB", "AAA", 5, ctx)[2] == 0
    assert adj.factor("receptions", "WR", "AAA", 5, ctx)[2] == 0
    assert adj.factor("receptions", "RB", "AAA", 1, ctx)[2] == 0


def test_factor_fails_closed_on_unknown_opponent():
    with pytest.raises(RuntimeError, match="not in context"):
        adj.factor("receptions", "RB", "ZZZ", 5, {"AAA": 0.1})


def test_load_context_fails_closed(tmp_path):
    p = tmp_path / "ctx.csv"
    assert adj.load_context(2026, 1, p) == {}                       # Week 1 is a no-op
    with pytest.raises(RuntimeError, match="missing"):
        adj.load_context(2026, 5, p)
    adj.build_context(_stats(), _opps(), 2026, 3).to_csv(p, index=False)
    with pytest.raises(RuntimeError, match="another slate"):
        adj.load_context(2026, 4, p)
    assert set(adj.load_context(2026, 3, p)) == {"AAA", "BBB"}


def test_promoted_coefficient_matches_the_research_artifact():
    import json
    from pathlib import Path
    art = json.loads(Path("data/research/opponent_position_defense/promoted_coefficients_v1.json").read_text())
    assert art["coefficients"]["receptions|RB"] == pytest.approx(adj.B)
    assert art["shrink_k"] == pytest.approx(adj.SHRINK_K)


# ---------------------------------------------------------------- wiring (the full price() loop
# needs the whole ML/state/rules/ensemble stack; follow the repo's source-wiring pattern)
def test_pricing_seam_is_wired_before_mean_alignment():
    from pathlib import Path
    text = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    assert "import scripts.modeling.rb_receptions_opponent_adj_v1 as rb_rec_opp" in text
    assert "rb_rec_opp.load_context(int(season), w)" in text
    seam = text.index("rb_rec_opp.factor(")
    align = text.index("adjusted_outcomes = base_outcomes * max(0.0, target_mean / mc_proj)")
    assert seam < align, "the adjustment must set target_mean before the draws are aligned to it"
    for col in ("rb_receptions_opp_adj_applied", "rb_receptions_opp_adj_version",
                "rb_receptions_opp_adj_D", "rb_receptions_opp_adj_factor"):
        assert f'"{col}"' in text
    assert "applied to zero non-Week-1 RB receptions rows" in text


def test_full_slate_builds_the_context_before_odds_and_pricing():
    from pathlib import Path
    wf = Path(".github/workflows/full-slate.yml").read_text(encoding="utf-8")
    step = wf.index("python scripts/build_rb_receptions_opponent_context_v1.py")
    assert step < wf.index("Fetch and gate player props and game odds")
    assert step < wf.index("Price from certified football-first Full Slate stack")
    assert step > wf.index("Build PlayerForm from production-eligible current roles")


def test_playerform_and_the_validation_share_one_recency_module():
    from pathlib import Path
    text = Path("scripts/player_form_v2.py").read_text(encoding="utf-8")
    assert "form = rb_recency.apply(form, rb_windows, unavailable_reason=rb_reason)" in text
    research = Path("scripts/research/evaluate_rb_recency_end_to_end_v2.py").read_text(encoding="utf-8")
    assert "import scripts.modeling.rb_carry_recency_v2 as rec" in research
