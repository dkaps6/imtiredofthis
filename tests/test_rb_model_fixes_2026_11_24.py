"""Tests for the two model fixes promoted 2026-11-24.

RB_CARRY_SHARE_RECENCY_V1 (scripts/player_form_v2.py) and
RB_RECEPTIONS_OPPONENT_ADJ_V1 (scripts/modeling/rb_receptions_opponent_adj_v1.py).
"""
import numpy as np
import pandas as pd
import pytest

import scripts.player_form_v2 as pf
import scripts.modeling.rb_receptions_opponent_adj_v1 as adj


# ---------------------------------------------------------------- recency
def _logs(rows):
    return pd.DataFrame(rows, columns=["player_identity_key", "team", "week", "rushes", "team_rushes"])


def _form(rows):
    return pd.DataFrame(rows, columns=["player_identity_key", "team", "position", "current_games", "rush_share_current"])


def test_recency_blends_season_and_last3_for_rbs():
    # 5 games: 5/25 carries each early, then 15/25 in the last three (a role change)
    logs = _logs([("rb1", "KC", w, r, 25) for w, r in [(1, 5), (2, 5), (3, 15), (4, 15), (5, 15)]])
    std = 55 / 125
    form = _form([("rb1", "KC", "RB", 5, std)])
    out = pf.apply_rb_carry_share_recency(form, logs)
    l3 = 45 / 75
    assert out.rush_share_last3.iat[0] == pytest.approx(l3)
    assert out.rush_share_current.iat[0] == pytest.approx(0.15 * std + 0.85 * l3)
    assert out.rush_share_current_season.iat[0] == pytest.approx(std)
    assert out.rb_carry_recency_applied.iat[0] == 1
    assert out.rb_carry_recency_version.iat[0] == pf.RB_RECENCY_VERSION


def test_recency_leaves_other_positions_and_thin_histories_alone():
    logs = _logs([("wr1", "KC", w, 1, 25) for w in (1, 2, 3, 4)] + [("rb2", "KC", w, 10, 25) for w in (1, 2)])
    form = _form([("wr1", "KC", "WR", 4, 0.04), ("rb2", "KC", "RB", 2, 0.40)])
    out = pf.apply_rb_carry_share_recency(form, logs)
    assert out.rush_share_current.tolist() == pytest.approx([0.04, 0.40])
    assert out.rb_carry_recency_applied.tolist() == [0, 0]


def test_recency_uses_only_games_with_the_current_team():
    # traded: three games for NYJ, then two for KC -> no 3 games with KC yet, so no change
    logs = _logs([("rb3", "NYJ", w, 20, 25) for w in (1, 2, 3)] + [("rb3", "KC", w, 5, 25) for w in (4, 5)])
    form = _form([("rb3", "KC", "RB", 5, 0.6)])
    out = pf.apply_rb_carry_share_recency(form, logs)
    assert out.rb_carry_recency_applied.iat[0] == 0
    assert out.rush_share_current.iat[0] == pytest.approx(0.6)


def test_recency_kill_switch(monkeypatch):
    monkeypatch.setenv("RB_CARRY_SHARE_RECENCY_V1", "0")
    logs = _logs([("rb1", "KC", w, 15, 25) for w in (1, 2, 3, 4)])
    form = _form([("rb1", "KC", "RB", 4, 0.3)])
    out = pf.apply_rb_carry_share_recency(form, logs)
    assert out.rb_carry_recency_applied.iat[0] == 0
    assert out.rush_share_current.iat[0] == pytest.approx(0.3)


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


def test_playerform_build_applies_recency():
    from pathlib import Path
    text = Path("scripts/player_form_v2.py").read_text(encoding="utf-8")
    assert "form = apply_rb_carry_share_recency(form, current_logs)" in text
