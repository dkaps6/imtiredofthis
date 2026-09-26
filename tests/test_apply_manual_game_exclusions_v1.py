"""Withholding one inconsistent game instead of the whole slate.

2026 Week 3: the simulation-universe rebuild and the pricing metrics disagreed
on rules_tgt_share for 6 LAR players and no one else, so pricing refused the
entire 15-game slate. Excluding that one game leaves 14 consistent games that
can be priced, and records why.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts._opponent_map import canon_team

GAMES = [("LAR", "SF"), ("KC", "LAC"), ("BUF", "MIA"), ("DAL", "NYG")]


def _cert(root, eligible=True):
    (root / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"season": 2026, "week": 3, "game_id": f"2026_03_{a}_{h}", "away_team": a,
         "home_team": h, "production_eligible": eligible, "certification_state": "CERTIFIED",
         "failure_reason": ""}
        for a, h in GAMES
    ]).to_csv(root / "data" / "current_player_availability_game_certification.csv", index=False)


def _excl(root, rows):
    pd.DataFrame(rows).to_csv(root / "data" / "manual_game_exclusions.csv", index=False)


def _row(team, season=2026, week=3, **kw):
    base = {"season": season, "week": week, "team": team, "reason": "inputs inconsistent",
            "verified_source": "drift dump from preserved run", "verified_date": "2026-09-26"}
    base.update(kw)
    return base


@pytest.fixture()
def mod(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    m = importlib.import_module("scripts.operations.apply_manual_game_exclusions_v1")
    importlib.reload(m)
    return m


def test_excluding_one_game_leaves_the_rest_priceable(mod, tmp_path):
    _cert(tmp_path); _excl(tmp_path, [_row("LAR")])
    r = mod.apply(season=2026, week=3)
    assert r["eligible_games_before"] == 4
    assert r["eligible_games_after"] == 3
    assert r["eligible_teams_after"] == 6
    assert r["full_league_slate"] is False
    cert = pd.read_csv(tmp_path / "data" / "current_player_availability_game_certification.csv")
    dropped = cert[cert.away_team.eq("LAR")]
    assert not bool(dropped.production_eligible.iloc[0])
    assert "manual_exclusion:LAR" in str(dropped.failure_reason.iloc[0])
    # the other games are untouched
    assert cert.loc[~cert.away_team.eq("LAR"), "production_eligible"].all()


def test_no_exclusions_file_is_a_clean_no_op(mod, tmp_path):
    _cert(tmp_path)
    r = mod.apply(season=2026, week=3)
    assert r["disposition"] == "NO_MANUAL_GAME_EXCLUSIONS"
    assert r["eligible_games_after"] == 4
    assert json.loads((tmp_path / "data" / "manual_game_exclusion_audit.json").read_text())


def test_exclusion_for_a_different_week_is_ignored(mod, tmp_path):
    _cert(tmp_path); _excl(tmp_path, [_row("LAR", week=2)])
    r = mod.apply(season=2026, week=3)
    assert r["eligible_games_after"] == 4
    assert r["exclusions_applied"] == []


def test_stale_exclusion_naming_no_game_fails_closed(mod, tmp_path):
    _cert(tmp_path); _excl(tmp_path, [_row("DEN")])
    with pytest.raises(RuntimeError, match="no 2026 week 3 game carries that team"):
        mod.apply(season=2026, week=3)


def test_every_row_must_record_why_and_how_it_was_verified(mod, tmp_path):
    _cert(tmp_path); _excl(tmp_path, [_row("LAR", verified_source="")])
    with pytest.raises(RuntimeError, match="missing 'verified_source'"):
        mod.apply(season=2026, week=3)


def test_excluding_the_whole_slate_is_refused(mod, tmp_path):
    _cert(tmp_path)
    _excl(tmp_path, [_row(a) for a, _ in GAMES])
    with pytest.raises(RuntimeError, match="zero production-eligible games"):
        mod.apply(season=2026, week=3)


def test_team_aliases_match(mod, tmp_path):
    _cert(tmp_path); _excl(tmp_path, [_row("LA")])
    r = mod.apply(season=2026, week=3)
    assert r["eligible_games_after"] == 3
    assert r["exclusions_applied"][0]["team"] == canon_team("LA")


def test_missing_certification_fails_closed(mod, tmp_path):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    _excl(tmp_path, [_row("LAR")])
    with pytest.raises(RuntimeError, match="game certification missing"):
        mod.apply(season=2026, week=3)
