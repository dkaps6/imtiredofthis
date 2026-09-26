"""Simulation-universe coverage is the certified slate, not a literal 32.

Full Slate used to require exactly 32 teams in the football simulation
universe. That holds only while an entire week is still upcoming: once a game
kicks off the books take those teams down, so a mid-week live run can never
reach 32. Observed on 2026 Week 3 (run 36276421905) -- Thursday had played,
live props covered 15 of 16 games, and pricing died with
"football simulation universe must cover 32 teams, found 30" after the odds had
already been paid for.

A count was also the wrong contract: 32 teams passes even when they are the
wrong 32.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts._opponent_map import canon_team

MOD = "scripts.run_pricing_with_full_roster_universe_v1"

FULL_SLATE = [
    ("ARI", "SEA"), ("ATL", "CAR"), ("BAL", "CIN"), ("BUF", "MIA"),
    ("CHI", "DET"), ("CLE", "PIT"), ("DAL", "NYG"), ("DEN", "LV"),
    ("GB", "MIN"), ("HOU", "IND"), ("JAX", "TEN"), ("KC", "LAC"),
    ("NE", "NYJ"), ("NO", "TB"), ("PHI", "WAS"), ("LAR", "SF"),
]


def _write_cert(root, games, ineligible=()):
    rows = []
    for away, home in games:
        rows.append({
            "season": 2026, "week": 3, "game_id": f"2026_03_{away}_{home}",
            "away_team": away, "home_team": home,
            "production_eligible": (away, home) not in set(ineligible),
            "certification_state": "CERTIFIED",
        })
    (root / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        root / "data" / "current_player_availability_game_certification.csv", index=False
    )


@pytest.fixture()
def mod(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    m = importlib.import_module(MOD)
    importlib.reload(m)
    return m


def test_full_week_yields_all_32_teams(mod, tmp_path):
    _write_cert(tmp_path, FULL_SLATE)
    teams = mod._certified_slate_teams()
    assert len(teams) == 32


def test_post_thursday_partial_week_yields_30_not_an_error(mod, tmp_path):
    # The Thursday game has kicked off, so its two teams are no longer
    # production-eligible. This is the exact 2026 Week 3 shape.
    _write_cert(tmp_path, FULL_SLATE, ineligible=[("ARI", "SEA")])
    teams = mod._certified_slate_teams()
    assert len(teams) == 30
    assert canon_team("ARI") not in teams and canon_team("SEA") not in teams


def test_missing_certification_fails_closed(mod, tmp_path):
    with pytest.raises(RuntimeError, match="game certification missing"):
        mod._certified_slate_teams()


def test_zero_eligible_games_fails_closed(mod, tmp_path):
    _write_cert(tmp_path, FULL_SLATE, ineligible=FULL_SLATE)
    with pytest.raises(RuntimeError, match="zero production-eligible games"):
        mod._certified_slate_teams()


def test_malformed_certification_columns_fail_closed(mod, tmp_path):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"away_team": "ARI", "home_team": "SEA"}]).to_csv(
        tmp_path / "data" / "current_player_availability_game_certification.csv", index=False
    )
    with pytest.raises(RuntimeError, match="missing columns"):
        mod._certified_slate_teams()


def test_odd_team_count_is_rejected_as_impossible(mod, tmp_path):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"away_team": "ARI", "home_team": "SEA", "production_eligible": True},
        {"away_team": "ARI", "home_team": "GB", "production_eligible": True},
    ]).to_csv(tmp_path / "data" / "current_player_availability_game_certification.csv", index=False)
    with pytest.raises(RuntimeError, match="impossible slate of 3 teams"):
        mod._certified_slate_teams()


def test_team_aliases_are_canonicalized(mod, tmp_path):
    _write_cert(tmp_path, [("JAC", "LA"), ("BLT", "ARZ")])
    teams = mod._certified_slate_teams()
    assert teams == {canon_team(t) for t in ("JAC", "LA", "BLT", "ARZ")}
    assert len(teams) == 4
