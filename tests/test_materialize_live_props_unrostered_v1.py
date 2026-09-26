"""A sportsbook player the production roster authority does not carry.

The downstream identity audit maps every model-facing row against that same
authority, so an unmappable row left in the compact layer fails the slate
closed after the odds have already been paid for. This module's existing
core/non-core split is the right place to decide what to do about it: raise on
a priced market, quarantine with a reason on a non-priced one.

Real case: 2026 Week 3 run 36275289734 fetched odds and then died at
"Audit live player identity semantics" on a single row -- Barion Brown, NO,
player_anytime_td -- an active Saints WR the Ourlads scrape had missed for
three weeks, in the one market production never prices.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_slate(root: Path, rows: list[dict], roster: list[tuple[str, str]] | None) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "live_odds_status.json").write_text(
        json.dumps({"available": True, "status": "live"}), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(root / "outputs" / "props_raw.csv", index=False)
    if roster is not None:
        pd.DataFrame(
            [{"team": t, "player": p, "role": "WR3", "position": "WR"} for t, p in roster]
        ).to_csv(root / "data" / "roles_current_production_eligible_v1.csv", index=False)


def _row(player: str, market: str, team: str = "NO") -> dict:
    return {
        "event_id": "evt1",
        "market": market,
        "player": player,
        "team_abbr": team,
        "opponent_abbr": "SEA",
        "offers_json": '[{"book":"draftkings","line":10.5,"odds":-114}]',
        "bookmaker_missing": 0,
    }


@pytest.fixture()
def slate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ACTIVE_ROLES_CSV", raising=False)
    return tmp_path


def _materialize():
    import importlib

    import scripts.materialize_live_props_for_model_v1 as mod

    importlib.reload(mod)
    return mod, mod.materialize()


def test_unrostered_noncore_row_is_quarantined_not_fatal(slate):
    _write_slate(
        slate,
        [_row("Chris Olave", "player_reception_yds"), _row("Barion Brown", "player_anytime_td")],
        roster=[("NO", "Chris Olave")],
    )
    mod, result = _materialize()
    q = pd.read_csv(slate / "data" / "live_odds_placeholder_rows.csv")
    assert "UNROSTERED_NONCORE_PLAYER" in set(q["quarantine_reason"])
    assert set(q.loc[q["quarantine_reason"].eq("UNROSTERED_NONCORE_PLAYER"), "player"]) == {"Barion Brown"}
    compact = pd.read_csv(slate / "outputs" / "props_raw_compact.csv")
    assert "Barion Brown" not in set(compact["player"])
    assert "Chris Olave" in set(compact["player"])
    assert result["quarantine_reasons"]["UNROSTERED_NONCORE_PLAYER"] == 1


def test_unrostered_priced_market_row_still_fails_closed(slate):
    _write_slate(
        slate,
        [_row("Chris Olave", "player_reception_yds"), _row("Barion Brown", "player_rush_yds")],
        roster=[("NO", "Chris Olave")],
    )
    with pytest.raises(RuntimeError, match="absent from the production roster authority"):
        _materialize()


def test_rostered_players_are_untouched(slate):
    _write_slate(
        slate,
        [_row("Chris Olave", "player_reception_yds"), _row("Alvin Kamara", "player_anytime_td")],
        roster=[("NO", "Chris Olave"), ("NO", "Alvin Kamara")],
    )
    mod, result = _materialize()
    compact = pd.read_csv(slate / "outputs" / "props_raw_compact.csv")
    assert set(compact["player"]) == {"Chris Olave", "Alvin Kamara"}
    assert "UNROSTERED_NONCORE_PLAYER" not in result.get("quarantine_reasons", {})


def test_suffix_and_team_alias_differences_do_not_count_as_unrostered(slate):
    # The roster spells him with the suffix and uses the alternate team code;
    # the book does neither. Keying must agree with the identity audit, which
    # strips suffixes and canonicalizes the team.
    _write_slate(
        slate,
        [_row("Brian Thomas", "player_anytime_td", team="JAC")],
        roster=[("JAX", "Brian Thomas Jr.")],
    )
    mod, result = _materialize()
    compact = pd.read_csv(slate / "outputs" / "props_raw_compact.csv")
    assert "Brian Thomas" in set(compact["player"])
    assert "UNROSTERED_NONCORE_PLAYER" not in result.get("quarantine_reasons", {})


def test_absent_roster_authority_skips_the_rule_rather_than_crashing(slate):
    _write_slate(slate, [_row("Barion Brown", "player_anytime_td")], roster=None)
    mod, result = _materialize()
    compact = pd.read_csv(slate / "outputs" / "props_raw_compact.csv")
    assert "Barion Brown" in set(compact["player"])


def test_everything_unrostered_is_treated_as_a_broken_authority(slate):
    _write_slate(
        slate,
        [_row("Barion Brown", "player_anytime_td"), _row("Juwan Johnson", "player_anytime_td")],
        roster=[("NO", "Somebody Else")],
    )
    with pytest.raises(RuntimeError, match="roster authority is almost certainly broken"):
        _materialize()
