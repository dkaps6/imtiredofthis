"""Absence from PlayerForm has two causes, and they are not the same defect.

A sportsbook row naming a player PlayerForm does not carry can mean:

  * we LOST him -- an identity collision, a missing name override, a signing the
    roster scrape missed. A real defect, and the slate must fail closed.
  * we RULED HIM OUT -- the official inactive section, the club's own
    injury-report designation, or weekly roster non-membership says he is not
    playing. Not a defect. His offers do not belong on a board we would bet.

Week 10 2026 run 43320022375 bought the odds and then died on twelve rows that
were a mix of both. Six players: Skattebo, Jaylen Wright and Rashid Shaheed
were lost to an ESPN injury-log proxy that had overwritten a club report
showing them at FULL practice participation; Burks, MarShawn Lloyd and Deshaun
Watson were genuinely ruled out by the football sources while three books
priced them anyway.

The exemption is deliberately narrow:
  * it needs AFFIRMATIVE evidence -- definitive_unavailable == 1 in
    data/current_player_availability.csv. A missing, empty or schema-less
    artifact exempts nothing.
  * it is bounded. Above RULED_UNAVAILABLE_MAX_SHARE of the board, "everyone is
    out" is itself the bug and the slate fails closed rather than draining.
  * a sportsbook offer is what gets DROPPED, never what decides. A book pricing
    a player we believe is out does not make him available (AGENTS.md s3).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _row(player: str, market: str, team: str = "NYG") -> dict:
    return {
        "event_id": "evt1",
        "market": market,
        "player": player,
        "team_abbr": team,
        "opponent_abbr": "WAS",
        "offers_json": '[{"book":"fanduel","line":40.5,"odds":-110}]',
        "bookmaker_missing": 0,
    }


def _write_slate(root: Path, rows: list[dict], roster: list[tuple[str, str]],
                 availability: list[dict] | None = None) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "live_odds_status.json").write_text(
        json.dumps({"available": True, "status": "live"}), encoding="utf-8")
    pd.DataFrame(rows).to_csv(root / "outputs" / "props_raw.csv", index=False)
    pd.DataFrame([{"team": t, "player": p, "position": "WR"} for t, p in roster]).to_csv(
        root / "data" / "player_form.csv", index=False)
    if availability is not None:
        pd.DataFrame(availability).to_csv(
            root / "data" / "current_player_availability.csv", index=False)


@pytest.fixture()
def slate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PLAYER_FORM_CSV", raising=False)
    return tmp_path


def _materialize():
    import importlib
    import scripts.materialize_live_props_for_model_v1 as mod
    importlib.reload(mod)
    return mod, mod.materialize()


def _filler(n: int) -> tuple[list[dict], list[tuple[str, str]]]:
    """n rostered priced rows, so one exemption stays under the share bound.

    Includes a rostered pass-yards row, because a priced market left with zero
    playable participants fails closed by design -- a real slate always has the
    opposing quarterback.
    """
    rows, roster = [], []
    for i in range(n):
        name = f"Filler Player{i}"
        rows.append(_row(name, "player_reception_yds"))
        roster.append(("NYG", name))
    rows.append(_row("Starting Quarterback", "player_pass_yds"))
    roster.append(("NYG", "Starting Quarterback"))
    return rows, roster


def test_ruled_out_player_is_quarantined_not_fatal(slate):
    rows, roster = _filler(20)
    rows.append(_row("Deshaun Watson", "player_pass_yds", team="CLE"))
    _write_slate(slate, rows, roster, availability=[
        {"team": "CLE", "player": "Deshaun Watson", "definitive_unavailable": 1},
    ])
    _mod, result = _materialize()
    q = pd.read_csv(slate / "data" / "live_odds_placeholder_rows.csv")
    assert "RULED_UNAVAILABLE_BY_FOOTBALL_SOURCES" in set(q["quarantine_reason"])
    assert set(q.loc[q["quarantine_reason"].eq("RULED_UNAVAILABLE_BY_FOOTBALL_SOURCES"),
                     "player"]) == {"Deshaun Watson"}
    compact = pd.read_csv(slate / "outputs" / "props_raw_compact.csv")
    assert "Deshaun Watson" not in set(compact["player"])
    assert result["quarantine_reasons"]["RULED_UNAVAILABLE_BY_FOOTBALL_SOURCES"] == 1


def test_lost_player_still_fails_closed(slate):
    """No availability row at all -> this is a roster/identity gap, not a ruling."""
    rows, roster = _filler(20)
    rows.append(_row("Cam Skattebo", "player_rush_yds"))
    _write_slate(slate, rows, roster, availability=[
        {"team": "NYG", "player": "Someone Else", "definitive_unavailable": 1},
    ])
    with pytest.raises(RuntimeError, match="absent from PlayerForm"):
        _materialize()


def test_available_player_absent_from_playerform_still_fails_closed(slate):
    """definitive_unavailable == 0 is not evidence of a ruling."""
    rows, roster = _filler(20)
    rows.append(_row("Cam Skattebo", "player_rush_yds"))
    _write_slate(slate, rows, roster, availability=[
        {"team": "NYG", "player": "Cam Skattebo", "definitive_unavailable": 0},
    ])
    with pytest.raises(RuntimeError, match="absent from PlayerForm"):
        _materialize()


def test_missing_availability_artifact_exempts_nothing(slate):
    rows, roster = _filler(20)
    rows.append(_row("Cam Skattebo", "player_rush_yds"))
    _write_slate(slate, rows, roster, availability=None)
    with pytest.raises(RuntimeError, match="absent from PlayerForm"):
        _materialize()


def test_availability_artifact_without_the_column_exempts_nothing(slate):
    rows, roster = _filler(20)
    rows.append(_row("Cam Skattebo", "player_rush_yds"))
    _write_slate(slate, rows, roster, availability=[
        {"team": "NYG", "player": "Cam Skattebo", "final_availability_state": "UNAVAILABLE_REPORTED"},
    ])
    with pytest.raises(RuntimeError, match="absent from PlayerForm"):
        _materialize()


def test_wholesale_exemption_fails_closed(slate):
    """If most of the board is "ruled out", the availability source is the bug."""
    rows, roster = _filler(4)
    out_names = [f"Out Player{i}" for i in range(6)]
    rows += [_row(n, "player_rush_yds") for n in out_names]
    _write_slate(slate, rows, roster, availability=[
        {"team": "NYG", "player": n, "definitive_unavailable": 1} for n in out_names
    ])
    with pytest.raises(RuntimeError, match="over-ruling players"):
        _materialize()


def test_ruled_out_is_not_also_counted_as_unrostered_noncore(slate):
    rows, roster = _filler(20)
    rows.append(_row("Deshaun Watson", "player_anytime_td", team="CLE"))
    _write_slate(slate, rows, roster, availability=[
        {"team": "CLE", "player": "Deshaun Watson", "definitive_unavailable": 1},
    ])
    _mod, result = _materialize()
    reasons = result.get("quarantine_reasons", {})
    assert reasons.get("RULED_UNAVAILABLE_BY_FOOTBALL_SOURCES") == 1
    assert "UNROSTERED_NONCORE_PLAYER" not in reasons


def test_sportsbook_offer_never_promotes_a_ruled_out_player(slate):
    """The offer is dropped; it does not reinstate him anywhere downstream."""
    rows, roster = _filler(20)
    rows.append(_row("Deshaun Watson", "player_pass_yds", team="CLE"))
    _write_slate(slate, rows, roster, availability=[
        {"team": "CLE", "player": "Deshaun Watson", "definitive_unavailable": 1},
    ])
    _mod, _result = _materialize()
    compact = pd.read_csv(slate / "outputs" / "props_raw_compact.csv")
    assert "Deshaun Watson" not in set(compact["player"])
    assert len(compact) == 21


def test_priced_market_emptied_by_exemptions_fails_closed(slate):
    """If every offer in a priced market belongs to a ruled-out player, stop.

    Pricing nothing is not the same as pricing a market, and the row-loss guard
    this exemption had to be accounted for in exists precisely so a priced
    market cannot drain silently.
    """
    rows, roster = _filler(20)
    rows = [r for r in rows if r["market"] != "player_pass_yds"]
    roster = [x for x in roster if x[1] != "Starting Quarterback"]
    rows.append(_row("Deshaun Watson", "player_pass_yds", team="CLE"))
    _write_slate(slate, rows, roster, availability=[
        {"team": "CLE", "player": "Deshaun Watson", "definitive_unavailable": 1},
    ])
    with pytest.raises(RuntimeError, match="no playable participant"):
        _materialize()


def test_row_loss_accounting_is_per_market(slate):
    """An exemption in one priced market must not excuse loss in another.

    The guard compares before/after per market and adds back only the rows
    removed from THAT market, so a rush-yards exemption cannot paper over a
    missing reception-yards row.
    """
    rows, roster = _filler(20)
    rows.append(_row("Rostered Back", "player_rush_yds"))
    roster.append(("NYG", "Rostered Back"))
    rows.append(_row("Deshaun Watson", "player_rush_yds", team="CLE"))
    rows.append(_row("Deshaun Watson", "player_reception_yds", team="CLE"))
    _write_slate(slate, rows, roster, availability=[
        {"team": "CLE", "player": "Deshaun Watson", "definitive_unavailable": 1},
    ])
    _mod, result = _materialize()
    # Both of his rows are accounted for, each against its own market.
    assert result["quarantine_reasons"]["RULED_UNAVAILABLE_BY_FOOTBALL_SOURCES"] == 2
    compact = pd.read_csv(slate / "outputs" / "props_raw_compact.csv")
    assert "Deshaun Watson" not in set(compact["player"])
    # the rostered reception-yards rows survive untouched
    assert int((compact["market"] == "player_reception_yds").sum()) == 20
