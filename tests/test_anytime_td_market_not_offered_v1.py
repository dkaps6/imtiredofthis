"""Zero anytime-TD rows has two causes and only one of them is a defect.

player_anytime_td was removed from the paid pull on 2026-11-04 by standing
instruction, in favour of the alternate-line ladders. The ATD status audit
still demanded at least one priced ATD row, so 2026 Week 9 replay run
42104253081 failed *after* the whole 1,544-row board was priced, for the
absence of a market nobody asked for.

The fix must not become a blanket excuse: a market that IS configured and
still produces nothing is a real provider/pricing gap and must keep failing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.validate_anytime_td_model_status_v1 as atd_mod


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    """Run the audit against a priced board that carries no ATD rows."""
    outputs = tmp_path / "outputs"
    data = tmp_path / "data"
    outputs.mkdir()
    data.mkdir()
    pd.DataFrame(
        [
            {"source_market": "player_pass_yds", "side": "OVER", "player": "A", "fair_prob": 0.5},
            {"source_market": "player_receptions", "side": "OVER", "player": "B", "fair_prob": 0.5},
        ]
    ).to_csv(outputs / "props_priced_clean.csv", index=False)
    (data / "player_scoring_td_prior_status.json").write_text(
        json.dumps({"sportsbook_inputs_used": False}), encoding="utf-8"
    )
    monkeypatch.setattr(atd_mod, "OUTPUTS", outputs)
    monkeypatch.setattr(atd_mod, "DATA", data)
    monkeypatch.setattr(atd_mod, "OUT_CSV", data / "anytime_td_model_status_v1.csv")
    monkeypatch.setattr(atd_mod, "OUT_JSON", data / "anytime_td_model_status_v1.json")
    return data


def test_absent_market_is_labelled_not_fatal(staged):
    """The real repository config does not pull player_anytime_td."""
    from scripts.fetch_props_oddsapi import MARKETS

    assert "player_anytime_td" not in set(MARKETS), (
        "this test documents the standing removal; if the market is pulled again, "
        "the audit should be strict and this test should be rewritten"
    )
    assert atd_mod.main() == 0
    payload = json.loads((staged / "anytime_td_model_status_v1.json").read_text(encoding="utf-8"))
    assert payload["disposition"] == "ATD_MARKET_NOT_OFFERED_BY_CONFIGURATION"
    assert payload["atd_market_offered"] == 0
    assert payload["priced_atd_book_line_rows"] == 0
    assert payload["priced_atd_players"] == 0
    # The certified-stack validator asserts all three of these.
    assert payload["execution_status"] == "PASS"
    assert payload["current_sportsbook_inputs_used_for_football_probability"] == 0
    assert payload["dedicated_probability_model_certified"] == 0


def test_configured_market_with_no_rows_still_fails_closed(staged, monkeypatch):
    """A requested market that produced nothing is a real gap, not a config choice."""
    monkeypatch.setattr(
        "scripts.fetch_props_oddsapi.MARKETS",
        ["player_pass_yds", "player_anytime_td"],
        raising=True,
    )
    with pytest.raises(RuntimeError, match="IS in the configured market pull"):
        atd_mod.main()


def test_unreadable_market_list_fails_closed(staged, monkeypatch):
    """If the audit cannot tell the two cases apart it must not guess."""
    import builtins

    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "scripts.fetch_props_oddsapi":
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    with pytest.raises(RuntimeError, match="could not be read"):
        atd_mod.main()


def test_stack_validator_rejects_a_mislabelled_not_offered_claim():
    """A not-offered claim is only acceptable alongside zero priced rows."""
    src = Path("scripts/validate_certified_full_slate_stack_v1.py").read_text(encoding="utf-8")
    assert 'atd.get("atd_market_offered", 1)' in src
    assert "claims the market was not offered but reports priced rows" in src
