"""Regression guard for a real production incident (2026-09-13, live run
34776119581): the rush+receiving conservation check is a P3-specific
guarantee (P3 anchors rush_rec_yards to its own synthesized rush mean, so
the two are conserved by construction). A player whose team fell back to
the generic model (see RB P3 team-scope fallback) prices rush_yds and
rush_rec_yds from two independently calibrated ensembles with no such
reconciliation ever promised -- asserting exact conservation for them is a
false positive, not a real bug.
"""
import json

import pandas as pd
import pytest

import scripts.validate_rb_rush_rec_conservation_v1 as validate_mod


def _write_conservation_audit(path):
    path.write_text(
        json.dumps({
            "disposition": "RB_RUSH_REC_DISTRIBUTION_CONSERVED_WITH_PROMOTED_P3",
            "sportsbook_inputs_used": False,
        }),
        encoding="utf-8",
    )


def _priced_row(player, team, market, side, model_proj, rb_synthesis_applied):
    return {
        "player": player,
        "team": team,
        "source_market": market,
        "side": side,
        "model_proj": model_proj,
        "rb_synthesis_applied": rb_synthesis_applied,
    }


def _write_priced(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _run(tmp_path, monkeypatch, rows):
    priced_path = tmp_path / "props_priced_clean.csv"
    conservation_path = tmp_path / "conservation_audit.json"
    out_path = tmp_path / "final_audit.csv"
    _write_priced(priced_path, rows)
    _write_conservation_audit(conservation_path)
    monkeypatch.setattr(validate_mod, "PRICED", priced_path)
    monkeypatch.setattr(validate_mod, "CONSERVATION", conservation_path)
    monkeypatch.setattr(validate_mod, "OUT", out_path)
    return validate_mod.main()


def test_p3_covered_player_with_exact_conservation_passes(tmp_path, monkeypatch, capsys):
    rows = [
        _priced_row("Bijan Robinson", "ATL", "player_rush_yds", "OVER", 50.0, 1),
        _priced_row("Bijan Robinson", "ATL", "player_reception_yds", "OVER", 20.0, 0),
        _priced_row("Bijan Robinson", "ATL", "player_rush_reception_yds", "OVER", 70.0, 0),
    ]
    assert _run(tmp_path, monkeypatch, rows) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1].split("[rb_rush_rec_final] ", 1)[1])
    assert out["players_checked"] == 1
    assert out["players_outside_p3_scope_skipped"] == 0


def test_p3_covered_player_with_a_real_gap_fails(tmp_path, monkeypatch):
    rows = [
        _priced_row("Bijan Robinson", "ATL", "player_rush_yds", "OVER", 50.0, 1),
        _priced_row("Bijan Robinson", "ATL", "player_reception_yds", "OVER", 20.0, 0),
        _priced_row("Bijan Robinson", "ATL", "player_rush_reception_yds", "OVER", 89.3, 0),
    ]
    with pytest.raises(RuntimeError, match="conservation failed"):
        _run(tmp_path, monkeypatch, rows)


def test_player_outside_p3_scope_is_not_asserted_against(tmp_path, monkeypatch, capsys):
    """The exact scenario from live run 34776119581: Tony Pollard (TEN) priced
    rush_yds from the generic model (rb_synthesis_applied=0) while
    rush_rec_yds still came from an independent calibration -- a real,
    expected gap that must not fail the run."""
    rows = [
        _priced_row("Tony Pollard", "TEN", "player_rush_yds", "OVER", 63.36, 0),
        _priced_row("Tony Pollard", "TEN", "player_reception_yds", "OVER", 10.0, 0),
        _priced_row("Tony Pollard", "TEN", "player_rush_reception_yds", "OVER", 55.85, 0),
    ]
    assert _run(tmp_path, monkeypatch, rows) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1].split("[rb_rush_rec_final] ", 1)[1])
    assert out["players_checked"] == 0
    assert out["players_outside_p3_scope_skipped"] == 1


def test_mixed_p3_and_non_p3_players_only_asserts_p3_covered_ones(tmp_path, monkeypatch, capsys):
    rows = [
        _priced_row("Bijan Robinson", "ATL", "player_rush_yds", "OVER", 50.0, 1),
        _priced_row("Bijan Robinson", "ATL", "player_reception_yds", "OVER", 20.0, 0),
        _priced_row("Bijan Robinson", "ATL", "player_rush_reception_yds", "OVER", 70.0, 0),
        _priced_row("Tony Pollard", "TEN", "player_rush_yds", "OVER", 63.36, 0),
        _priced_row("Tony Pollard", "TEN", "player_reception_yds", "OVER", 10.0, 0),
        _priced_row("Tony Pollard", "TEN", "player_rush_reception_yds", "OVER", 55.85, 0),
    ]
    assert _run(tmp_path, monkeypatch, rows) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1].split("[rb_rush_rec_final] ", 1)[1])
    assert out["players_checked"] == 1
    assert out["players_outside_p3_scope_skipped"] == 1


def test_missing_rb_synthesis_applied_column_raises(tmp_path, monkeypatch):
    priced_path = tmp_path / "props_priced_clean.csv"
    conservation_path = tmp_path / "conservation_audit.json"
    pd.DataFrame([{"player": "X", "team": "ATL", "source_market": "player_rush_yds", "side": "OVER", "model_proj": 1.0}]).to_csv(
        priced_path, index=False
    )
    _write_conservation_audit(conservation_path)
    monkeypatch.setattr(validate_mod, "PRICED", priced_path)
    monkeypatch.setattr(validate_mod, "CONSERVATION", conservation_path)
    monkeypatch.setattr(validate_mod, "OUT", tmp_path / "final_audit.csv")
    with pytest.raises(RuntimeError, match="missing conservation columns"):
        validate_mod.main()
