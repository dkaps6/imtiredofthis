from pathlib import Path

import pandas as pd
import pytest

from scripts.operations.apply_current_availability_qb_c2_stale_starter_seam_v1 import OLD, NEW, transform
from scripts.utils.qb_starter_availability_v1 import definitive_unavailable_evidence


def _write(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(**overrides):
    row = {
        "team": "ATL",
        "player": "Tua Tagovailoa",
        "player_clean_key": "tuatagovailoa",
        "definitive_unavailable": 1,
        "final_availability_state": "UNAVAILABLE_REPORTED",
        "availability_authority": "weekly_injury_report",
        "availability_reason": "definitive reported non-participation",
    }
    row.update(overrides)
    return row


def test_definitive_unavailable_authority_player_returns_evidence(tmp_path):
    path = _write(tmp_path / "availability.csv", [_row()])
    evidence = definitive_unavailable_evidence("ATL", "Tua Tagovailoa", path=path)
    assert evidence is not None
    assert evidence["starter_key"] == "tuatagovailoa"
    assert evidence["final_availability_state"] == "UNAVAILABLE_REPORTED"
    assert evidence["sportsbook_inputs_used"] == "0"


def test_available_authority_player_does_not_authorize_fallback(tmp_path):
    path = _write(
        tmp_path / "availability.csv",
        [_row(definitive_unavailable=0, final_availability_state="AVAILABLE_REPORTED")],
    )
    assert definitive_unavailable_evidence("ATL", "Tua Tagovailoa", path=path) is None


def test_missing_authority_identity_fails_closed(tmp_path):
    path = _write(tmp_path / "availability.csv", [_row(player="Cooper Rush", player_clean_key="cooperrush")])
    with pytest.raises(RuntimeError, match="must match exactly once"):
        definitive_unavailable_evidence("ATL", "Tua Tagovailoa", path=path)


def test_duplicate_authority_identity_fails_closed(tmp_path):
    path = _write(tmp_path / "availability.csv", [_row(), _row()])
    with pytest.raises(RuntimeError, match="must match exactly once"):
        definitive_unavailable_evidence("ATL", "Tua Tagovailoa", path=path)


def test_unavailable_flag_state_disagreement_fails_closed(tmp_path):
    path = _write(
        tmp_path / "availability.csv",
        [_row(definitive_unavailable=1, final_availability_state="AVAILABLE_REPORTED")],
    )
    with pytest.raises(RuntimeError, match="flag/state disagreement"):
        definitive_unavailable_evidence("ATL", "Tua Tagovailoa", path=path)


def test_starter_seam_is_single_anchor_and_preserves_fail_closed_path():
    protected = "\n".join([
        "FORBIDDEN_SELECTOR_FIELDS",
        'raise RuntimeError(f"team={team} has ambiguous Ourlads QB1 fallback: {sample}")',
        'if not audit["sportsbook_inputs_used"].eq(0).all():',
    ])
    source = protected + "\nbefore\n" + OLD + "after\n"
    out = transform(source)
    assert OLD not in out
    assert NEW in out
    assert "no definitive unavailable evidence" in out
    assert "availability_reconciled_from_" in out
