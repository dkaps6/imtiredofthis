from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.utils import current_roles_v1 as cr


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _rows():
    return [
        {"team":"IND","player":"Alpha Back","player_clean_key":"alphaback","player_key":"alphaback","role":"RB1","position":"RB"},
        {"team":"IND","player":"Beta Back","player_clean_key":"betaback","player_key":"betaback","role":"RB2","position":"RB"},
    ]


def test_default_resolver_preserves_raw_roles(monkeypatch, tmp_path):
    raw = tmp_path / "roles_ourlads.csv"
    _write(raw, _rows())
    monkeypatch.delenv("ACTIVE_ROLES_CSV", raising=False)
    monkeypatch.setattr(cr, "RAW_ROLES", raw)
    assert cr.resolve_current_roles_path(require_active=False) == raw
    got = cr.load_current_roles(require_active=False)
    assert list(got.player_clean_key) == ["alphaback", "betaback"]


def test_active_override_is_explicit_and_removes_unavailable_identity(monkeypatch, tmp_path):
    raw = tmp_path / "roles_ourlads.csv"
    active = tmp_path / "roles_ourlads_active_v1.csv"
    _write(raw, _rows())
    _write(active, [_rows()[1] | {"role":"RB1"}])
    monkeypatch.setenv("ACTIVE_ROLES_CSV", str(active))
    monkeypatch.setattr(cr, "RAW_ROLES", raw)
    got = cr.load_current_roles(require_active=True)
    assert list(got.player_clean_key) == ["betaback"]
    assert list(got.role) == ["RB1"]
    assert "alphaback" not in set(got.player_clean_key)


def test_require_active_fails_closed_when_artifact_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("ACTIVE_ROLES_CSV", raising=False)
    monkeypatch.setattr(cr, "DEFAULT_ACTIVE_ROLES", tmp_path / "missing.csv")
    with pytest.raises(RuntimeError, match="active reconciled roles missing/empty"):
        cr.resolve_current_roles_path(require_active=True)


def test_explicit_override_never_falls_back_to_raw(monkeypatch, tmp_path):
    raw = tmp_path / "roles_ourlads.csv"
    _write(raw, _rows())
    monkeypatch.setattr(cr, "RAW_ROLES", raw)
    monkeypatch.setenv("ACTIVE_ROLES_CSV", str(tmp_path / "missing-active.csv"))
    with pytest.raises(RuntimeError, match="active reconciled roles missing/empty"):
        cr.resolve_current_roles_path(require_active=False)
