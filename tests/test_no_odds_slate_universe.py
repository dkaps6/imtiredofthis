from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.artifact_io import read_valid_csv
from scripts.slate_universe_v2 import build_slate_universe


def _fake_pf(tmp_path: Path):
    props = tmp_path / "props_raw.csv"
    props.write_text("\n", encoding="utf-8")  # tracked-placeholder equivalent
    data = tmp_path / "data"
    data.mkdir()

    roles = pd.DataFrame({
        "display_name": ["Player A", "Player B"],
        "player_clean_key": ["playera", "playerb"],
        "team": ["DAL", "PHI"],
        "position": ["WR", "WR"],
        "role": ["WR1", "WR1"],
    })

    return SimpleNamespace(
        PROPS=props,
        DATA=data,
        _load_roles=lambda: roles.copy(),
        _canon_name=lambda value: (str(value), str(value).lower().replace(" ", "")),
        _first=lambda frame, cols, default="": next((frame[c] for c in cols if c in frame.columns), pd.Series(default, index=frame.index)),
    )


def _schedule():
    return pd.DataFrame({
        "season": [2026, 2026],
        "week": [1, 1],
        "team": ["DAL", "PHI"],
        "opponent": ["PHI", "DAL"],
    })


def test_no_odds_mode_ignores_invalid_props_and_uses_roster_schedule(tmp_path):
    pf = _fake_pf(tmp_path)
    out = build_slate_universe(
        pf,
        _schedule,
        2026,
        1,
        live_odds_enabled=False,
    )
    assert len(out) == 2
    assert set(out["team"]) == {"DAL", "PHI"}
    assert dict(zip(out["team"], out["opponent"])) == {"DAL": "PHI", "PHI": "DAL"}


def test_live_odds_mode_rejects_invalid_props_placeholder(tmp_path):
    pf = _fake_pf(tmp_path)
    with pytest.raises(RuntimeError, match="Required artifact live props_raw is invalid"):
        build_slate_universe(
            pf,
            _schedule,
            2026,
            1,
            live_odds_enabled=True,
        )


def test_optional_csv_placeholder_returns_none(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("\n", encoding="utf-8")
    assert read_valid_csv(path, required=False) is None
