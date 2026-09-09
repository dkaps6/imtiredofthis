from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.artifact_io import read_valid_csv
from scripts.slate_universe_v2 import build_slate_universe


_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
]


def _fake_pf(tmp_path: Path):
    props = tmp_path / "props_raw.csv"
    props.write_text("\n", encoding="utf-8")  # tracked-placeholder equivalent
    data = tmp_path / "data"
    data.mkdir()

    roles = pd.DataFrame({
        "display_name": [f"Player {i}" for i in range(len(_TEAMS))],
        "player_clean_key": [f"player{i}" for i in range(len(_TEAMS))],
        "team": _TEAMS,
        "position": ["WR"] * len(_TEAMS),
        "role": ["WR1"] * len(_TEAMS),
    })

    return SimpleNamespace(
        PROPS=props,
        DATA=data,
        _load_roles=lambda: roles.copy(),
        _canon_name=lambda value: (str(value), str(value).lower().replace(" ", "")),
        _first=lambda frame, cols, default="": next((frame[c] for c in cols if c in frame.columns), pd.Series(default, index=frame.index)),
    )


def _schedule():
    rows = []
    for i in range(0, len(_TEAMS), 2):
        a, b = _TEAMS[i], _TEAMS[i + 1]
        rows.extend([
            {"season": 2026, "week": 1, "team": a, "opponent": b},
            {"season": 2026, "week": 1, "team": b, "opponent": a},
        ])
    return pd.DataFrame(rows)


def test_no_odds_mode_ignores_invalid_props_and_uses_roster_schedule(tmp_path):
    pf = _fake_pf(tmp_path)
    out = build_slate_universe(
        pf,
        _schedule,
        2026,
        1,
        live_odds_enabled=False,
    )
    assert len(out) == len(_TEAMS)
    assert set(out["team"]) == set(_TEAMS)
    assert out["opponent"].notna().all()


def test_live_odds_mode_does_not_define_playerform_universe(tmp_path):
    pf = _fake_pf(tmp_path)
    without_odds = build_slate_universe(
        pf,
        _schedule,
        2026,
        1,
        live_odds_enabled=False,
    )
    with_odds = build_slate_universe(
        pf,
        _schedule,
        2026,
        1,
        live_odds_enabled=True,
    )
    cols = ["player", "player_clean_key", "team", "opponent", "season", "week"]
    pd.testing.assert_frame_equal(
        without_odds[cols].sort_values(["team", "player_clean_key"]).reset_index(drop=True),
        with_odds[cols].sort_values(["team", "player_clean_key"]).reset_index(drop=True),
    )


def test_optional_csv_placeholder_returns_none(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("\n", encoding="utf-8")
    assert read_valid_csv(path, required=False) is None
