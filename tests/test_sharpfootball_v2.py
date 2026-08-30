import pandas as pd

import scripts.providers.sharpfootball_pull as sharp
from scripts.run_sharpfootball_v2 import normalize_pace_table_v2, rename_expected_cols_v2


def test_live_2026_sharp_pace_header_maps_to_canonical_neutral_pace():
    raw = pd.DataFrame({
        "#": [1, 2],
        "Offense": ["Arizona Cardinals", "Los Angeles Rams"],
        "Play Clock Used (Sec/Play)": [27.1, 28.2],
        "Neutral Script (Sec/Play)": [29.4, 30.1],
        "Gear Change (Sec/Play)": [-2.3, -1.9],
        "No Huddle (% of Plays)": [12.0, 8.0],
        "Plays per Game": [63.0, 65.0],
    })
    out = normalize_pace_table_v2(raw)
    assert list(out["team"]) == ["Arizona Cardinals", "Los Angeles Rams"]
    assert list(out["neutral_pace"]) == [29.4, 30.1]

    normalized = sharp._normalize_team_col(out)
    assert set(normalized["team"]) == {"ARI", "LAR"}


def test_sharp_pace_adapter_prefers_seconds_per_play_not_neutral_pass_rate():
    raw = pd.DataFrame({
        "Offense": ["Buffalo Bills"],
        "Neutral DB Rate": [61.2],
        "Neutral Script (Sec/Play)": [27.9],
    })
    out = normalize_pace_table_v2(raw)
    assert out.loc[0, "neutral_pace"] == 27.9


def test_pace_alias_normalization_is_idempotent_and_preserves_canonical_column():
    frame = pd.DataFrame({
        "team": ["NO", "DAL"],
        "neutral_pace": [29.11, 30.22],
        "plays_per_game": [63.0, 65.0],
    })
    once = rename_expected_cols_v2("pace", frame)
    twice = rename_expected_cols_v2("pace", once)

    assert "neutral_pace" in once.columns
    assert "neutral_pace" in twice.columns
    assert list(twice["neutral_pace"]) == [29.11, 30.22]


def test_pace_alias_normalization_keeps_last5_distinct_from_neutral_pace():
    frame = pd.DataFrame({
        "team": ["BUF"],
        "neutral_pace": [28.4],
        "neutral_pace_last5": [27.6],
    })
    out = rename_expected_cols_v2("pace", rename_expected_cols_v2("pace", frame))
    assert out.loc[0, "neutral_pace"] == 28.4
    assert out.loc[0, "neutral_pace_last5"] == 27.6
