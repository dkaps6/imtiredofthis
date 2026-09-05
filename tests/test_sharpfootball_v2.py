import pandas as pd
import pytest

import scripts.providers.sharpfootball_pull as sharp
from scripts.run_sharpfootball_v2 import normalize_pace_table_v2, rename_expected_cols_v2


def test_current_sharp_neutral_header_maps_to_canonical_neutral_pace():
    # Live schema observed 2026-09-03.  Sharp defines Neutral as neutral
    # play-clock-used pace; the adjacent Play Clock Used value is all-situation.
    raw = pd.DataFrame({
        "Rank": [1, 2],
        "Offense": ["NO", "DAL"],
        "Play Clock Used": [28.61, 29.40],
        "Neutral": [29.11, 30.22],
        "Neutral Pass Rate": ["56.6%", "57.8%"],
        "Gear Change": [0.52, 2.50],
        "No Huddle": ["23%", "9%"],
        "Plays/Game": [62.4, 64.8],
        "Plays/Drive": [6.17, 6.36],
        "Time of Possession": [27.8, 29.7],
        "Plays": [1061, 1101],
    })
    out = normalize_pace_table_v2(raw)
    assert list(out["team"]) == ["NO", "DAL"]
    assert list(out["neutral_pace"]) == [29.11, 30.22]
    assert list(out["Play Clock Used"]) == [28.61, 29.40]


def test_live_2026_legacy_sharp_pace_header_maps_to_canonical_neutral_pace():
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


def test_current_sharp_neutral_does_not_map_neutral_pass_rate():
    raw = pd.DataFrame({
        "Offense": ["BUF"],
        "Neutral": [33.71],
        "Neutral Pass Rate": ["55.8%"],
    })
    out = normalize_pace_table_v2(raw)
    assert out.loc[0, "neutral_pace"] == 33.71


def test_neutral_pace_semantic_guard_rejects_percentage_like_values():
    raw = pd.DataFrame({"Offense": ["BUF"], "Neutral": [55.8]})
    with pytest.raises(RuntimeError, match="outside play-clock seconds range"):
        normalize_pace_table_v2(raw)


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
