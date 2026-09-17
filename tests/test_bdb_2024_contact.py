import pandas as pd
import pytest

from scripts.data_frontier import bdb_2024_contact as m


def _tracking(play_direction="right", simultaneous=False):
    rows = []
    carrier_x = [30, 31, 32, 33, 34]
    defender1_x = [33, 32.5, 32.8, 33.4, 34.2]
    defender2_x = [40, 39, 38, 37, 36]
    if simultaneous:
        defender2_x = [35, 34, 32.7, 33.3, 34.1]
    for fr, cx, d1x, d2x in zip(range(1, 6), carrier_x, defender1_x, defender2_x):
        event = "handoff" if fr == 1 else ("tackle" if fr == 5 else "")
        rows.append(
            dict(
                gameId=1,
                playId=10,
                frameId=fr,
                playDirection=play_direction,
                x=cx if play_direction == "right" else 120 - cx,
                y=20,
                club="O",
                nflId=100,
                displayName="RB",
                s=5,
                a=0,
                event=event,
            )
        )
        rows.append(
            dict(
                gameId=1,
                playId=10,
                frameId=fr,
                playDirection=play_direction,
                x=d1x if play_direction == "right" else 120 - d1x,
                y=20,
                club="D",
                nflId=200,
                displayName="LB1",
                s=4,
                a=0,
                event="",
            )
        )
        rows.append(
            dict(
                gameId=1,
                playId=10,
                frameId=fr,
                playDirection=play_direction,
                x=d2x if play_direction == "right" else 120 - d2x,
                y=20,
                club="D",
                nflId=201,
                displayName="LB2",
                s=4,
                a=0,
                event="",
            )
        )
    return pd.DataFrame(rows)


def _plays():
    return pd.DataFrame(
        [
            dict(
                gameId=1,
                playId=10,
                ballCarrierId=100,
                possessionTeam="O",
                defensiveTeam="D",
            )
        ]
    )


def _tackles():
    return pd.DataFrame(
        [
            dict(
                gameId=1,
                playId=10,
                nflId=200,
                tackle=1,
                assist=0,
                forcedFumble=0,
                pff_missedTackle=0,
            ),
            dict(
                gameId=1,
                playId=10,
                nflId=201,
                tackle=0,
                assist=0,
                forcedFumble=0,
                pff_missedTackle=1,
            ),
        ]
    )


def test_coordinate_mirror_equivalence():
    right = m.offense_x(pd.Series([30.0]), pd.Series(["right"])).iloc[0]
    left = m.offense_x(pd.Series([90.0]), pd.Series(["left"])).iloc[0]
    assert right == left == 30.0


def test_contact_requires_two_consecutive_frames():
    tracking = m.normalize_tracking(_tracking())
    plays = m.normalize_plays(_plays())
    tackles = m.normalize_tackles(_tackles())
    features, dispositions = m.derive_contact_features(tracking, plays, tackles)
    assert dispositions.loc[0, "benchmark_disposition"] == "SCOREABLE"
    assert features.loc[0, "firstContactFrameId"] == 3
    assert features.loc[0, "firstContactDefenderIds"] == "200"
    assert features.loc[0, "firstContactSourceLabelOverlap"] == 1
    assert features.loc[0, "yardsBeforeContactGeom"] == pytest.approx(2.0)


def test_left_and_right_derive_same_geometry():
    out = []
    for direction in ["right", "left"]:
        features, _ = m.derive_contact_features(
            m.normalize_tracking(_tracking(direction)),
            m.normalize_plays(_plays()),
            m.normalize_tackles(_tackles()),
        )
        out.append(features.iloc[0])
    assert out[0]["firstContactXOffense"] == pytest.approx(out[1]["firstContactXOffense"])
    assert out[0]["yardsBeforeContactGeom"] == pytest.approx(out[1]["yardsBeforeContactGeom"])
    assert out[0]["yardsAfterFirstContactGeom"] == pytest.approx(
        out[1]["yardsAfterFirstContactGeom"]
    )


def test_simultaneous_contact_preserved():
    features, _ = m.derive_contact_features(
        m.normalize_tracking(_tracking(simultaneous=True)),
        m.normalize_plays(_plays()),
        m.normalize_tackles(_tackles()),
    )
    assert features.loc[0, "firstContactFrameId"] == 3
    assert set(features.loc[0, "firstContactDefenderIds"].split("|")) == {"200", "201"}
    assert features.loc[0, "simultaneousFirstContactDefenders"] == 2


def test_duplicate_player_frame_fails_closed():
    raw = _tracking()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate player-frame"):
        m.normalize_tracking(raw)


def test_no_contact_is_visible_abstention():
    raw = _tracking()
    raw.loc[raw["club"].eq("D"), "x"] = 80
    tracking = m.normalize_tracking(raw)
    features, dispositions = m.derive_contact_features(
        tracking,
        m.normalize_plays(_plays()),
        m.normalize_tackles(_tackles()),
    )
    assert features.empty
    assert dispositions.loc[0, "benchmark_disposition"] == "NO_CONTACT_EVENT_RESOLUTION"


def test_qa_explicitly_proves_no_prediction_or_sportsbook_use():
    tracking = m.normalize_tracking(_tracking())
    features, dispositions = m.derive_contact_features(
        tracking,
        m.normalize_plays(_plays()),
        m.normalize_tackles(_tackles()),
    )
    qa = m.build_qa(features, dispositions, {"files": []})
    assert qa["predictive_metrics_computed"] is False
    assert qa["sportsbook_inputs_used"] is False
    assert qa["production_changed"] is False
    assert qa["attempted_plays"] == 1
