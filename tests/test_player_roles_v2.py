import pandas as pd

from scripts.run_player_form_v2 import _assign_model_roles


def test_wr_model_roles_follow_usage_not_left_right_alignment():
    frame = pd.DataFrame([
        {"player": "Wide A", "team": "BUF", "position": "RWR", "role": "WR3", "tgt_share": 0.30, "rush_share": 0.0, "ypa": None},
        {"player": "Wide B", "team": "BUF", "position": "LWR", "role": "WR1", "tgt_share": 0.18, "rush_share": 0.0, "ypa": None},
        {"player": "Wide C", "team": "BUF", "position": "SWR", "role": "WR2", "tgt_share": 0.22, "rush_share": 0.0, "ypa": None},
    ])
    out = _assign_model_roles(frame)
    roles = dict(zip(out["player"], out["role"]))
    assert roles["Wide A"] == "WR1"
    assert roles["Wide C"] == "WR2"
    assert roles["Wide B"] == "WR3"
    depth = dict(zip(out["player"], out["depth_role"]))
    assert depth["Wide A"] == "WR3"
    assert depth["Wide B"] == "WR1"
    assert set(out["position"]) == {"WR"}


def test_rb_roles_follow_rush_share():
    frame = pd.DataFrame([
        {"player": "Back A", "team": "KC", "position": "RB", "role": "RB1", "tgt_share": 0.05, "rush_share": 0.25, "ypa": None},
        {"player": "Back B", "team": "KC", "position": "RB", "role": "RB2", "tgt_share": 0.10, "rush_share": 0.55, "ypa": None},
    ])
    out = _assign_model_roles(frame)
    roles = dict(zip(out["player"], out["role"]))
    assert roles["Back B"] == "RB1"
    assert roles["Back A"] == "RB2"
