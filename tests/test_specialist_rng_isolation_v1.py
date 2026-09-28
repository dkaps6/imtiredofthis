import numpy as np
import pandas as pd

from scripts.research.audit_specialist_rng_isolation_v1 import (
    _allocate_room,
    _hierarchical_targets,
    _stable_seed,
)


def test_stable_seed_is_semantic_and_repeatable():
    a = _stable_seed(42, "TARGET_TOP", "game1", "IND")
    b = _stable_seed(42, "TARGET_TOP", "game1", "IND")
    c = _stable_seed(42, "TARGET_TOP", "game1", "CHI")
    assert a == b
    assert a != c


def test_room_allocator_conserves_integer_totals():
    totals = np.array([0, 1, 2, 7, 11, 3], dtype=int)
    shares = np.array([0.2, 0.3, 0.5], dtype=float)
    out = _allocate_room(np.random.default_rng(7), totals, shares)
    assert out.shape == (len(totals), 3)
    assert np.array_equal(out.sum(axis=1), totals)
    assert (out >= 0).all()


def test_hierarchical_targets_protect_outside_player_when_te_room_changes():
    n = 400
    pass_att = np.full(n, 30, dtype=int)
    frame = pd.DataFrame(
        [
            {"event_id": "g", "team": "T", "player_clean_key": "protected", "position": "WR"},
            {"event_id": "g", "team": "T", "player_clean_key": "te_a", "position": "TE"},
            {"event_id": "g", "team": "T", "player_clean_key": "te_b", "position": "TE"},
        ]
    )
    plan = {
        "te_changed": {("g", "T", "te_a"), ("g", "T", "te_b")},
        "wr_changed": set(),
        "authority": {
            ("g", "T", "protected"): 0.20,
            ("g", "T", "te_a"): 0.10,
            ("g", "T", "te_b"): 0.10,
        },
    }
    left, _ = _hierarchical_targets(
        game="g", team="T", team_df=frame, pass_att=pass_att,
        stage_shares=np.array([0.20, 0.10, 0.10]), plan=plan, base_seed=42,
    )
    right, _ = _hierarchical_targets(
        game="g", team="T", team_df=frame, pass_att=pass_att,
        stage_shares=np.array([0.20, 0.16, 0.04]), plan=plan, base_seed=42,
    )
    assert np.array_equal(left[:, 0], right[:, 0])
    assert not np.array_equal(left[:, 1:], right[:, 1:])
    assert np.array_equal(left.sum(axis=1), right.sum(axis=1))


def test_hierarchical_targets_protect_wr_anchor_when_wr_room_changes():
    n = 400
    pass_att = np.full(n, 32, dtype=int)
    frame = pd.DataFrame(
        [
            {"event_id": "g", "team": "T", "player_clean_key": "wr1", "position": "WR"},
            {"event_id": "g", "team": "T", "player_clean_key": "wr2", "position": "WR"},
            {"event_id": "g", "team": "T", "player_clean_key": "wr3", "position": "WR"},
        ]
    )
    plan = {
        "te_changed": set(),
        "wr_changed": {("g", "T", "wr2"), ("g", "T", "wr3")},
        "authority": {
            ("g", "T", "wr1"): 0.24,
            ("g", "T", "wr2"): 0.14,
            ("g", "T", "wr3"): 0.08,
        },
    }
    left, _ = _hierarchical_targets(
        game="g", team="T", team_df=frame, pass_att=pass_att,
        stage_shares=np.array([0.24, 0.14, 0.08]), plan=plan, base_seed=42,
    )
    right, _ = _hierarchical_targets(
        game="g", team="T", team_df=frame, pass_att=pass_att,
        stage_shares=np.array([0.24, 0.18, 0.04]), plan=plan, base_seed=42,
    )
    assert np.array_equal(left[:, 0], right[:, 0])
    assert not np.array_equal(left[:, 1:], right[:, 1:])
