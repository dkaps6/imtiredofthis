import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "audit_prism_wr_te_opportunity_bias_conservation_bridge_v1_source_recovery.py"
spec = importlib.util.spec_from_file_location("bridge_recovery", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)


def _signal(position, season, players):
    rows = []
    for i, p in enumerate(players):
        rows.append({
            "position": position,
            "component": "OPPORTUNITY",
            "season": season,
            "week": 5,
            "team": "AAA",
            "player_clean_key": p,
            "source_eligible": True,
            "hierarchical_signed_bias": float(i + 1),
        })
    return pd.DataFrame(rows)


def test_te_audit_excludes_single_player_room_but_keeps_multi_player_room(monkeypatch):
    monkeypatch.setitem(m.SUPPORT, ("TE", 2024), {"rows": 1, "players": 1})
    sig = pd.concat([
        _signal("TE", 2024, ["alpha"]),
        pd.DataFrame([{
            "position": "TE", "component": "OPPORTUNITY", "season": 2024,
            "week": 6, "team": "BBB", "player_clean_key": "solo",
            "source_eligible": True, "hierarchical_signed_bias": -2.0,
        }]),
    ], ignore_index=True)
    direct = pd.DataFrame([
        {"family":"TE_TARGET_OPPORTUNITY","season":2024,"week":5,"team":"AAA","player_clean_key":"alpha","baseline_pred":5.0},
        {"family":"TE_TARGET_OPPORTUNITY","season":2024,"week":5,"team":"AAA","player_clean_key":"bravo","baseline_pred":3.0},
        {"family":"TE_TARGET_OPPORTUNITY","season":2024,"week":6,"team":"BBB","player_clean_key":"solo","baseline_pred":4.0},
    ])
    market = pd.DataFrame([
        {"position":"TE","market":"rec_yards","season":2024,"week":5,"team":"AAA","player_clean_key":"alpha","baseline_proj":40.0},
        {"position":"TE","market":"rec_yards","season":2024,"week":5,"team":"AAA","player_clean_key":"bravo","baseline_proj":21.0},
        {"position":"TE","market":"rec_yards","season":2024,"week":6,"team":"BBB","player_clean_key":"solo","baseline_proj":30.0},
    ])
    rec, eligible, full = m.audit_te(2024, sig, direct, market)
    assert len(full) == 3
    assert rec["signal_rows"] == 2
    assert rec["single_player_signal_rows"] == 1
    assert rec["eligible_rows"] == 1
    assert eligible.iloc[0]["player_clean_key"] == "alpha"
    assert eligible.iloc[0]["bias_factor"] == pytest.approx(
        __import__("math").exp(-1.0 / 40.0)
    )


def test_wr_audit_requires_exact_2024_preserved_target_parity(monkeypatch):
    monkeypatch.setitem(m.SUPPORT, ("WR", 2024), {"rows": 1, "players": 1})
    sig = _signal("WR", 2024, ["alpha"])
    pred = pd.DataFrame([{
        "season": 2024, "week": 5, "team": "AAA", "player_clean_key": "alpha",
        "pred_targets": 8.0, "pred_rec_yards": 64.0,
    }])
    topo = pd.DataFrame([
        {"season":2024,"week":5,"team":"AAA","player_clean_key":"alpha",
         "candidate_entitlement_tgt_share":0.6,"room_size":2,
         "room_entitlement_mass":1.0,"authority_row_type":"FROZEN_ANCHOR_IDENTITY_COMPLEMENT_BRIDGE"},
        {"season":2024,"week":5,"team":"AAA","player_clean_key":"bravo",
         "candidate_entitlement_tgt_share":0.4,"room_size":2,
         "room_entitlement_mass":1.0,"authority_row_type":"FROZEN_SECONDARY_FEATURE_ROW"},
    ])
    direct = pd.DataFrame([{
        "family":"WR_TARGET_OPPORTUNITY","season":2024,"week":5,"team":"AAA",
        "player_clean_key":"alpha","baseline_pred":8.0,
    }])
    rec, eligible = m.audit_wr(2024, sig, pred, topo, direct)
    assert rec["source_ready"] is True
    assert rec["wr2024_target_parity_max_gap"] == pytest.approx(0.0)
    assert eligible.iloc[0]["bias_factor"] == pytest.approx(
        __import__("math").exp(-1.0 / 64.0)
    )

    bad = direct.copy()
    bad.loc[0, "baseline_pred"] = 8.5
    rec_bad, _ = m.audit_wr(2024, sig, pred, topo, bad)
    assert rec_bad["source_gates"]["preserved_wr2024_target_parity"] is False
    assert rec_bad["source_ready"] is False


def test_wr_audit_does_not_infer_flat_target_pool_from_entitlement(monkeypatch):
    monkeypatch.setitem(m.SUPPORT, ("WR", 2023), {"rows": 1, "players": 1})
    sig = _signal("WR", 2023, ["alpha"])
    pred = pd.DataFrame([{
        "season": 2023, "week": 5, "team": "AAA", "player_clean_key": "alpha",
        "pred_targets": 7.25, "pred_rec_yards": 58.0,
    }])
    topo = pd.DataFrame([
        {"season":2023,"week":5,"team":"AAA","player_clean_key":"alpha",
         "candidate_entitlement_tgt_share":0.47,"room_size":2,
         "room_entitlement_mass":0.8,"authority_row_type":"FROZEN_SECONDARY_FEATURE_ROW"},
        {"season":2023,"week":5,"team":"AAA","player_clean_key":"bravo",
         "candidate_entitlement_tgt_share":0.33,"room_size":2,
         "room_entitlement_mass":0.8,"authority_row_type":"FROZEN_SECONDARY_FEATURE_ROW"},
    ])
    rec, eligible = m.audit_wr(2023, sig, pred, topo, pd.DataFrame())
    assert rec["source_ready"] is True
    assert "room_target_sum" not in eligible.columns
    assert eligible.iloc[0]["pred_targets"] == pytest.approx(7.25)
