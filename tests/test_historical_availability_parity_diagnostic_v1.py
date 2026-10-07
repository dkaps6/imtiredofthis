from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.research.build_historical_availability_parity_universe_v1 as build
import scripts.research.compare_historical_availability_parity_v1 as compare


def test_join_key_is_exact_alphanumeric_normalization():
    assert build._join_key("Michael Pittman Jr.") == "michaelpittmanjr"
    assert build._join_key("A.J. Brown") == "ajbrown"


def test_compare_pairs_exact_act_rows_and_measures_inactive_mass(tmp_path: Path):
    baseline = pd.DataFrame([
        {
            "season": 2026, "week": 1, "event_id": "G1", "team": "IND",
            "player": "Active WR", "player_clean_key": "activewr",
            "position_family": "WR", "opportunity_type": "targets",
            "predicted_opportunities": 4.0, "actual_opportunities": 8.0,
            "actual_opportunity_bin": "06_08", "linked_yards_error": -30.0,
            "linked_count_error": -3.0,
        },
        {
            "season": 2026, "week": 1, "event_id": "G1", "team": "IND",
            "player": "Inactive WR", "player_clean_key": "inactivewr",
            "position_family": "WR", "opportunity_type": "targets",
            "predicted_opportunities": 2.0, "actual_opportunities": 0.0,
            "actual_opportunity_bin": "ZERO", "linked_yards_error": 10.0,
            "linked_count_error": 1.0,
        },
    ])
    act = pd.DataFrame([
        {
            "season": 2026, "week": 1, "event_id": "G1", "team": "IND",
            "player": "Active WR", "player_clean_key": "activewr",
            "position_family": "WR", "opportunity_type": "targets",
            "predicted_opportunities": 6.0, "actual_opportunities": 8.0,
            "actual_opportunity_bin": "06_08", "linked_yards_error": -30.0,
            "linked_count_error": -3.0,
        },
    ])
    status = pd.DataFrame([
        {"season": 2026, "week": 1, "team": "IND", "player": "Active WR", "player_join": "activewr", "status": "ACT"},
        {"season": 2026, "week": 1, "team": "IND", "player": "Inactive WR", "player_join": "inactivewr", "status": "INA"},
    ])

    bp=tmp_path/"b.csv"; ap=tmp_path/"a.csv"; sp=tmp_path/"s.csv"
    gp=tmp_path/"g.csv"; mp=tmp_path/"m.csv"; jp=tmp_path/"j.json"
    baseline.to_csv(bp,index=False); act.to_csv(ap,index=False); status.to_csv(sp,index=False)

    payload=compare.compare(
        baseline_rows_path=bp,
        act_rows_path=ap,
        status_map_path=sp,
        group_out=gp,
        mass_out=mp,
        summary_out=jp,
    )
    assert payload["baseline_ina_rows"] == 1
    assert payload["paired_act_rows"] == 1
    mass=pd.read_csv(mp).iloc[0]
    assert mass["ina_predicted_opportunity_mass"] == pytest.approx(2.0)
    assert mass["ina_share_of_modeled_player_opportunity"] == pytest.approx(1/3)
    group=pd.read_csv(gp).iloc[0]
    assert group["baseline_mae"] == pytest.approx(4.0)
    assert group["act_mae"] == pytest.approx(2.0)
    assert group["mae_improvement_baseline_minus_act"] == pytest.approx(2.0)
    assert group["mean_prediction_delta_act_minus_baseline"] == pytest.approx(2.0)


def test_compare_fails_if_actuals_change_across_variants(tmp_path: Path):
    base = pd.DataFrame([{
        "season":2026,"week":1,"event_id":"G1","team":"IND","player":"A",
        "player_clean_key":"a","position_family":"WR","opportunity_type":"targets",
        "predicted_opportunities":2.0,"actual_opportunities":4.0,
        "actual_opportunity_bin":"03_05","linked_yards_error":-10.0,"linked_count_error":-1.0,
    }])
    act=base.copy(); act["actual_opportunities"]=5.0
    status=pd.DataFrame([{"season":2026,"week":1,"team":"IND","player":"A","player_join":"a","status":"ACT"}])
    bp=tmp_path/"b.csv"; ap=tmp_path/"a.csv"; sp=tmp_path/"s.csv"
    base.to_csv(bp,index=False); act.to_csv(ap,index=False); status.to_csv(sp,index=False)
    with pytest.raises(RuntimeError, match="actual opportunities changed"):
        compare.compare(
            baseline_rows_path=bp, act_rows_path=ap, status_map_path=sp,
            group_out=tmp_path/"g.csv", mass_out=tmp_path/"m.csv", summary_out=tmp_path/"j.json",
        )
