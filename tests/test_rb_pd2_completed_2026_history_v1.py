from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research import build_rb_pd2_completed_2026_history_v1 as w1
from scripts.research import rb_pd2_forward_shadow_v1 as fwd


def _context():
    rows=[]
    for i in range(w1.WEEK1_EXPECTED_ROWS):
        rows.append({
            "season": 2026, "week": 1, "event_id": f"E{i//4}",
            "player": f"Back {i}", "player_clean_key": f"back{i}",
            "team": "CHI" if i%2==0 else "CAR",
            "opponent": "GB" if i%2==0 else "ATL",
            "position": "RB", "stack_yards": 20.0+i/10,
            "rb_synthesis_proj": 20.0+i/10,
            "rb_synthesis_route": w1.WEEK1_ROUTE,
            "rb_synthesis_version": w1.WEEK1_VERSION,
            "rb_synthesis_applied": 1,
            "football_only_no_odds": 1,
            "sportsbook_inputs_used": 0,
            "rush_yards_ensemble_weight_mc": fwd.RUSH_YARDS_MC_WEIGHT,
            "rush_yards_ensemble_weight_ml": fwd.RUSH_YARDS_ML_WEIGHT,
            "rush_yards_ensemble_weight_state": 0.0,
        })
    return pd.DataFrame(rows)


def test_week1_projection_freezes_exact_p3_stack1_parent():
    out,audit=w1.build_week1_projection_frame(_context())
    assert len(out)==107
    assert out["pregame_lineage_certified"].all()
    assert out["week1_p3_stack1_parity_pass"].all()
    assert audit["max_p3_stack1_abs_diff"] == 0.0
    assert audit["sportsbook_inputs_used"] == 0


def test_week1_projection_rejects_sportsbook_or_parity_drift():
    x=_context()
    x.loc[0,"sportsbook_inputs_used"]=1
    with pytest.raises(RuntimeError,match="sportsbook"):
        w1.build_week1_projection_frame(x)

    x=_context()
    x.loc[0,"rb_synthesis_proj"] += 1.0
    with pytest.raises(RuntimeError,match="parity"):
        w1.build_week1_projection_frame(x)


def test_attach_actuals_uses_stats_then_roster_verified_zero():
    projections=pd.DataFrame([
        {"season":2026,"week":1,"event_id":"E1","player":"A","player_clean_key":"alpha","team":"CHI","opponent":"GB","position":"RB","projection_mean":30.0,"pregame_lineage_certified":True,"projection_lineage":"x","week1_p3_stack1_parity_pass":True},
        {"season":2026,"week":1,"event_id":"E2","player":"B","player_clean_key":"bravo","team":"CAR","opponent":"ATL","position":"RB","projection_mean":20.0,"pregame_lineage_certified":True,"projection_lineage":"x","week1_p3_stack1_parity_pass":True},
    ])
    actual=pd.DataFrame([
        {"season":2026,"week":1,"team":"CHI","gsis_id":"00-A","player":"A","player_clean_key":"alpha","receptions":0.0,"rec_yards":0.0,"rush_yards":44.0,"pass_yards":0.0},
    ])
    roster=pd.DataFrame([
        {"season":2026,"week":1,"team":"CHI","gsis_id":"00-A","player_clean_key":"alpha","status":"ACT"},
        {"season":2026,"week":1,"team":"CAR","gsis_id":"00-B","player_clean_key":"bravo","status":"INA"},
    ])
    out,audit=w1.attach_verified_actuals(projections,actual,roster)
    got=dict(zip(out.player_clean_key,out.actual_rush_yards))
    assert got=={"alpha":44.0,"bravo":0.0}
    assert audit["verified_stats_table"]==1
    assert audit["verified_roster_zero"]==1
    assert audit["excluded_rows"]==0
