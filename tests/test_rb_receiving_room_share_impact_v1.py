from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.simulation_v2 import SimulationResult
import scripts.research.run_rb_receiving_room_share_impact_v1 as impact


def test_hybrid_changes_only_rb_receiving_and_recomputes_combo():
    base=SimulationResult(
        values={
            ("G1","rb1","receptions"):np.array([1.,2.]),
            ("G1","rb1","rec_yards"):np.array([10.,20.]),
            ("G1","rb1","rush_att"):np.array([8.,9.]),
            ("G1","rb1","rush_yards"):np.array([40.,50.]),
            ("G1","rb1","rush_rec_yards"):np.array([50.,70.]),
            ("G1","wr1","rec_yards"):np.array([70.,80.]),
        },
        iterations=2,
    )
    cand=SimulationResult(
        values={
            ("G1","rb1","receptions"):np.array([3.,4.]),
            ("G1","rb1","rec_yards"):np.array([30.,35.]),
            ("G1","rb1","rush_att"):np.array([1.,2.]),
            ("G1","rb1","rush_yards"):np.array([5.,6.]),
            ("G1","rb1","rush_rec_yards"):np.array([35.,41.]),
            ("G1","wr1","rec_yards"):np.array([1.,2.]),
        },
        iterations=2,
    )
    out=impact._hybrid_receiving_only(base,cand,{("G1","rb1")})
    assert np.array_equal(out.values[("G1","rb1","receptions")],np.array([3.,4.]))
    assert np.array_equal(out.values[("G1","rb1","rec_yards")],np.array([30.,35.]))
    assert np.array_equal(out.values[("G1","rb1","rush_att")],np.array([8.,9.]))
    assert np.array_equal(out.values[("G1","rb1","rush_yards")],np.array([40.,50.]))
    assert np.array_equal(out.values[("G1","rb1","rush_rec_yards")],np.array([70.,85.]))
    # Non-RB remains exact baseline.
    assert np.array_equal(out.values[("G1","wr1","rec_yards")],np.array([70.,80.]))


def test_compare_point_counts_candidate_baseline_and_ties():
    base=pd.DataFrame([
        {"season":2026,"week":1,"event_id":"G","team":"IND","player_clean_key":"a","market":"rec_yards","player":"A","position_family":"RB","projection_mean":20.0,"actual":30.0,"actual_opportunities":5},
        {"season":2026,"week":1,"event_id":"G","team":"IND","player_clean_key":"b","market":"rec_yards","player":"B","position_family":"RB","projection_mean":20.0,"actual":10.0,"actual_opportunities":2},
        {"season":2026,"week":1,"event_id":"G","team":"IND","player_clean_key":"c","market":"rec_yards","player":"C","position_family":"RB","projection_mean":15.0,"actual":15.0,"actual_opportunities":3},
    ])
    cand=base.copy()
    cand["projection_mean"]=[26.0,25.0,15.0]
    out=impact._compare_point(base,cand)
    assert list(out.winner)==["CANDIDATE","BASELINE","TIE"]
    m=impact._metrics(out,"baseline_projection","candidate_projection","actual")
    assert m["candidate_closer"]==1
    assert m["baseline_closer"]==1
    assert m["ties"]==1


def test_compare_targets_preserves_exact_identity():
    b=pd.DataFrame([{
        "week":1,"event_id":"G","team":"IND","player_clean_key":"a",
        "player":"A","position_family":"RB","predicted_opportunities":3.0,
        "actual_opportunities":5.0,
    }])
    c=b.copy(); c["predicted_opportunities"]=4.0
    out=impact._compare_targets(b,c)
    assert out.iloc[0].abs_error_improvement==pytest.approx(1.0)
    assert out.iloc[0].winner=="CANDIDATE"
