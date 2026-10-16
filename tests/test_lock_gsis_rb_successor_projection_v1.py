import pandas as pd
import scripts.research.lock_gsis_rb_successor_projection_v1 as mod
from scripts.research.lock_gsis_rb_successor_projection_v1 import apply_transfer_arm

def _prepared():
    return pd.DataFrame([
      {"team":"DEN","player_clean_key":"a","rules_rush_share":0.3,"rules_ypc":4.2,"other":"x"},
      {"team":"DEN","player_clean_key":"b","rules_rush_share":0.2,"rules_ypc":4.5,"other":"y"},
    ])

def _lock():
    return pd.DataFrame([
      {"team":"DEN","successor_player_clean_key":"a","snap_transfer_rush_share":0.08,"gsis_transfer_rush_share":0.15},
      {"team":"DEN","successor_player_clean_key":"b","snap_transfer_rush_share":0.12,"gsis_transfer_rush_share":0.05},
    ])

def test_transfer_arm_changes_only_rush_share():
    base=_prepared()
    out=apply_transfer_arm(base,_lock(),transfer_col="gsis_transfer_rush_share",label="GSIS")
    assert abs(out.loc[out.player_clean_key.eq("a"),"rules_rush_share"].iloc[0]-.45)<1e-12
    assert abs(out.loc[out.player_clean_key.eq("b"),"rules_rush_share"].iloc[0]-.25)<1e-12
    assert out.rules_ypc.equals(base.rules_ypc)
    assert out.other.equals(base.other)

def test_zero_transfer_still_requires_active_successor_identity():
    lock=_lock().copy()
    lock.loc[lock.player_clean_key.eq("b") if "player_clean_key" in lock.columns else lock.successor_player_clean_key.eq("b"),"gsis_transfer_rush_share"]=0.0
    out=apply_transfer_arm(_prepared(),lock,transfer_col="gsis_transfer_rush_share",label="GSIS")
    assert abs(out.loc[out.player_clean_key.eq("b"),"rules_rush_share"].iloc[0]-.2)<1e-12

def test_projection_uses_promoted_production_simulator(monkeypatch):
    sentinel=object()
    seen={}
    def fake(frame,*,iterations=None,seed=None,allocation_trace=None):
        seen["rows"]=len(frame)
        seen["iterations"]=iterations
        seen["seed"]=seed
        seen["trace"]=allocation_trace
        return sentinel
    monkeypatch.setattr(mod.production_v3,"_simulate_promoted_stack",fake)
    trace=[]
    frame=pd.DataFrame([{"team":"DEN","player_clean_key":"a"}])
    got=mod._simulate_current_week_gt1_production(
        frame,iterations=25000,seed=42,allocation_trace=trace
    )
    assert got is sentinel
    assert seen=={"rows":1,"iterations":25000,"seed":42,"trace":trace}
