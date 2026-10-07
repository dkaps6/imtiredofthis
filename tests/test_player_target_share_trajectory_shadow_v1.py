import pandas as pd
import scripts.research.lock_player_target_share_trajectory_shadow_v1 as m

def test_trajectory_state():
    ti={"IND":pd.DataFrame([
      {"season":2026,"week":1,"game_id":"g1","team":"IND","team_targets":10},
      {"season":2026,"week":2,"game_id":"g2","team":"IND","team_targets":10},
      {"season":2026,"week":3,"game_id":"g3","team":"IND","team_targets":10},
      {"season":2026,"week":4,"game_id":"g4","team":"IND","team_targets":10},
    ])}
    p={(1,"g1","IND","p"):1,(2,"g2","IND","p"):1,(3,"g3","IND","p"):3,(4,"g4","IND","p"):3}
    z=m.trajectory_state(ti,p,"IND","p")
    assert abs(z["trajectory_delta"]-.20)<1e-12
    assert z["trajectory_feature_max_week"]==4

def test_shadow_preserves_te_pool_and_wr_anchor():
    x=pd.DataFrame([
      {"event_id":"g","team":"IND","position":"TE","entitlement_tgt_share":.10,"trajectory_delta":.10,"wr_r15_anchor":False},
      {"event_id":"g","team":"IND","position":"TE","entitlement_tgt_share":.05,"trajectory_delta":-.10,"wr_r15_anchor":False},
      {"event_id":"g","team":"IND","position":"WR","entitlement_tgt_share":.20,"trajectory_delta":.20,"wr_r15_anchor":True},
      {"event_id":"g","team":"IND","position":"WR","entitlement_tgt_share":.15,"trajectory_delta":.10,"wr_r15_anchor":False},
      {"event_id":"g","team":"IND","position":"WR","entitlement_tgt_share":.10,"trajectory_delta":-.10,"wr_r15_anchor":False},
    ])
    out,aud,gap=m.apply_shadow(x)
    assert gap<1e-12
    assert abs(out.loc[2,"shadow_entitlement_tgt_share"]-.20)<1e-12
    assert abs(out.loc[out.position.eq("TE"),"shadow_entitlement_tgt_share"].sum()-.15)<1e-12
    assert abs(out.loc[[3,4],"shadow_entitlement_tgt_share"].sum()-.25)<1e-12
