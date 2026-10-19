import pandas as pd

from scripts.research.audit_prism_rb_rushing_team_context_stage1_participation_authority_reset_v1_source import select


def test_rb_rushing_source_skips_zero_entitlement_and_selects_first_positive_row():
    full=pd.DataFrame([
        {"season":2024,"week":1,"team":"AAA","stable_id":"00-1","transition_kind":"OFFSEASON_TEAM_CHANGE"},
    ])
    eligible=full.copy()
    eligible["transition_id"]="t1"
    parent=pd.DataFrame([
        {"season":2024,"week":5,"team":"AAA","player_id":"00-1","game_key":"g5",
         "authority_q":0.0,"canonical_zero_share":True,"stage1_p_active":0.8,
         "stage1_named_player_supported":True},
        {"season":2024,"week":6,"team":"AAA","player_id":"00-1","game_key":"g6",
         "authority_q":0.2,"canonical_zero_share":False,"stage1_p_active":0.8,
         "stage1_named_player_supported":True},
    ])
    out,ex=select(full,eligible,parent)
    assert ex["NO_POSITIVE_CANONICAL_RB_RUSHING_BASELINE"]==0
    assert len(out)==1
    assert int(out.iloc[0]["target_week"])==6
    assert float(out.iloc[0]["authority_q"])>0
