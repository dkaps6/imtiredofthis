import pandas as pd

from scripts.research.audit_prism_te_receiving_team_context_stage1_participation_authority_reset_v1_source import (
    attach_reset,
    select_rows,
)


def test_te_first_market_row_respects_next_transition():
    full=pd.DataFrame([
        {"season":2024,"week":1,"team":"AAA","stable_id":"00-1","transition_kind":"OFFSEASON_TEAM_CHANGE"},
        {"season":2024,"week":8,"team":"BBB","stable_id":"00-1","transition_kind":"INSEASON_TEAM_CHANGE"},
    ])
    eligible=full.iloc[[0]].copy()
    eligible["transition_id"]="t1"
    parent=pd.DataFrame([
        {"season":2024,"week":5,"team":"AAA","player_id":"00-1","game_key":"g5",
         "stage1_p_active":.8,"stage1_named_player_supported":True,
         "authority_q":.2,"canonical_zero_share":False},
        {"season":2024,"week":9,"team":"BBB","player_id":"00-1","game_key":"g9",
         "stage1_p_active":.7,"stage1_named_player_supported":True,
         "authority_q":.2,"canonical_zero_share":False},
    ])
    out,ex=select_rows(full,eligible,parent)
    assert ex["NO_CANONICAL_TE_RECEIVING_BASELINE"]==0
    assert len(out)==1
    assert int(out.iloc[0]["target_week"])==5
    assert int(out.iloc[0]["next_transition_week"])==8


def test_te_reset_uses_stage1_role_prior_only_when_named_history_exists():
    selected=pd.DataFrame([{
        "transition_id":"t1","season":2024,"transition_week":1,"team":"AAA","stable_id":"00-1",
        "transition_kind":"OFFSEASON_TEAM_CHANGE","next_transition_week":None,"target_week":5,
        "game_key":"g5","stage1_p_active":.8,"stage1_named_player_supported":True,
        "authority_q":.2,"canonical_zero_share":False,
    }])
    stage1=pd.DataFrame([{
        "season":2024,"week":5,"team":"AAA","player_id":"00-1",
        "p_baseline":.6,"p_candidate":.8,"n_player_prior":4,
    }])
    out=attach_reset(selected,stage1)
    assert bool(out.iloc[0]["stage1_exact_mechanism_row"])
    assert bool(out.iloc[0]["stage1_support_flag_matches_exact_row"])
    assert float(out.iloc[0]["reset_p_active"])==.6
    assert bool(out.iloc[0]["actionable_stage1_reset"])
