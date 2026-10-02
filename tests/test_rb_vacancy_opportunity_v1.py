import pandas as pd
import pytest
from scripts.research.rb_vacancy_opportunity_v1 import build_state


def fixtures():
    av=pd.DataFrame([
        {"team":"BUF","player_clean_key":"out","position_group":"RB","definitive_unavailable":1,"final_availability_state":"UNAVAILABLE_OUT"},
        {"team":"BUF","player_clean_key":"a","position_group":"RB","definitive_unavailable":0,"final_availability_state":"AVAILABLE"},
        {"team":"BUF","player_clean_key":"b","position_group":"FB","definitive_unavailable":0,"final_availability_state":"AVAILABLE"},
        {"team":"BUF","player_clean_key":"doubt","position_group":"RB","definitive_unavailable":0,"final_availability_state":"UNCERTAIN_DOUBTFUL"},
    ])
    roles=av.loc[av.definitive_unavailable.eq(0),["team","player_clean_key","position_group"]].copy()
    logs=pd.DataFrame([
        {"season":2026,"week":1,"team":"BUF","player_clean_key":"out","rush_share_game":.40},
        {"season":2026,"week":3,"team":"BUF","player_clean_key":"out","rush_share_game":.99},
    ])
    snaps=pd.DataFrame([
        {"season":2026,"week":1,"team":"BUF","player_clean_key":"a","offense_pct":.60},
        {"season":2026,"week":1,"team":"BUF","player_clean_key":"b","offense_pct":.30},
        {"season":2026,"week":1,"team":"BUF","player_clean_key":"doubt","offense_pct":.10},
        {"season":2026,"week":3,"team":"BUF","player_clean_key":"a","offense_pct":.99},
    ])
    return av,roles,logs,snaps


def test_strict_prior_conservation_and_doubtful_not_vacancy():
    out,exc=build_state(*fixtures(),2026,2)
    assert exc.empty
    assert set(out.successor_player_clean_key)=={"a","b","doubt"}
    assert out.vacated_rush_share.unique().tolist()==[.4]
    assert abs(out.successor_weight.sum()-1)<1e-10
    assert abs(out.transfer_rush_share.sum()-.4)<1e-10
    assert out.snap_source_week.max()==1
    assert "doubt" not in out.unavailable_players.iloc[0]


def test_missing_prior_rush_share_fails_closed():
    av,roles,logs,snaps=fixtures(); logs=logs.loc[logs.player_clean_key.ne("out")]
    out,exc=build_state(av,roles,logs,snaps,2026,2)
    assert out.empty
    assert "NO_PRIOR_RUSH_SHARE" in set(exc.reason)


def test_forbidden_target_or_market_fields_rejected():
    av,roles,logs,snaps=fixtures(); snaps["line"]=42.5
    with pytest.raises(RuntimeError,match="forbidden"):
        build_state(av,roles,logs,snaps,2026,2)
