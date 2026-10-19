import pandas as pd

from scripts.research.audit_prism_return_state_receiving_opportunity_v1 import (
    CLEAN_STATE,
    RETURN_RESTORED,
    build_membership,
    finite_opportunity_audit,
)


def _rec_row(week, *, state=CLEAN_STATE, targets=4.0, team_targets=30.0,
             positive=True, pf=True, family="WR", pid="p1", team="AAA",
             season=2024):
    return {
        "season":season,
        "week":week,
        "team":team,
        "stable_id":pid,
        "family":family,
        "positive_participation":positive,
        "offense_snaps":40.0 if positive else 0.0,
        "offense_pct":0.70 if positive else 0.0,
        "observation_state":state,
        "playerform_weekly_row_present":pf,
        "targets":targets,
        "team_targets":team_targets,
        "target_share_game":targets/team_targets if team_targets else float("nan"),
    }


def _episode():
    return {
        "season":2024,
        "team":"AAA",
        "stable_id":"p1",
        "family":"WR",
        "episode_number":1,
        "episode_start_week":5,
        "pre_episode_snap_pct":0.72,
        "return_week":7,
        "return_state":RETURN_RESTORED,
    }


def test_exact_prior_return_and_immediate_next_membership():
    episodes=pd.DataFrame([_episode()])
    receiving=pd.DataFrame([
        _rec_row(1,targets=3),
        _rec_row(2,targets=4),
        _rec_row(3,targets=5),
        _rec_row(4,targets=6),
        _rec_row(5,state="REPORTED_INJURY_PARTICIPATED",targets=1),
        _rec_row(6,state="REPORTED_INJURY_NO_OFFENSIVE_SNAPS",targets=0,positive=False,pf=False),
        _rec_row(7,state="REPORTED_INJURY_PARTICIPATED",targets=7),
        _rec_row(8,state=CLEAN_STATE,targets=8),
    ])
    mem,pri,team_weeks,audit=build_membership(episodes,receiving)
    assert audit["chronology_violations"] == 0
    assert audit["skipped_immediate_next_violations"] == 0
    assert len(pri) == 4
    r=mem.iloc[0]
    assert bool(r["prior_support_ok"])
    assert bool(r["return_row_unique"])
    assert bool(r["return_opportunity_valid"])
    assert int(r["immediate_next_week"]) == 8
    assert r["immediate_next_state"] == CLEAN_STATE
    assert bool(r["immediate_next_clean"])
    assert (2024,8,"AAA") in team_weeks


def test_missing_playerform_row_never_becomes_valid_zero_target_prior():
    episodes=pd.DataFrame([_episode()])
    receiving=pd.DataFrame([
        _rec_row(1,targets=3),
        _rec_row(2,targets=4),
        _rec_row(3,targets=0,pf=False),
        _rec_row(4,targets=6),
        _rec_row(7,state="REPORTED_INJURY_PARTICIPATED",targets=7),
        _rec_row(8,state=CLEAN_STATE,targets=8),
    ])
    mem,pri,_,_=build_membership(episodes,receiving)
    assert len(pri) == 4
    assert not bool(mem.iloc[0]["prior_support_ok"])
    bad=pri.loc[pri["prior_week"].eq(3)].iloc[0]
    assert bad["targets"] == 0
    assert not bool(bad["playerform_weekly_row_present"])
    assert not bool(bad["opportunity_row_valid"])


def test_immediate_next_is_not_skipped_when_reported():
    episodes=pd.DataFrame([_episode()])
    receiving=pd.DataFrame([
        _rec_row(1),_rec_row(2),_rec_row(3),_rec_row(4),
        _rec_row(7,state="REPORTED_INJURY_PARTICIPATED"),
        _rec_row(8,state="REPORTED_INJURY_PARTICIPATED"),
        _rec_row(9,state=CLEAN_STATE),
    ])
    mem,_,_,audit=build_membership(episodes,receiving)
    r=mem.iloc[0]
    assert int(r["immediate_next_week"]) == 8
    assert r["immediate_next_state"] == "REPORTED_INJURY_PARTICIPATED"
    assert not bool(r["immediate_next_clean"])
    assert audit["skipped_immediate_next_violations"] == 0


def test_finite_team_opportunity_preserves_residual():
    receiving=pd.DataFrame([
        _rec_row(1,targets=10,family="WR",pid="p1"),
        _rec_row(1,targets=8,family="WR",pid="p2"),
        _rec_row(1,targets=5,family="TE",pid="p3"),
        _rec_row(1,targets=4,family="RB",pid="p4"),
    ])
    out,audit=finite_opportunity_audit(receiving,{(2024,1,"AAA")})
    r=out.iloc[0]
    assert r["team_targets"] == 30
    assert r["rb_wr_te_targets"] == 27
    assert r["residual_team_targets"] == 3
    assert bool(r["target_sum_within_team_total"])
    assert bool(r["residual_nonnegative"])
    assert audit["rb_wr_te_target_sum_exceeds_team_total_weeks"] == 0
