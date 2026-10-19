import numpy as np
import pandas as pd

from scripts.research.evaluate_prism_rb_team_context_carry_identity_v1 import (
    old_team_anchor,
    score,
)


def test_old_team_anchor_uses_latest_up_to_eight_strictly_prior_rows():
    rows=[]
    for week,share in enumerate([.10,.20,.30,.40,.50,.60,.70,.80,.90],start=1):
        rows.append({
            "season":2024,"week":week,"team":"AAA",
            "player_identity_key":"p1","position":"RB",
            "rushes":share*20,"team_rushes":20,
        })
    counts=pd.DataFrame(rows)
    anchor,hist=old_team_anchor(
        counts,pid="p1",prev_team="AAA",target_season=2024,target_week=10,
    )
    # Latest eight strictly-prior weeks are W2-W9.
    assert len(hist)==8
    assert [x["week"] for x in hist]==list(range(2,10))
    assert np.isclose(anchor,np.mean([.20,.30,.40,.50,.60,.70,.80,.90]))


def test_score_preserves_finite_room_mass_and_changes_only_transition_identity_mass():
    transitions=pd.DataFrame([{
        "season":2024,"week":10,"team":"BBB","stable_id":"p1",
        "family":"RB","previous_rostered_team":"AAA",
        "transition_kind":"INSEASON_TEAM_CHANGE","prior_team_support_ge4":True,
    }])
    authority=pd.DataFrame([
        {"season":2024,"week":10,"team":"BBB","family":"RB","player_id":"p1",
         "cell":"RB_RUSHING","authority_q":0.20,"authority_covered":True},
        {"season":2024,"week":10,"team":"BBB","family":"RB","player_id":"p2",
         "cell":"RB_RUSHING","authority_q":0.30,"authority_covered":True},
    ])
    prior=[]
    for week,rushes in [(4,4),(5,6),(6,8),(7,10)]:
        prior.append({
            "season":2024,"week":week,"team":"AAA","player_identity_key":"p1",
            "position":"RB","rushes":rushes,"team_rushes":20,
        })
    target=[
        {"season":2024,"week":10,"team":"BBB","player_identity_key":"p1",
         "position":"RB","rushes":12,"team_rushes":30},
        {"season":2024,"week":10,"team":"BBB","player_identity_key":"p2",
         "position":"RB","rushes":8,"team_rushes":30},
    ]
    counts=pd.DataFrame(prior+target)
    scored,rooms,extra=score(transitions,authority,counts)
    assert len(scored)==1
    assert len(rooms)==1
    r=rooms.iloc[0]
    assert np.isclose(r["baseline_mass"],1.0)
    assert np.isclose(r["candidate_mass"],1.0)
    s=scored.iloc[0]
    # Old-team anchor = mean(.2,.3,.4,.5)=.35.
    assert np.isclose(s["old_team_identity_q"],.35)
    assert np.isclose(s["candidate_q"],.35)
    assert extra["audit"]["stable_link_rate"]==1.0
