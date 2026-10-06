import numpy as np
import pandas as pd
import scripts.research.score_public_te_ypt_integration_candidate_v1 as m


def test_public_te_ypt_uses_latest_eight_strict_prior():
    logs=[]
    for w in range(1,11):
        logs.append({"season":2022,"week":w,"team":"CLE","opponent":"PIT","position":"TE","targets":10,"rec_yards":10*w})
    u=pd.DataFrame([{"season":2022,"week":11,"team":"CLE","opponent":"PIT"}])
    out=m.build_public_te_ypt(pd.DataFrame(logs),u,seasons=(2022,))
    r=out.iloc[0]
    assert r["source_max_week"]==10
    assert r["public_def_te_targets_faced"]==80
    assert abs(r["public_def_te_ypt_allowed"]-6.5)<1e-12


def test_zero_intercept_beta():
    q=pd.DataFrame([
        {"season":2022,"candidate_scoreable":True,"weakness_z":-1.0,"actual":8.0,"baseline_projection":10.0,"game_id":"g1"},
        {"season":2022,"candidate_scoreable":True,"weakness_z":1.0,"actual":12.0,"baseline_projection":10.0,"game_id":"g2"},
    ])
    f=m.fit_beta(q)
    assert abs(f["beta_train"]-2.0)<1e-12
    assert f["positive_beta"]


def test_position_group():
    assert m._position_group("TE")=="TE"
    assert m._position_group("TE1")=="TE"
    assert m._position_group("WR")=="WR"
    assert m._position_group("HB")=="RB"
