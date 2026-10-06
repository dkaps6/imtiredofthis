import pandas as pd
import scripts.research.audit_public_position_ypt_parity_v1 as m


def test_builder_strict_prior_and_latest_eight():
    rows=[]
    for w in range(1,11):
        rows.append({"season":2026,"week":w,"position":"TE","targets":10,"rec_yards":10*w,"opponent":"PIT"})
    x=pd.DataFrame(rows)
    out=m.build_position_ypt(x,[11])
    r=out.iloc[0]
    # target W11 uses W3-W10 only => mean weekly yards 65 / 10 targets = 6.5 YPT.
    assert r["source_max_week"]==10
    assert abs(r["te_ypt_allowed_public"]-6.5)<1e-12
    assert r["te_targets_faced_public"]==80


def test_position_groups_are_separate():
    x=pd.DataFrame([
        {"season":2026,"week":1,"position":"TE","targets":2,"rec_yards":20,"opponent":"PIT"},
        {"season":2026,"week":1,"position":"WR","targets":4,"rec_yards":20,"opponent":"PIT"},
        {"season":2026,"week":1,"position":"RB","targets":5,"rec_yards":10,"opponent":"PIT"},
    ])
    r=m.build_position_ypt(x,[2]).iloc[0]
    assert r["te_ypt_allowed_public"]==10
    assert r["wr_ypt_allowed_public"]==5
    assert r["rb_ypt_allowed_public"]==2
