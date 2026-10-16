from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.score_joint_opportunity_allocation_dirichlet_v1 import (
    _cap_with_residual,
    canonical_player_probabilities,
    log_dirichlet_multinomial,
    log_multinomial,
)


def test_cap_with_residual_matches_canonical_095_semantics():
    p,resid=_cap_with_residual(np.array([0.7,0.5]))
    assert abs(float(p.sum())-0.95)<1e-12
    assert abs(resid-0.05)<1e-12
    assert abs(float(p.sum())+resid-1.0)<1e-12


def test_multinomial_and_dirichlet_log_scores_finite():
    y=np.array([3,2,5])
    p=np.array([0.3,0.2,0.5])
    alpha=np.array([3.0,2.0,5.0])
    assert np.isfinite(log_multinomial(y,p))
    assert np.isfinite(log_dirichlet_multinomial(y,alpha))


def test_dirichlet_multinomial_approaches_multinomial_at_high_concentration():
    y=np.array([4,1,5])
    p=np.array([0.4,0.1,0.5])
    base=log_multinomial(y,p)
    cand=log_dirichlet_multinomial(y,p*1e8)
    assert abs(cand-base)<1e-5


def test_canonical_rb_probability_uses_top5_and_residual():
    rows=[]
    for i,share in enumerate([0.30,0.25,0.20,0.15,0.10,0.08],start=1):
        rows.append({
            "team":"IND","player":f"Back {i}","player_clean_key":f"back{i}",
            "position":"RB","market":"rush_yards","rules_rush_share":share,
            "rules_tgt_share":0.0,
        })
    out=canonical_player_probabilities(pd.DataFrame(rows),"RB_CARRY_SHARE")
    players=out.loc[~out["player_clean_key"].eq("__RESIDUAL__")]
    # Sixth back is removed by canonical top-5 rule.
    assert players.loc[players["player_clean_key"].eq("back6"),"q"].iloc[0]==0.0
    assert abs(float(out["q"].sum())-1.0)<1e-12


def test_canonical_wr_probability_preserves_probability_mass():
    rows=[
        {"team":"IND","player":"WR A","player_clean_key":"wra","position":"WR","market":"rec_yards","rules_tgt_share":0.30,"rules_rush_share":0.0},
        {"team":"IND","player":"WR B","player_clean_key":"wrb","position":"WR","market":"rec_yards","rules_tgt_share":0.20,"rules_rush_share":0.0},
        {"team":"IND","player":"TE A","player_clean_key":"tea","position":"TE","market":"rec_yards","rules_tgt_share":0.15,"rules_rush_share":0.0},
        {"team":"IND","player":"RB A","player_clean_key":"rba","position":"RB","market":"rec_yards","rules_tgt_share":0.10,"rules_rush_share":0.2},
    ]
    out=canonical_player_probabilities(pd.DataFrame(rows),"WR_TARGET_SHARE")
    assert abs(float(out["q"].sum())-1.0)<1e-12
    # WR sharpening preserves total WR mass before any team cap.
    wr=out.loc[out["position"].eq("WR"),"q"].sum()
    assert abs(float(wr)-0.50)<1e-12
