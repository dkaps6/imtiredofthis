"""Candidate: movers should lean on this season's usage more than stayers.

Default behaviour (flag unset) must be byte-identical to the fixed 4-game prior.
"""
import numpy as np, pandas as pd, pytest
from scripts import player_form_v2 as pf

K = "player_identity_key"
M = ["tgt_share", "rush_share", "route_rate", "yprr", "ypt", "ypc", "ypa", "receptions_per_target"]

def _frames(prior_team):
    def row(share, games, **x): return {K: "gsis:1", "games": games, **{m: np.nan for m in M}, "tgt_share": share, **x}
    prior = pd.DataFrame([row(0.133, 12, historical_team=prior_team)])
    cur = pd.DataFrame([row(0.265, 8)])
    uni = pd.DataFrame([{K: "gsis:1", "player": "Isaiah Likely", "team": "NYG"}])
    return prior, cur, uni

def test_default_is_unchanged(monkeypatch):
    monkeypatch.delenv("PLAYERFORM_TEAM_CHANGE_PRIOR_K", raising=False)
    out = pf._blend(*_frames("BAL"))
    assert out.tgt_share.iloc[0] == pytest.approx((4/12)*0.133 + (8/12)*0.265)
    assert out.prior_k_applied.iloc[0] == 4.0 and out.prior_team_changed.iloc[0] == 1

def test_flag_changes_only_movers(monkeypatch):
    monkeypatch.setenv("PLAYERFORM_TEAM_CHANGE_PRIOR_K", "1")
    moved = pf._blend(*_frames("BAL")).tgt_share.iloc[0]
    stayed = pf._blend(*_frames("NYG")).tgt_share.iloc[0]
    assert moved == pytest.approx((1/9)*0.133 + (8/9)*0.265)
    assert stayed == pytest.approx((4/12)*0.133 + (8/12)*0.265)

def test_missing_prior_team_is_treated_as_stayer(monkeypatch):
    monkeypatch.setenv("PLAYERFORM_TEAM_CHANGE_PRIOR_K", "1")
    prior, cur, uni = _frames("BAL"); prior = prior.drop(columns="historical_team")
    out = pf._blend(prior, cur, uni)
    assert out.prior_team_changed.iloc[0] == 0 and out.prior_k_applied.iloc[0] == 4.0
