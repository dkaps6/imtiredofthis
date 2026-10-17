import numpy as np
import pandas as pd

from scripts.research.lock_prism_week6_sunmon_stage2_shadow_v1 import (
    ORIGIN_ARTIFACT,
    ORIGIN_RUN,
    PRIOR_WEEKS,
    TARGET_SEASON,
    TARGET_WEEK,
    _positive_skill_snap_bridge,
    rush_entitlement,
)


def test_frozen_week6_contract_constants():
    assert TARGET_SEASON == 2026
    assert TARGET_WEEK == 6
    assert PRIOR_WEEKS == (1, 2, 3, 4, 5)
    assert ORIGIN_RUN == 38791377571
    assert ORIGIN_ARTIFACT == 11999923880


def test_rush_entitlement_preserves_mass_with_residual():
    x = pd.DataFrame({
        "event_id": ["g"] * 6,
        "team": ["AAA"] * 6,
        "player_clean_key": list("abcdef"),
        "rules_rush_share": [0.50, 0.30, 0.20, 0.10, 0.05, 0.01],
    })
    out = rush_entitlement(x)
    assert len(out) == 6
    q = out["rush_q"].to_numpy(float)
    assert np.isfinite(q).all()
    assert (q >= 0).all()
    assert q.sum() <= 0.95 + 1e-12
    assert np.isclose(q.sum(), 0.95, atol=1e-12)
    # Literal top-five support: the sixth-ranked player stays at zero.
    assert np.isclose(float(out.loc[out.player_clean_key.eq("f"), "rush_q"].iloc[0]), 0.0)


def test_positive_skill_snap_bridge_keeps_unmatched_skill_rows_in_denominator():
    raw = pd.DataFrame({
        "season": [2026, 2026, 2026],
        "week": [1, 1, 1],
        "gsis_id": ["00-0000001", "00-0000002", "00-0000003"],
        "position": ["RB", "TE", "OL"],
        "team": ["CLE", "CLE", "CLE"],
        "offense_pct": [0.50, 0.40, 0.90],
    })
    roster_all = pd.DataFrame({
        "season": [2026],
        "week": [1],
        "team": ["CLE"],
        "player_id": ["00-0000001"],
        "family": ["RB"],
    })

    mapped, rate = _positive_skill_snap_bridge(raw, roster_all)

    # RB + TE are the frozen Stage-1 skill-family denominator. The OL is not.
    assert len(mapped) == 2
    assert set(mapped["family_snap"]) == {"RB", "TE"}
    # The unmatched TE must remain in the denominator instead of disappearing.
    assert mapped["bridge_match"].tolist().count(True) == 1
    assert mapped["bridge_match"].tolist().count(False) == 1
    assert np.isclose(rate, 0.5)
