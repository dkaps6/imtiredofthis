import numpy as np
import pandas as pd

from scripts.research.audit_qb_c2_rng_isolation_extension_v1 import apply_c2_isolated
from scripts.simulation_c2_qb_candidate import StateSimulationResult


def _fixture():
    n = 500
    metrics = pd.DataFrame(
        [
            {
                "event_id": "g",
                "team": "T",
                "player": "Quarter Back",
                "player_clean_key": "qb",
                "position": "QB",
                "qb_projection_eligible": 1,
                "qb_role_score": 0.0,
                "entitlement_tgt_share": 0.0,
            },
            {
                "event_id": "g",
                "team": "T",
                "player": "Wide One",
                "player_clean_key": "wr1",
                "position": "WR",
                "qb_projection_eligible": 0,
                "qb_role_score": -999.0,
                "entitlement_tgt_share": 0.20,
                "rules_catch_rate": 0.65,
                "rules_ypt": 8.0,
                "rules_volatility_mult": 1.0,
            },
            {
                "event_id": "g",
                "team": "T",
                "player": "Tight A",
                "player_clean_key": "tea",
                "position": "TE",
                "qb_projection_eligible": 0,
                "qb_role_score": -999.0,
                "entitlement_tgt_share": 0.12,
                "rules_catch_rate": 0.70,
                "rules_ypt": 7.0,
                "rules_volatility_mult": 1.0,
            },
            {
                "event_id": "g",
                "team": "T",
                "player": "Tight B",
                "player_clean_key": "teb",
                "position": "TE",
                "qb_projection_eligible": 0,
                "qb_role_score": -999.0,
                "entitlement_tgt_share": 0.08,
                "rules_catch_rate": 0.60,
                "rules_ypt": 6.5,
                "rules_volatility_mult": 1.0,
            },
        ]
    )
    pass_att = np.full(n, 30, dtype=int)
    pass_eff = np.ones(n, dtype=float)
    base = StateSimulationResult(
        values={
            ("g", "qb", "pass_yards"): np.full(n, 210.0),
            ("g", "wr1", "rec_yards"): np.arange(n, dtype=float),
            ("g", "tea", "rec_yards"): np.arange(n, dtype=float) + 1,
            ("g", "teb", "rec_yards"): np.arange(n, dtype=float) + 2,
        },
        iterations=n,
        team_states={
            ("g", "T", "pass_att"): pass_att,
            ("g", "T", "pass_eff_shock"): pass_eff,
        },
    )
    plan = {
        "te_changed": {("g", "T", "tea"), ("g", "T", "teb")},
        "wr_changed": set(),
        "authority": {
            ("g", "T", "qb"): 0.0,
            ("g", "T", "wr1"): 0.20,
            ("g", "T", "tea"): 0.12,
            ("g", "T", "teb"): 0.08,
        },
    }
    return metrics, base, plan


def test_isolated_c2_is_repeatable_and_mean_neutral():
    metrics, base, plan = _fixture()
    a, da, pa = apply_c2_isolated(
        base, metrics, plan=plan, anchor_map={("g", "T"): 210.0}, seed=5601
    )
    b, db, pb = apply_c2_isolated(
        base, metrics, plan=plan, anchor_map={("g", "T"): 210.0}, seed=5601
    )
    assert np.array_equal(a.values[("g", "qb", "pass_yards")], b.values[("g", "qb", "pass_yards")])
    assert abs(a.values[("g", "qb", "pass_yards")].mean() - 210.0) <= 1e-10
    assert pa["primary_qb_rows"] == 1
    assert pb["primary_qb_rows"] == 1
    assert float(da["raw_mean_gap"].abs().max()) <= 1e-10
    assert float(db["raw_mean_gap"].abs().max()) <= 1e-10


def test_isolated_c2_only_replaces_primary_qb_pass_yards():
    metrics, base, plan = _fixture()
    out, _, payload = apply_c2_isolated(
        base, metrics, plan=plan, anchor_map={("g", "T"): 210.0}, seed=5601
    )
    assert payload["all_changed_keys_are_primary_qb_pass_yards"] is True
    assert payload["changed_simulation_keys"] == 1
    for key in [("g", "wr1", "rec_yards"), ("g", "tea", "rec_yards"), ("g", "teb", "rec_yards")]:
        assert np.array_equal(base.values[key], out.values[key])
