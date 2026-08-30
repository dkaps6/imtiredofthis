import pandas as pd

from scripts.modeling.qb_pass_synthesis_v1 import qb_prior


def test_qb_prior_prefers_stable_identity_over_spelling_sensitive_name():
    logs = pd.DataFrame([
        {
            "_identity_key": "gsis:00-0012345",
            "_pkey": "examplequarterbackjr",
            "season": 2025,
            "week": 17,
            "pass_att": 30,
            "pass_yards": 210,
            "ypa_game": 7.0,
        },
        {
            "_identity_key": "gsis:00-0012345",
            "_pkey": "examplequarterbackjr",
            "season": 2025,
            "week": 18,
            "pass_att": 40,
            "pass_yards": 320,
            "ypa_game": 8.0,
        },
    ])

    # Current provider spelling has lost the suffix. Stable identity should still
    # recover the same historical QB prior.
    att, ypa, n = qb_prior(
        logs,
        "examplequarterback",
        2026,
        1,
        player_identity_key="gsis:00-0012345",
    )
    assert n == 2
    assert att == 35.0
    assert ypa == 7.5


def test_qb_prior_keeps_backwards_compatible_name_fallback():
    logs = pd.DataFrame([
        {
            "_identity_key": "",
            "_pkey": "legacyqb",
            "season": 2025,
            "week": 18,
            "pass_att": 32,
            "pass_yards": 224,
            "ypa_game": 7.0,
        }
    ])
    att, ypa, n = qb_prior(logs, "Legacy QB", 2026, 1)
    assert (att, ypa, n) == (32.0, 7.0, 1)
