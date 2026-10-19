from scripts.research.audit_prism_rb_team_context_carry_count_identity_bridge_v1 import (
    canonical_count_identity,
    count_roster_status_eligible,
)


def test_raw_gsis_maps_to_canonical_count_identity_without_names():
    assert canonical_count_identity("00-0034367") == "gsis:00-0034367"


def test_canonical_count_status_policy_is_exact_act_ina():
    assert count_roster_status_eligible("ACT")
    assert count_roster_status_eligible("ina")
    assert not count_roster_status_eligible("DEV")
    assert not count_roster_status_eligible("CUT")
    assert not count_roster_status_eligible("")
