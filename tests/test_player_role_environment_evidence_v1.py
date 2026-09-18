from scripts.research.validate_player_role_environment_evidence_v1 import load_contract, validate_contract


def test_player_role_environment_evidence_v1_passes():
    result=validate_contract(load_contract())
    assert result["disposition"]=="PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1_VALIDATED"
    assert result["field_count"] >= 40
    assert result["sources"] >= 8
    assert result["production_integration_authorized"] is False
    assert result["predictive_coefficients_authorized"] is False


def test_role_schema_covers_all_positions_and_key_families():
    d=load_contract()
    assert set(d["target_positions"])=={"QB","RB","WR","TE"}
    families={x["family"] for x in d["evidence_fields"]}
    for family in {"availability","hierarchy","team_continuity","room_continuity","vacancy","current_season_role","qb_environment","coaching","trench_context","qualitative_role","meta"}:
        assert family in families


def test_qualitative_evidence_requires_pre_kickoff_provenance():
    d=load_contract()
    req=set(d["qualitative_statement_grain"]["required_fields"])
    assert {"source_reference","published_at_utc","available_before_kickoff","authority_tier","direct_vs_interpreted","confidence"}.issubset(req)
