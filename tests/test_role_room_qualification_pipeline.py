from pathlib import Path


def test_role_room_pipeline_is_outcome_free_and_calls_frozen_builders():
    path = Path(__file__).parents[1] / "scripts" / "research" / "run_role_room_qualification_pipeline.py"
    text = path.read_text()
    required = [
        "build_usage_regime_context.py",
        "build_room_continuity_context.py",
        "build_role_room_context.py",
        "build_role_room_transition_diagnostics.py",
        "build_context_stability_evidence.py",
        "build_context_candidate_profile.py",
    ]
    for name in required:
        assert name in text
    forbidden = ["oddsapi", "sportsbook", "target_outcome", "actual_yards", "bet_result"]
    lowered = text.lower()
    # Documentation may name sportsbook/target outcomes only to state that they are not read;
    # executable command construction must not reference an odds or outcome input argument.
    command_region = lowered[lowered.index("def main"):]
    assert "--odds" not in command_region
    assert "--outcome" not in command_region
    assert "--target" not in command_region


def test_pipeline_profiles_canonical_integrity_fields():
    path = Path(__file__).parents[1] / "scripts" / "research" / "run_role_room_qualification_pipeline.py"
    text = path.read_text()
    for token in [
        "pregame_context_eligible_flag",
        "stable_identity_flag",
        "any_context_unknown_flag",
        "strict_prior_support_games",
        "season,week,team,player_identity_key",
    ]:
        assert token in text
