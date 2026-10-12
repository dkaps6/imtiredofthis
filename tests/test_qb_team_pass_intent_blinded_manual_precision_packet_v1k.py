from scripts.research.build_qb_team_pass_intent_blinded_manual_precision_packet_v1k import (
    _row_hash,
    _review_excerpts,
)


def test_row_hash_is_deterministic():
    row = {
        "season": 2024,
        "week": 5,
        "team": "SEA",
        "locator": "https://example.test/page",
    }
    assert _row_hash(row) == _row_hash(dict(row))


def test_review_excerpts_prioritizes_action_language_and_excludes_postgame():
    text = (
        "The quarterback completed several passes last week. "
        "We want to establish the run against the Giants and create more opportunities for our backs. "
        "After the game the offense explained the final score and passing totals."
    )
    excerpts = _review_excerpts(text, "NYG")
    assert excerpts
    assert "establish the run" in excerpts[0].lower()
    assert all("final score" not in x.lower() for x in excerpts)
