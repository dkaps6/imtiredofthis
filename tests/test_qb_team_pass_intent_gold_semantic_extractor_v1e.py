from scripts.research.audit_qb_team_pass_intent_gold_semantic_extractor_v1e import (
    FROZEN_TAGS,
    SemanticResult,
    _best_evidence,
    evaluate,
)


def _r(team, positive=True, auto=True, relevant=True, evidence=True, tags=None):
    if tags is None:
        tags=["PASS_EMPHASIS"] if evidence else []
    return SemanticResult(
        team=team,
        opponent="PIT",
        expected_positive=positive,
        url="https://www.example.com/news/x",
        fetch_success=True,
        first_party=True,
        timestamp_safe=True,
        timestamp_method="JSON_LD_datePublished",
        target_relevant=relevant,
        evidence_found=evidence,
        candidate_tags=tags,
        evidence="We want to get the passing game going with more downfield opportunities." if evidence else "",
        auto_review_ready=auto,
        errors=[],
    )


def test_generic_pass_candidate_extracts_without_team_specific_terms():
    text=(
        "The coordinator met with reporters Wednesday. "
        "We want to get the passing game going and take more shots downfield this week. "
        "The defense presents a difficult challenge."
    )
    evidence,tags=_best_evidence(text)
    assert evidence
    assert "PASS_EMPHASIS" in tags
    assert len(evidence.split()) <= 25


def test_generic_run_candidate_extracts():
    text="Our goal is to establish the run and stick with the run game longer this week."
    evidence,tags=_best_evidence(text)
    assert evidence
    assert "RUN_EMPHASIS" in tags


def test_postgame_result_phrase_is_not_candidate():
    text="After the game, the coach said the goal was to establish the run and attack downfield."
    evidence,tags=_best_evidence(text)
    assert evidence == ""
    assert tags == []


def test_frozen_gold_threshold_passes_at_11_of_12_with_clean_atl_negative():
    teams=["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows=[]
    for i,t in enumerate(teams):
        rows.append(_r(t,auto=(i != 0),relevant=True,evidence=True))
    rows.append(_r("ATL",positive=False,auto=False,relevant=True,evidence=False,tags=[]))
    out=evaluate(rows)
    assert out["positive_auto_review_ready"]==11
    assert out["negative_auto_review_ready"]==0
    assert out["auto_review_ready_precision"]==1.0
    assert out["disposition"]=="GOLD_SEMANTIC_EXTRACTOR_QUALIFIED_FOR_FULL_SOURCE_SCREEN"
    assert out["full_source_screen_authorized"] is True
    assert out["predictive_candidate_authorized"] is False


def test_ten_of_twelve_positive_recall_fails():
    teams=["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows=[]
    for i,t in enumerate(teams):
        rows.append(_r(t,auto=(i >= 2),relevant=True,evidence=True))
    rows.append(_r("ATL",positive=False,auto=False,relevant=True,evidence=False,tags=[]))
    out=evaluate(rows)
    assert out["positive_auto_review_ready"]==10
    assert out["disposition"]=="GOLD_SEMANTIC_EXTRACTOR_NOT_QUALIFIED"


def test_atl_negative_auto_acceptance_fails_even_if_precision_above_85pct():
    teams=["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows=[_r(t) for t in teams]
    rows.append(_r("ATL",positive=False,auto=True,relevant=True,evidence=True))
    out=evaluate(rows)
    assert out["auto_review_ready_precision"] >= 0.85
    assert out["gates"]["atl_official_negative_not_auto_ready"] is False
    assert out["disposition"]=="GOLD_SEMANTIC_EXTRACTOR_NOT_QUALIFIED"


def test_only_frozen_tags_are_emitted_by_mechanics():
    text=(
        "We are going to protect the quarterback, attack their coverage, and use more two-back packages."
    )
    evidence,tags=_best_evidence(text)
    assert evidence
    assert tags
    assert set(tags) <= FROZEN_TAGS
