from scripts.research.audit_qb_team_pass_intent_pretrained_semantic_router_v1f import (
    MODEL_ID,
    MODEL_REVISION,
    SemanticResult,
    evaluate,
)


def _r(team, positive=True, auto=True, relevant=True, semantic=True, tag="PASS_EMPHASIS"):
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
        semantic_candidate_found=semantic,
        candidate_tag=tag if semantic else "",
        positive_score=0.55 if semantic else 0.0,
        negative_score=0.25 if semantic else 0.0,
        semantic_margin=0.30 if semantic else 0.0,
        evidence="We intend to create more opportunities in the passing game this week." if semantic else "",
        auto_review_ready=auto,
        errors=[],
    )


def test_model_identity_is_pinned():
    assert MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"
    assert MODEL_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_gold_gate_passes_at_11_of_12_with_clean_atl_negative():
    teams=["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows=[_r(t,auto=(i != 0)) for i,t in enumerate(teams)]
    rows.append(_r("ATL",positive=False,auto=False,relevant=True,semantic=False,tag=""))
    out=evaluate(rows)
    assert out["positive_auto_review_ready"] == 11
    assert out["negative_auto_review_ready"] == 0
    assert out["auto_review_ready_precision"] == 1.0
    assert out["disposition"] == "PRETRAINED_SEMANTIC_ROUTER_QUALIFIED_FOR_FULL_SOURCE_SCREEN"
    assert out["full_source_screen_authorized"] is True
    assert out["predictive_candidate_authorized"] is False


def test_ten_of_twelve_fails():
    teams=["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows=[_r(t,auto=(i >= 2)) for i,t in enumerate(teams)]
    rows.append(_r("ATL",positive=False,auto=False,relevant=True,semantic=False,tag=""))
    out=evaluate(rows)
    assert out["positive_auto_review_ready"] == 10
    assert out["disposition"] == "PRETRAINED_SEMANTIC_ROUTER_NOT_QUALIFIED"


def test_atl_auto_accept_fails_even_if_precision_still_high():
    teams=["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows=[_r(t) for t in teams]
    rows.append(_r("ATL",positive=False,auto=True,relevant=True,semantic=True))
    out=evaluate(rows)
    assert out["auto_review_ready_precision"] >= 0.85
    assert out["gates"]["atl_official_negative_not_auto_ready"] is False
    assert out["disposition"] == "PRETRAINED_SEMANTIC_ROUTER_NOT_QUALIFIED"


def test_governance_is_source_only():
    teams=["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows=[_r(t) for t in teams]
    rows.append(_r("ATL",positive=False,auto=False,relevant=True,semantic=False,tag=""))
    out=evaluate(rows)
    assert out["external_search_engine_html_contacted"] is False
    assert out["football_outcomes_read"] == 0
    assert out["model_residuals_read"] == 0
    assert out["sportsbook_inputs_used"] is False
    assert out["paid_oddsapi_calls"] == 0
    assert out["week5_2026_outcomes_read"] == 0
    assert out["supervised_fit_or_finetune_used"] is False
    assert out["football_predictive_models_fit"] == 0
    assert out["production_changed"] is False
