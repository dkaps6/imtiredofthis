from scripts.research.grade_week5_player_failure_casebook_v1 import _component_label


def test_component_label_opportunity_dominant():
    assert _component_label(30.0,10.0)=="OPPORTUNITY_DOMINANT"


def test_component_label_efficiency_dominant():
    assert _component_label(10.0,30.0)=="EFFICIENCY_DOMINANT"


def test_component_label_mixed():
    assert _component_label(10.0,8.0)=="MIXED_OPPORTUNITY_EFFICIENCY"
