import pytest

from scripts.data_frontier import bdb_2024_contact_geometry as g


def test_closing_speed_positive_when_defender_moves_toward_stationary_carrier():
    carrier = {"x": 10.0, "y": 20.0, "s": 0.0, "dir": 0.0}
    # Defender is east of carrier and moves west. NGS dir=270 -> negative x.
    defender = {"x": 15.0, "y": 20.0, "s": 4.0, "dir": 270.0}
    assert g.closing_speed_proxy(carrier, defender) == pytest.approx(4.0)


def test_closing_speed_negative_when_defender_moves_away():
    carrier = {"x": 10.0, "y": 20.0, "s": 0.0, "dir": 0.0}
    defender = {"x": 15.0, "y": 20.0, "s": 4.0, "dir": 90.0}
    assert g.closing_speed_proxy(carrier, defender) == pytest.approx(-4.0)


def test_pursuit_angle_zero_when_motion_points_at_carrier():
    carrier = {"x": 10.0, "y": 20.0}
    defender = {"x": 15.0, "y": 20.0, "s": 4.0, "dir": 270.0}
    assert g.pursuit_angle_error_deg(carrier, defender) == pytest.approx(0.0)


def test_pursuit_angle_is_geometry_not_lead_angle_claim():
    carrier = {"x": 10.0, "y": 20.0}
    defender = {"x": 15.0, "y": 20.0, "s": 4.0, "dir": 0.0}
    assert g.pursuit_angle_error_deg(carrier, defender) == pytest.approx(90.0)


def test_sideline_distance():
    assert g.sideline_distance_yards(0.0) == pytest.approx(0.0)
    assert g.sideline_distance_yards(10.0) == pytest.approx(10.0)
    assert g.sideline_distance_yards(50.0) == pytest.approx(3.3)
    assert g.sideline_distance_yards(60.0) is None


def test_source_overlap_breakdown_preserves_semantics():
    out = g.source_overlap_breakdown({20, 21}, {20}, {30}, {21})
    assert out == {
        "first_contact_primary_tackle_overlap": 1,
        "first_contact_assist_overlap": 0,
        "first_contact_missed_tackle_overlap": 1,
        "first_contact_any_source_label_overlap": 1,
    }


def test_missing_direction_abstains_instead_of_guessing():
    carrier = {"x": 10.0, "y": 20.0, "s": 3.0, "dir": None}
    defender = {"x": 12.0, "y": 20.0, "s": 4.0, "dir": 270.0}
    assert g.closing_speed_proxy(carrier, defender) is None
