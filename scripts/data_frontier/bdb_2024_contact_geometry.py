"""Deterministic Phase-0 geometry helpers for the BDB 2024 contact benchmark.

This module intentionally does not change the frozen V1 contact detector. It adds
quantities derivable from an already-selected carrier/defender frame pair.
"""
from __future__ import annotations

import math
from typing import Mapping

FIELD_WIDTH_YARDS = 53.3


def _finite(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _velocity(speed: object, direction_deg: object) -> tuple[float, float] | None:
    """Return (vx, vy) in tracking-field coordinates from NGS speed/direction.

    NGS direction is degrees clockwise from north/up-field-y. Therefore
    vx=s*sin(theta), vy=s*cos(theta). Only relative geometry is used here.
    """
    s = _finite(speed)
    d = _finite(direction_deg)
    if s is None or d is None:
        return None
    theta = math.radians(d)
    return s * math.sin(theta), s * math.cos(theta)


def closing_speed_proxy(carrier: Mapping, defender: Mapping) -> float | None:
    """Positive rate (yd/s) at which defender-carrier separation is shrinking.

    This is an instantaneous 2-D derivative from source x/y/s/dir. It is not a
    collision-force or football-quality metric. Negative values mean separation
    is increasing at that instant.
    """
    cx, cy = _finite(carrier.get("x")), _finite(carrier.get("y"))
    dx, dy = _finite(defender.get("x")), _finite(defender.get("y"))
    cv = _velocity(carrier.get("s"), carrier.get("dir"))
    dv = _velocity(defender.get("s"), defender.get("dir"))
    if None in (cx, cy, dx, dy) or cv is None or dv is None:
        return None
    rx, ry = dx - cx, dy - cy
    distance = math.hypot(rx, ry)
    if distance == 0:
        return None
    rvx, rvy = dv[0] - cv[0], dv[1] - cv[1]
    radial_rate = (rx * rvx + ry * rvy) / distance
    return -radial_rate


def pursuit_angle_error_deg(carrier: Mapping, defender: Mapping) -> float | None:
    """Absolute angle between defender motion and line from defender to carrier.

    0 degrees means the defender is moving directly toward the carrier's current
    location; 180 means directly away. This is a geometry proxy, not a claim
    about optimal pursuit because a moving carrier may require a lead angle.
    """
    cx, cy = _finite(carrier.get("x")), _finite(carrier.get("y"))
    dx, dy = _finite(defender.get("x")), _finite(defender.get("y"))
    dv = _velocity(defender.get("s"), defender.get("dir"))
    if None in (cx, cy, dx, dy) or dv is None:
        return None
    tx, ty = cx - dx, cy - dy
    tnorm = math.hypot(tx, ty)
    vnorm = math.hypot(*dv)
    if tnorm == 0 or vnorm == 0:
        return None
    cosine = max(-1.0, min(1.0, (tx * dv[0] + ty * dv[1]) / (tnorm * vnorm)))
    return math.degrees(math.acos(cosine))


def sideline_distance_yards(y: object) -> float | None:
    """Distance to nearest sideline from BDB y coordinate."""
    yy = _finite(y)
    if yy is None or yy < 0 or yy > FIELD_WIDTH_YARDS:
        return None
    return min(yy, FIELD_WIDTH_YARDS - yy)


def source_overlap_breakdown(
    contact_ids: set[int],
    primary_tackle_ids: set[int],
    assist_ids: set[int],
    missed_tackle_ids: set[int],
) -> dict[str, int]:
    """Separate contact-candidate overlap with each source semantic label."""
    return {
        "first_contact_primary_tackle_overlap": int(bool(contact_ids & primary_tackle_ids)),
        "first_contact_assist_overlap": int(bool(contact_ids & assist_ids)),
        "first_contact_missed_tackle_overlap": int(bool(contact_ids & missed_tackle_ids)),
        "first_contact_any_source_label_overlap": int(
            bool(contact_ids & (primary_tackle_ids | assist_ids | missed_tackle_ids))
        ),
    }
