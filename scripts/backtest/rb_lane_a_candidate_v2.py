"""RB Lane A -- candidate mechanism V2 (production-scoreable recipient universe).

Implements the "V2 recipient universe" amendment in
``docs/research/RB_LANE_A_TRANSITION_GATED_ALLOCATION_V2_PLAN.md``, frozen
after V1's terminal ``RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE`` (both
rotations, run ``35167213629``, plan-freeze commit ``dfcd45ed``). V1 is not
being rescued and remains immutable; this module is additive, importing
V1's unchanged pieces from ``rb_lane_a_candidate_v1`` and adding only the
recipient-eligibility filter and its two pre-outcome integrity checks that
the V2 plan specifies.

The only scientific change from V1: a scored transition team-week's carry
pool is reallocated only among "eligible recipients" -- active-room RB-room
players who also have an exact-identity, finite match in the dual-market
promotion comparator (finite ``promotion_rush_yards``, finite
``promotion_rush_att > 0.20``). The pre-transition-room HHI, the exponent
``p = 1 + 2H``, the conservation pool, and the rush-yard translation are
unchanged from V1 and reused unmodified.
"""
from __future__ import annotations

import pandas as pd

from scripts.backtest.rb_lane_a_candidate_v1 import (
    CONSTRUCTIBILITY_IDENTITY_KEYS,
    RUSH_ATT_CONSTRUCTIBILITY_FLOOR,
)

V2_RECIPIENT_UNIVERSE_FAILURE = "V2_RECIPIENT_UNIVERSE_FAILURE"
V2_RECIPIENT_UNIVERSE_OK = "V2_RECIPIENT_UNIVERSE_OK"
V2_RECIPIENT_WEIGHT_FAILURE = "V2_RECIPIENT_WEIGHT_FAILURE"
V2_RECIPIENT_WEIGHT_OK = "V2_RECIPIENT_WEIGHT_OK"


def filter_eligible_recipients(
    active_room: pd.DataFrame, dual_market_comparator: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V2 eligible-recipient intersection (plan section "V2 recipient
    universe"): ``post_transition_active_rb_room`` intersected with the
    ``production_scoreable_dual_market_rb_universe``, on the exact identity
    ``(season, week, team, player_clean_key)``, requiring a finite
    ``promotion_rush_yards`` and a finite ``promotion_rush_att > 0.20``.

    No fuzzy identity join, imputation, fallback efficiency, clipping, row
    synthesis, or projection fabrication is performed. Returns
    ``(eligible_room, excluded_room)`` -- the excluded rows are disclosed,
    never silently dropped.
    """
    required = set(CONSTRUCTIBILITY_IDENTITY_KEYS)
    missing_active = required - set(active_room.columns)
    if missing_active:
        raise RuntimeError(f"filter_eligible_recipients: active_room missing {sorted(missing_active)}")
    missing_comp = (required | {"promotion_rush_att", "promotion_rush_yards"}) - set(
        dual_market_comparator.columns
    )
    if missing_comp:
        raise RuntimeError(f"filter_eligible_recipients: dual_market_comparator missing {sorted(missing_comp)}")

    comparator = dual_market_comparator[
        CONSTRUCTIBILITY_IDENTITY_KEYS + ["promotion_rush_att", "promotion_rush_yards"]
    ].drop_duplicates(CONSTRUCTIBILITY_IDENTITY_KEYS).copy()
    rush_att = pd.to_numeric(comparator["promotion_rush_att"], errors="coerce")
    rush_yards = pd.to_numeric(comparator["promotion_rush_yards"], errors="coerce")
    comparator = comparator.loc[
        rush_yards.notna() & rush_att.notna() & (rush_att > RUSH_ATT_CONSTRUCTIBILITY_FLOOR)
    ]

    merged = active_room.merge(
        comparator[CONSTRUCTIBILITY_IDENTITY_KEYS],
        on=CONSTRUCTIBILITY_IDENTITY_KEYS,
        how="left",
        indicator=True,
    )
    eligible = merged.loc[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()
    excluded = merged.loc[merged["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
    return eligible, excluded


def check_recipient_universe_integrity(event_key: dict, eligible_room: pd.DataFrame) -> dict:
    """V2 "Team-week coverage rule": fail closed with
    ``V2_RECIPIENT_UNIVERSE_FAILURE`` if a scored transition team-week has
    zero eligible recipients -- evaluated outcome-blind, before any outcome
    is opened, per event.
    """
    n = int(len(eligible_room))
    return {
        **event_key,
        "disposition": V2_RECIPIENT_UNIVERSE_OK if n > 0 else V2_RECIPIENT_UNIVERSE_FAILURE,
        "n_eligible_recipients": n,
    }


def check_recipient_weight_integrity(event_key: dict, eligible_room_with_raw_w: pd.DataFrame) -> dict:
    """V2 hard assertion (plan section "V2 allocation formula"): the
    eligible-recipient ``raw_w`` sum must be positive, otherwise fail closed
    with ``V2_RECIPIENT_WEIGHT_FAILURE`` before outcomes -- distinct from
    V1's "all_zero_weights" disclosure-and-exclude pattern, since V2 treats
    a recipient-universe integrity problem as a stop condition, not a
    tolerable thin-history exclusion.
    """
    raw_sum = float(pd.to_numeric(eligible_room_with_raw_w.get("raw_w"), errors="coerce").fillna(0.0).sum())
    return {
        **event_key,
        "disposition": V2_RECIPIENT_WEIGHT_OK if raw_sum > 0 else V2_RECIPIENT_WEIGHT_FAILURE,
        "eligible_raw_w_sum": raw_sum,
    }


def aggregate_recipient_integrity_disposition(event_reports: list[dict], ok_label: str, failure_label: str) -> dict:
    """Aggregate a per-event integrity check across a whole rotation. Any
    single failing event fails the whole rotation closed, matching the
    fail-closed handling already used for Gate 0 / authority / Amendment-8
    constructibility in the V1 runner -- a rotation-level PASS requires
    every scored event to individually pass.
    """
    failing = [e for e in event_reports if e.get("disposition") == failure_label]
    return {
        "disposition": failure_label if failing else ok_label,
        "events_checked": len(event_reports),
        "events_failing": len(failing),
        "failing_events": failing,
    }
