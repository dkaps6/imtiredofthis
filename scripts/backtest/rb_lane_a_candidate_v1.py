"""RB Lane A -- candidate mechanism (transition-gated allocation V1).

Implements the frozen "Candidate mechanism" / "Rush-yard translation" /
"Two candidate arms" sections of
``docs/research/RB_LANE_A_TRANSITION_GATED_ALLOCATION_V1_PLAN.md``
(Amendments 1-4, 8). Computes NO candidate-vs-outcome comparison -- this
module only constructs the candidate's own output and the checks required
before any outcome is scored.

Per Amendment 8 (Issue #535 comment `5705529323`): the original P3
efficiency seam (STACK1 `stack_yards/stack_att`, M94C implied-YPC fallback)
is unconstructible for Rotation 1, so the rush-yard translation instead
holds each player's incumbent production-ensemble efficiency fixed --
``incumbent_ypc_i = promotion_rush_yards_i / promotion_rush_att_i`` -- and
only carry allocation varies on scored V1 transition weeks.
"""
from __future__ import annotations

import pandas as pd

CONSTRUCTIBILITY_IDENTITY_KEYS = ["season", "week", "team", "player_clean_key"]
RUSH_ATT_CONSTRUCTIBILITY_FLOOR = 0.20


def check_rush_yard_translation_constructibility(
    scored_active_rows: pd.DataFrame, dual_market_comparator: pd.DataFrame
) -> dict:
    """Amendment 8 step 3: fail-closed constructibility check, before any
    outcome is scored.

    ``scored_active_rows`` must contain exactly the active-room player rows
    for every scored-V1 transition team-week (season, week, team,
    player_clean_key), for one rotation. ``dual_market_comparator`` is the
    output of
    ``rb_lane_a_comparator_reconstruction_v1.build_dual_market_promotion_comparator``.

    Every required row must join to a finite ``promotion_rush_yards`` and a
    finite ``promotion_rush_att > 0.20``. No row is dropped, imputed,
    clipped, or given an invented fallback -- a failing row is reported and
    the whole check fails closed.
    """
    required = set(CONSTRUCTIBILITY_IDENTITY_KEYS)
    for label, frame in (
        ("scored_active_rows", scored_active_rows),
        ("dual_market_comparator", dual_market_comparator),
    ):
        missing = required - set(frame.columns)
        if missing:
            return {
                "disposition": "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE",
                "reason": f"{label} frame missing required columns: {sorted(missing)}",
            }
    missing_comparator_cols = {"promotion_rush_att", "promotion_rush_yards"} - set(
        dual_market_comparator.columns
    )
    if missing_comparator_cols:
        return {
            "disposition": "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE",
            "reason": f"dual_market_comparator missing columns: {sorted(missing_comparator_cols)}",
        }

    scored = scored_active_rows[CONSTRUCTIBILITY_IDENTITY_KEYS].drop_duplicates()
    comparator = dual_market_comparator[
        CONSTRUCTIBILITY_IDENTITY_KEYS + ["promotion_rush_att", "promotion_rush_yards"]
    ].drop_duplicates(CONSTRUCTIBILITY_IDENTITY_KEYS)

    merged = scored.merge(
        comparator, on=CONSTRUCTIBILITY_IDENTITY_KEYS, how="left", validate="one_to_one"
    )

    rush_att = pd.to_numeric(merged["promotion_rush_att"], errors="coerce")
    rush_yards = pd.to_numeric(merged["promotion_rush_yards"], errors="coerce")

    yards_ok = rush_yards.notna()
    att_ok = rush_att.notna() & (rush_att > RUSH_ATT_CONSTRUCTIBILITY_FLOOR)
    row_ok = yards_ok & att_ok

    failing = merged.loc[~row_ok, CONSTRUCTIBILITY_IDENTITY_KEYS + ["promotion_rush_att", "promotion_rush_yards"]]

    disposition = (
        "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE"
        if row_ok.all()
        else "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"
    )

    return {
        "disposition": disposition,
        "rows_checked": int(len(merged)),
        "rows_failing": int((~row_ok).sum()),
        "failing_rows": failing.to_dict(orient="records"),
    }


def compute_incumbent_ypc(dual_market_comparator: pd.DataFrame) -> pd.DataFrame:
    """Amendment 8 step 4: ``incumbent_ypc_i = promotion_rush_yards_i /
    promotion_rush_att_i``, no fitting or clipping. Assumes constructibility
    has already been checked and passed -- does not itself validate inputs.
    """
    out = dual_market_comparator.copy()
    out["incumbent_ypc"] = pd.to_numeric(out["promotion_rush_yards"], errors="coerce") / pd.to_numeric(
        out["promotion_rush_att"], errors="coerce"
    )
    return out


def translate_candidate_rush_yards(candidate_att: pd.DataFrame, incumbent_ypc: pd.DataFrame) -> pd.DataFrame:
    """Amendment 8 step 5: ``candidate_rush_yards_i = candidate_att_i *
    incumbent_ypc_i``.

    ``candidate_att`` must carry ``candidate_att`` per
    ``CONSTRUCTIBILITY_IDENTITY_KEYS``. ``incumbent_ypc`` must carry
    ``incumbent_ypc`` (output of ``compute_incumbent_ypc``) on the same keys.
    Fails closed (raises) on any row in ``candidate_att`` that doesn't join.
    """
    merged = candidate_att.merge(
        incumbent_ypc[CONSTRUCTIBILITY_IDENTITY_KEYS + ["incumbent_ypc"]],
        on=CONSTRUCTIBILITY_IDENTITY_KEYS,
        how="left",
        validate="one_to_one",
    )
    if merged["incumbent_ypc"].isna().any():
        missing = merged.loc[merged["incumbent_ypc"].isna(), CONSTRUCTIBILITY_IDENTITY_KEYS]
        raise RuntimeError(
            f"translate_candidate_rush_yards: {len(missing)} candidate rows have no "
            f"incumbent_ypc match -- constructibility check should have caught this: "
            f"{missing.to_dict(orient='records')[:5]}"
        )
    merged["candidate_rush_yards"] = pd.to_numeric(
        merged["candidate_att"], errors="coerce"
    ) * pd.to_numeric(merged["incumbent_ypc"], errors="coerce")
    return merged
