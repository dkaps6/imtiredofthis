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

import numpy as np
import pandas as pd

from scripts.backtest.evaluate_rb_stack2_enriched_allocation import (
    RB_POS as STACK2_RB_POS,
    add_team_competition,
    enrich_history,
    history_maps,
)

CONSTRUCTIBILITY_IDENTITY_KEYS = ["season", "week", "team", "player_clean_key"]
RUSH_ATT_CONSTRUCTIBILITY_FLOOR = 0.20
TRAILING_WINDOW = 3
CONCENTRATION_EXPONENT_BASE = 1
CONCENTRATION_EXPONENT_SLOPE = 2


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


def compute_historical_rb_room_rush_share(player_logs: pd.DataFrame, window: int = TRAILING_WINDOW) -> pd.DataFrame:
    """Team-level, strictly-prior trailing-window average of the RB/FB/HB room's
    share of the team's realized rush attempts -- same trailing-window
    convention as STACK2's own `prior3_*` features, reused for consistency
    (Amendment 1, "Predicted pregame conservation pool"), but computed at
    team level rather than per-player since this feeds the team-level pool,
    not a per-player role weight.
    """
    logs = player_logs.copy()
    logs.columns = [str(c).strip().lower() for c in logs.columns]
    logs["season"] = pd.to_numeric(logs["season"], errors="coerce")
    logs["week"] = pd.to_numeric(logs["week"], errors="coerce")
    logs["rushes"] = pd.to_numeric(logs.get("rushes"), errors="coerce").fillna(0.0)
    pos = logs.get("position", pd.Series("", index=logs.index)).fillna("").astype(str).str.upper()

    team_totals = (
        logs.groupby(["season", "week", "team"], as_index=False)["rushes"]
        .sum()
        .rename(columns={"rushes": "team_total_rushes"})
    )
    rb_totals = (
        logs.loc[pos.isin(STACK2_RB_POS)]
        .groupby(["season", "week", "team"], as_index=False)["rushes"]
        .sum()
        .rename(columns={"rushes": "rb_room_rushes"})
    )
    merged = team_totals.merge(rb_totals, on=["season", "week", "team"], how="left")
    merged["rb_room_rushes"] = merged["rb_room_rushes"].fillna(0.0)
    merged["rb_room_share"] = np.where(
        merged["team_total_rushes"] > 0, merged["rb_room_rushes"] / merged["team_total_rushes"], 0.0
    )
    merged["order"] = merged["season"] * 100 + merged["week"]

    out = []
    for team, g in merged.groupby("team", sort=False):
        g = g.sort_values("order")
        orders = g["order"].to_numpy()
        shares = g["rb_room_share"].to_numpy()
        for i in range(len(g)):
            prior_mask = orders < orders[i]
            prior_shares = shares[prior_mask][-window:]
            share = float(prior_shares.mean()) if len(prior_shares) else float("nan")
            out.append(
                {
                    "season": int(g["season"].iloc[i]),
                    "week": int(g["week"].iloc[i]),
                    "team": team,
                    "historical_rb_room_rush_share": share,
                }
            )
    return pd.DataFrame(out)


def compute_conservation_pool(
    component_predictions: pd.DataFrame, historical_rb_room_rush_share: pd.DataFrame
) -> pd.DataFrame:
    """Amendment 1/2: `pool = mc_projected_plays * (1 - mc_dropback_rate) *
    historical_rb_room_rush_share`, team level, every team-week. Fails closed
    (raises) if `mc_projected_plays`/`mc_dropback_rate` are missing --
    these must already exist in `component_predictions` (built by
    `component_predictions.py::build_mc_predictions`, not recomputed here).
    """
    cp = component_predictions.copy()
    cp.columns = [str(c).strip().lower() for c in cp.columns]
    required = {"season", "week", "team", "mc_projected_plays", "mc_dropback_rate"}
    missing = required - set(cp.columns)
    if missing:
        raise RuntimeError(f"compute_conservation_pool: component_predictions missing {sorted(missing)}")

    team_week = cp[list(required)].drop_duplicates(["season", "week", "team"])
    merged = team_week.merge(
        historical_rb_room_rush_share, on=["season", "week", "team"], how="left"
    )
    merged["pool"] = (
        pd.to_numeric(merged["mc_projected_plays"], errors="coerce")
        * (1 - pd.to_numeric(merged["mc_dropback_rate"], errors="coerce"))
        * pd.to_numeric(merged["historical_rb_room_rush_share"], errors="coerce")
    )
    return merged


def compute_role_weights_and_hhi(
    active_room: pd.DataFrame,
    pre_transition_room: pd.DataFrame,
    player_logs: pd.DataFrame,
    player_snaps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Amendment 1 steps 1 and 4: per-player `raw_w_i = prior3_rb_share_i`
    for the active (post-transition) room, and per-team-week
    `H = prior_backfield_hhi` computed from the pre-transition room's own
    players' `prior3_rb_share` values (both reuse STACK2's existing
    `enrich_history()`/`add_team_competition()` unchanged, per Amendment 1
    point #3/Amendment 5's HHI-prose correction).

    ``active_room``/``pre_transition_room`` must each carry
    ``season, week, team, name_key`` -- the scored-transition team-week's
    active and pre-transition RB/FB rooms respectively (Gate 0's
    `harmonize_roster_membership` output, restricted to the relevant rows;
    ``pre_transition_room`` uses the transition detector's own
    ``prior_season``/``prior_week``, never an assumed ``week - 1``).

    `H` is computed at the pre-transition room's own coordinates
    (`prior_season`/`prior_week`), so `prior3_rb_share` there reflects
    history strictly before that week -- consistent with the frozen
    formula's "not dependent on any current-week row." This coordinate
    choice is a documented implementation reading of "pre-transition room,"
    not itself specified verbatim in the frozen text; reported as such.
    """
    logs = player_logs.copy()
    logs.columns = [str(c).strip().lower() for c in logs.columns]
    logs["rushes"] = pd.to_numeric(logs.get("rushes"), errors="coerce").fillna(0.0)
    logs["rush_yards"] = pd.to_numeric(logs.get("rush_yards"), errors="coerce").fillna(0.0)
    if "name_key" not in logs.columns:
        raise RuntimeError("compute_role_weights_and_hhi: player_logs missing name_key")
    # Team-level RB-position rush denominator needs the FULL weekly log (all
    # positions), same as STACK2's own load_weekly_logs -- never filter logs
    # down to RB rows before this point, or the share denominator breaks.
    logs["position"] = logs.get("position", pd.Series("", index=logs.index)).fillna("").astype(str).str.upper()
    team_rb = (
        logs.loc[logs["position"].isin(STACK2_RB_POS)]
        .groupby(["season", "week", "team"], as_index=False)["rushes"]
        .sum()
        .rename(columns={"rushes": "team_rb_carries"})
    )
    logs = logs.merge(team_rb, on=["season", "week", "team"], how="left")
    logs["team_rb_carries"] = logs["team_rb_carries"].fillna(0.0)
    logs["rb_share"] = np.where(
        logs["position"].isin(STACK2_RB_POS) & logs["team_rb_carries"].gt(0),
        logs["rushes"] / logs["team_rb_carries"],
        0.0,
    )
    logs["order"] = pd.to_numeric(logs["season"], errors="coerce") * 100 + pd.to_numeric(
        logs["week"], errors="coerce"
    )

    snaps = player_snaps if player_snaps is not None and not player_snaps.empty else pd.DataFrame(
        columns=["name_key", "order"]
    )

    active = active_room.copy()
    active.columns = [str(c).strip().lower() for c in active.columns]
    active_enriched = enrich_history(active, logs, snaps)
    active_enriched["raw_w"] = pd.to_numeric(active_enriched["prior3_rb_share"], errors="coerce").fillna(0.0)

    pre = pre_transition_room.copy()
    pre.columns = [str(c).strip().lower() for c in pre.columns]
    pre_enriched = enrich_history(pre, logs, snaps)
    # add_team_competition() also computes injured_comp_count/depth_back_count
    # from injury_out_doubtful/depth_present -- Lane A's HHI formula only
    # consumes prior_backfield_hhi (computed purely from prior3_rb_share), so
    # these are neutral placeholders, not real signal, needed only so the
    # reused function doesn't crash on missing columns it doesn't get from
    # enrich_history() alone (those come from STACK2's own finalize_features()
    # in its own pipeline, not reused here).
    pre_enriched["injury_out_doubtful"] = 0.0
    pre_enriched["depth_present"] = 0.0
    pre_enriched = add_team_competition(pre_enriched)
    # add_team_competition computes prior_backfield_hhi identically for every
    # row within a (season,week,team) group -- take the group-level value.
    hhi_lookup = (
        pre_enriched[["season", "week", "team", "prior_backfield_hhi"]]
        .drop_duplicates(["season", "week", "team"])
        .rename(columns={"season": "transition_season", "week": "transition_week"})
    )

    return active_enriched, hhi_lookup


def compute_hhi_dampened_reallocation(
    active_room_with_raw_w: pd.DataFrame, hhi_lookup: pd.DataFrame, pool_row: dict
) -> pd.DataFrame:
    """Amendment 1 steps 2-8: explicit normalization, HHI concentration
    exponent, recipient weights, final allocation shares, reallocated
    predicted carries. `pool_row` is one row of `compute_conservation_pool`'s
    output (season, week, team, pool). Returns one row per active-room
    player with `candidate_att`, or an empty frame with
    `all_zero_weights=True` if the whole active room is history-less
    (Amendment 1 step 3 -- excluded from the transition subpopulation,
    reported not silently dropped).
    """
    room = active_room_with_raw_w.copy()
    raw_sum = float(room["raw_w"].sum())
    if raw_sum <= 0:
        return pd.DataFrame(), {"all_zero_weights": True, "team": pool_row.get("team"), "week": pool_row.get("week")}

    room["w"] = room["raw_w"] / raw_sum
    assert abs(float(room["w"].sum()) - 1.0) < 1e-9, "role-weight normalization must sum to 1.0 exactly"

    match = hhi_lookup.loc[
        (hhi_lookup["transition_season"] == pool_row["season"])
        & (hhi_lookup["transition_week"] == pool_row["week"])
        & (hhi_lookup["team"] == pool_row["team"])
    ]
    H = float(match["prior_backfield_hhi"].iloc[0]) if len(match) else 0.0
    p = CONCENTRATION_EXPONENT_BASE + CONCENTRATION_EXPONENT_SLOPE * H

    room["v"] = room["w"].clip(lower=0) ** p
    v_sum = float(room["v"].sum())
    room["recipient_share"] = room["v"] / v_sum if v_sum > 0 else 0.0
    room["candidate_att"] = room["recipient_share"] * float(pool_row["pool"])

    conservation_delta = abs(float(room["candidate_att"].sum()) - float(pool_row["pool"]))
    assert conservation_delta < 1e-6, f"conservation identity violated: delta={conservation_delta}"

    room["season"] = pool_row["season"]
    room["week"] = pool_row["week"]
    room["team"] = pool_row["team"]
    room["concentration_H"] = H
    room["concentration_exponent_p"] = p
    return room, {"all_zero_weights": False, "team": pool_row.get("team"), "week": pool_row.get("week")}


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
