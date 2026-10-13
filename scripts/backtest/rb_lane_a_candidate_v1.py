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

    ``active_room`` must carry ``season, week, team, name_key`` at the
    scored-transition team-week's own coordinates (Gate 0's
    `harmonize_roster_membership` output, restricted to the relevant rows).

    ``pre_transition_room`` must carry the **same transition-week**
    ``season, week, team`` coordinates (not the prior week's), with
    membership drawn from the transition detector's own
    ``prior_season``/``prior_week`` roster snapshot -- i.e. the caller
    re-tags the pre-transition roster rows onto the transition week before
    calling this function. This makes `enrich_history()`'s own "strictly
    before this row's (season,week)" cutoff naturally compute each
    pre-transition player's `prior3_rb_share` using history strictly before
    the *transition* week (never the current-week outcome), and makes the
    resulting `H` lookup key directly match the transition week's own
    `(season, week, team)`, the same coordinates `compute_conservation_pool`
    produces. This coordinate choice is a documented implementation reading
    of "pre-transition room," not itself specified verbatim in the frozen
    text; reported as such.
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
    hhi_lookup = pre_enriched[["season", "week", "team", "prior_backfield_hhi"]].drop_duplicates(
        ["season", "week", "team"]
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
        (hhi_lookup["season"] == pool_row["season"])
        & (hhi_lookup["week"] == pool_row["week"])
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


# ---------------------------------------------------------------------------
# Two candidate arms (Amendment 4)
# ---------------------------------------------------------------------------


def build_deployable_candidate(
    all_rush_yards_rows: pd.DataFrame, scored_transition_candidate: pd.DataFrame
) -> pd.DataFrame:
    """Amendment 4: `deployable_candidate` -- the only arm with any bearing on
    a future production question.

    On a scored V1 transition week, equal to
    `scored_transition_candidate`'s `candidate_rush_yards`. On every other
    week (stable weeks, and detected-but-not-scored transition weeks alike),
    equal to the promotion comparator (`promotion_rush_yards`) --
    byte-identical, by construction, never the mechanism comparator.

    `all_rush_yards_rows` must carry `CONSTRUCTIBILITY_IDENTITY_KEYS` +
    `promotion_rush_yards` for **every** row (the universe this arm is
    defined over). `scored_transition_candidate` must carry the same keys +
    `candidate_rush_yards`, for scored-transition rows only (a subset).
    """
    required = set(CONSTRUCTIBILITY_IDENTITY_KEYS) | {"promotion_rush_yards"}
    missing = required - set(all_rush_yards_rows.columns)
    if missing:
        raise RuntimeError(f"build_deployable_candidate: all_rush_yards_rows missing {sorted(missing)}")

    out = all_rush_yards_rows[CONSTRUCTIBILITY_IDENTITY_KEYS + ["promotion_rush_yards"]].copy()
    scored = scored_transition_candidate[
        CONSTRUCTIBILITY_IDENTITY_KEYS + ["candidate_rush_yards"]
    ].drop_duplicates(CONSTRUCTIBILITY_IDENTITY_KEYS)
    out = out.merge(scored, on=CONSTRUCTIBILITY_IDENTITY_KEYS, how="left")
    out["is_scored_v1_transition_row"] = out["candidate_rush_yards"].notna()
    out["deployable_candidate_rush_yards"] = out["candidate_rush_yards"].where(
        out["is_scored_v1_transition_row"], out["promotion_rush_yards"]
    )
    return out.drop(columns=["candidate_rush_yards"])


def build_mechanism_diagnostic(
    scored_transition_candidate: pd.DataFrame, mechanism_comparator: pd.DataFrame
) -> pd.DataFrame:
    """Amendment 4: `mechanism_diagnostic` -- informative only, never a
    candidate for production. Scored V1 transition weeks only, compared
    directly against the mechanism comparator (P3/STACK2). Per Amendment 7,
    only constructible for Rotation 2 -- callers must not invoke this for
    Rotation 1 (the mechanism comparator itself is `NOT_CONSTRUCTIBLE_
    NO_CASEBOOK` there).

    `mechanism_comparator` must carry `season, week, team, name_key,
    arch_enriched_opp_stack_eff_yards` (the STACK2 casebook's own column,
    per `rb_lane_a_comparator_reconstruction_v1.MECHANISM_COMPARATOR_COLUMN`).
    Joins on `name_key` (STACK2's own identity scheme), not `player_clean_key`
    -- `scored_transition_candidate` must therefore also carry `name_key`.
    """
    mech_col = "arch_enriched_opp_stack_eff_yards"
    required_candidate = {"season", "week", "team", "name_key", "candidate_rush_yards"}
    missing_candidate = required_candidate - set(scored_transition_candidate.columns)
    if missing_candidate:
        raise RuntimeError(f"build_mechanism_diagnostic: candidate frame missing {sorted(missing_candidate)}")
    required_mech = {"season", "week", "team", "name_key", mech_col}
    missing_mech = required_mech - set(mechanism_comparator.columns)
    if missing_mech:
        raise RuntimeError(f"build_mechanism_diagnostic: mechanism_comparator missing {sorted(missing_mech)}")

    out = scored_transition_candidate.merge(
        mechanism_comparator[["season", "week", "team", "name_key", mech_col]],
        on=["season", "week", "team", "name_key"],
        how="left",
    )
    out = out.rename(columns={mech_col: "mechanism_comparator_rush_yards"})
    return out


def check_stable_identity_gate(deployable_candidate: pd.DataFrame) -> dict:
    """Amendment 4's hard stable-identity gate: for every non-scored row
    (stable weeks, and detected-but-not-scored transition weeks), the
    deployable candidate must equal the promotion comparator EXACTLY --
    `max(abs(deployable_candidate_rush_yards - promotion_comparator_rush_yards)) == 0.0`.
    A failure here means the deployable arm was built wrong, not that the
    candidate is weak.
    """
    required = {"is_scored_v1_transition_row", "deployable_candidate_rush_yards", "promotion_rush_yards"}
    missing = required - set(deployable_candidate.columns)
    if missing:
        raise RuntimeError(f"check_stable_identity_gate: missing columns {sorted(missing)}")

    non_scored = deployable_candidate.loc[~deployable_candidate["is_scored_v1_transition_row"]]
    delta = (
        pd.to_numeric(non_scored["deployable_candidate_rush_yards"], errors="coerce")
        - pd.to_numeric(non_scored["promotion_rush_yards"], errors="coerce")
    ).abs()
    max_delta = float(delta.max()) if len(delta) else 0.0
    return {
        "disposition": "STABLE_IDENTITY_GATE_PASS" if max_delta == 0.0 else "STABLE_IDENTITY_GATE_FAILURE",
        "rows_checked": int(len(non_scored)),
        "max_abs_delta": max_delta,
    }


# ---------------------------------------------------------------------------
# Orchestration: per-event room construction + reallocation across a rotation
# ---------------------------------------------------------------------------


def build_active_and_pre_transition_rooms(
    scored_events: pd.DataFrame, roster_state: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For every scored V1 event (season, week, team, prior_season,
    prior_week), extract the active (post-transition) room at the event's
    own coordinates and the pre-transition room re-tagged onto those same
    coordinates (membership from `prior_season`/`prior_week`, per
    `compute_role_weights_and_hhi`'s corrected contract). Both outputs carry
    `name_key` and `player_clean_key` from `roster_state` (Gate 0's
    `harmonize_roster_membership`, extended with `player_clean_key`).
    """
    required = {"season", "week", "team", "name_key", "player_clean_key"}
    missing = required - set(roster_state.columns)
    if missing:
        raise RuntimeError(f"build_active_and_pre_transition_rooms: roster_state missing {sorted(missing)}")

    active_frames, pre_frames = [], []
    for ev in scored_events.itertuples(index=False):
        active = roster_state.loc[
            (roster_state["season"] == ev.season)
            & (roster_state["week"] == ev.week)
            & (roster_state["team"] == ev.team)
        ]
        active_frames.append(active)

        pre = roster_state.loc[
            (roster_state["season"] == ev.prior_season)
            & (roster_state["week"] == ev.prior_week)
            & (roster_state["team"] == ev.team)
        ].copy()
        pre["season"] = ev.season
        pre["week"] = ev.week
        pre_frames.append(pre)

    active_room = (
        pd.concat(active_frames, ignore_index=True).drop_duplicates(["season", "week", "team", "player_clean_key"])
        if active_frames
        else roster_state.iloc[0:0].copy()
    )
    pre_transition_room = (
        pd.concat(pre_frames, ignore_index=True).drop_duplicates(["season", "week", "team", "player_clean_key"])
        if pre_frames
        else roster_state.iloc[0:0].copy()
    )
    return active_room, pre_transition_room


def run_candidate_mechanism_for_rotation(
    *,
    scored_events: pd.DataFrame,
    roster_state: pd.DataFrame,
    player_logs: pd.DataFrame,
    component_predictions: pd.DataFrame,
    rotation: int,
) -> dict:
    """Full "Candidate mechanism" pipeline (Amendments 1-2, 5, 8) for one
    rotation: conservation pool -> role weights/HHI -> HHI-dampened
    reallocation -> Amendment-8 translation to rush yards, across every
    scored V1 transition team-week. Returns the per-player candidate rows
    plus disclosure of any team-weeks excluded for all-zero role weights
    (Amendment 1 step 3 -- never silently dropped).
    """
    from scripts.backtest.rb_lane_a_comparator_reconstruction_v1 import (
        build_dual_market_promotion_comparator,
    )

    if scored_events.empty:
        return {
            "candidate_rows": pd.DataFrame(),
            "excluded_all_zero_weight_team_weeks": [],
            "dual_market_comparator": pd.DataFrame(),
        }

    active_room, pre_transition_room = build_active_and_pre_transition_rooms(scored_events, roster_state)
    active_enriched, hhi_lookup = compute_role_weights_and_hhi(active_room, pre_transition_room, player_logs)

    historical_share = compute_historical_rb_room_rush_share(player_logs)
    pool = compute_conservation_pool(component_predictions, historical_share)
    pool_by_team_week = {
        (int(r["season"]), int(r["week"]), r["team"]): r for _, r in pool.iterrows()
    }

    dual_market = build_dual_market_promotion_comparator(component_predictions, rotation)

    candidate_frames = []
    excluded = []
    for (season, week, team), room_group in active_enriched.groupby(["season", "week", "team"], sort=False):
        pool_row = pool_by_team_week.get((int(season), int(week), team))
        if pool_row is None or pd.isna(pool_row.get("pool")):
            excluded.append(
                {"season": int(season), "week": int(week), "team": team, "reason": "no_conservation_pool"}
            )
            continue
        realloc, meta = compute_hhi_dampened_reallocation(room_group, hhi_lookup, pool_row.to_dict())
        if meta["all_zero_weights"]:
            excluded.append({"season": int(season), "week": int(week), "team": team, "reason": "all_zero_weights"})
            continue
        candidate_frames.append(realloc)

    candidate_att = (
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    )
    if candidate_att.empty:
        return {
            "candidate_rows": pd.DataFrame(),
            "excluded_all_zero_weight_team_weeks": excluded,
            "dual_market_comparator": dual_market,
        }

    constructibility_input = candidate_att[["season", "week", "team", "player_clean_key"]].drop_duplicates()
    constructibility = check_rush_yard_translation_constructibility(constructibility_input, dual_market)
    if constructibility["disposition"] != "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE":
        return {
            "candidate_rows": pd.DataFrame(),
            "excluded_all_zero_weight_team_weeks": excluded,
            "dual_market_comparator": dual_market,
            "constructibility": constructibility,
        }

    incumbent_ypc = compute_incumbent_ypc(dual_market)
    translated = translate_candidate_rush_yards(candidate_att, incumbent_ypc)

    return {
        "candidate_rows": translated,
        "excluded_all_zero_weight_team_weeks": excluded,
        "dual_market_comparator": dual_market,
        "constructibility": constructibility,
    }
