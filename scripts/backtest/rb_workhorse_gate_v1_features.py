"""RB Workhorse-Transition-Gate V1 -- frozen 13-feature event builder.

Implements the frozen 13-feature contract from
``docs/research/RB_WORKHORSE_TRANSITION_GATE_V1_PLAN.md`` Section 7, as
amended by Section 15's ambiguity resolutions. Every scored loss/vacancy
event must produce one complete finite 13-feature row, or the WHOLE
experiment fails closed (Section 15 point 2, Section 16) -- there is no
per-event exclusion path anywhere in this module.

Reuses, unchanged, the already-audited building blocks:

* ``rb_lane_a_candidate_v1.build_active_and_pre_transition_rooms`` for the
  exact active/pre-transition room construction (including the corrected
  HHI temporal-coordinate contract).
* ``rb_lane_a_candidate_v1.compute_role_weights_and_hhi`` (STACK2's
  ``enrich_history``/``add_team_competition`` under the hood) for the
  active room's own per-player ``prior3_rb_share`` and the pre-transition
  room's HHI. Called a SECOND time with the pre-transition room in the
  ``active_room`` argument slot (Section 15 point 4) to mechanically expose
  that same room's own per-player ``prior3_rb_share`` for the
  ``departed_room_*`` features -- no formula or semantic change, purely
  reusing the existing function's own return value for a different input.
* ``rb_lane_a_candidate_v1.compute_historical_rb_room_rush_share`` and
  ``compute_conservation_pool`` for the already-computed team-week
  ``mc_projected_plays``/``mc_dropback_rate``/``historical_rb_room_rush_share``
  triple -- only their already-computed columns are read here, not the
  pool formula itself.

All raw MC component values come from canonical same-job Build A only
(Section 15 point 3); Build B is never read by this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtest.rb_lane_a_candidate_v1 import (
    CONSTRUCTIBILITY_IDENTITY_KEYS,
    build_active_and_pre_transition_rooms,
    compute_conservation_pool,
    compute_historical_rb_room_rush_share,
    compute_role_weights_and_hhi,
)

FEATURE_COLUMNS = [
    "active_top_prior3_rb_share",
    "active_second_prior3_rb_share",
    "pre_transition_backfield_hhi",
    "departed_room_prior3_share_sum",
    "departed_room_max_prior3_share",
    "active_rb_room_size",
    "prior_rb_room_size",
    "mc_projected_plays",
    "mc_dropback_rate",
    "raw_mc_team_rush_volume",
    "historical_rb_room_rush_share",
    "raw_mc_top_active_rush_att",
    "raw_mc_second_active_rush_att",
]

RUSH_ATT_MARKET = "rush_att"

WHOLE_EXPERIMENT_FAIL_CLOSED = "WORKHORSE_GATE_V1_WHOLE_EXPERIMENT_FAIL_CLOSED"
FEATURES_CONSTRUCTIBLE = "WORKHORSE_GATE_V1_FEATURES_CONSTRUCTIBLE"


def _build_a_rush_att_rows(component_predictions_build_a: pd.DataFrame) -> pd.DataFrame:
    """Canonical Build-A raw ``mc_proj`` rows for ``market == rush_att``, one
    row per ``(season, week, team, player_clean_key)``. Build B is never
    read here (Section 15 point 3).
    """
    cp = component_predictions_build_a.copy()
    cp.columns = [str(c).strip().lower() for c in cp.columns]
    market = cp.get("market", pd.Series("", index=cp.index)).fillna("").astype(str).str.lower()
    x = cp.loc[market.eq(RUSH_ATT_MARKET)].copy()
    required = set(CONSTRUCTIBILITY_IDENTITY_KEYS) | {"mc_proj"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"_build_a_rush_att_rows: component_predictions missing {sorted(missing)}")
    x = x[CONSTRUCTIBILITY_IDENTITY_KEYS + ["mc_proj"]].copy()
    if x.duplicated(CONSTRUCTIBILITY_IDENTITY_KEYS).any():
        raise RuntimeError("_build_a_rush_att_rows: duplicate player identities in Build-A rush_att rows")
    return x


def _team_week_mc_frame(component_predictions_build_a: pd.DataFrame, player_logs: pd.DataFrame) -> pd.DataFrame:
    """``season/week/team -> mc_projected_plays, mc_dropback_rate,
    historical_rb_room_rush_share, raw_mc_team_rush_volume``. Reuses
    ``compute_historical_rb_room_rush_share``/``compute_conservation_pool``
    unchanged -- only their already-computed team-week columns are read;
    the conservation-pool formula itself is not used here.
    """
    room_share = compute_historical_rb_room_rush_share(player_logs)
    merged = compute_conservation_pool(component_predictions_build_a, room_share)
    merged["raw_mc_team_rush_volume"] = pd.to_numeric(
        merged["mc_projected_plays"], errors="coerce"
    ) * (1 - pd.to_numeric(merged["mc_dropback_rate"], errors="coerce"))
    return merged[
        [
            "season", "week", "team", "mc_projected_plays", "mc_dropback_rate",
            "historical_rb_room_rush_share", "raw_mc_team_rush_volume",
        ]
    ]


def build_event_features(
    *,
    scored_events: pd.DataFrame,
    roster_state: pd.DataFrame,
    player_logs: pd.DataFrame,
    component_predictions_build_a: pd.DataFrame,
) -> dict:
    """Build the frozen 13-feature row for every scored event.

    Fails the WHOLE experiment closed (Section 15 point 2) rather than
    excluding any single event -- every scored event is checked, and every
    failing event's key and reason are preserved, before the overall
    disposition is decided.

    ``scored_events`` must carry ``season, week, team, prior_season,
    prior_week`` (``rb_lane_a_transition_detector_v1.
    build_scored_v1_event_population``'s exact output shape).
    ``roster_state`` is Gate 0's harmonized roster
    (``harmonize_roster_membership``), carrying ``player_clean_key``/``name_key``.
    ``component_predictions_build_a`` is the canonical same-job Build-A
    component-prediction frame -- never Build B.
    """
    active_room, pre_transition_room = build_active_and_pre_transition_rooms(scored_events, roster_state)
    rush_att_rows = _build_a_rush_att_rows(component_predictions_build_a)
    team_week_mc = _team_week_mc_frame(component_predictions_build_a, player_logs)

    failing: list[dict] = []
    feature_rows: list[dict] = []

    for e in scored_events.itertuples(index=False):
        key = {"season": int(e.season), "week": int(e.week), "team": e.team}

        active = active_room.loc[
            active_room["season"].eq(e.season) & active_room["week"].eq(e.week) & active_room["team"].eq(e.team)
        ]
        pre = pre_transition_room.loc[
            pre_transition_room["season"].eq(e.season)
            & pre_transition_room["week"].eq(e.week)
            & pre_transition_room["team"].eq(e.team)
        ]

        if len(active) == 0:
            failing.append({**key, "reason": "zero_active_room_members"})
            continue

        active_enriched, hhi_lookup = compute_role_weights_and_hhi(active, pre, player_logs)
        shares = active_enriched.sort_values("raw_w", ascending=False).reset_index(drop=True)
        top_share = float(shares.iloc[0]["raw_w"])
        second_share = float(shares.iloc[1]["raw_w"]) if len(shares) > 1 else 0.0

        h_match = hhi_lookup.loc[
            hhi_lookup["season"].eq(e.season) & hhi_lookup["week"].eq(e.week) & hhi_lookup["team"].eq(e.team)
        ]
        hhi = float(h_match.iloc[0]["prior_backfield_hhi"]) if len(h_match) else 0.0

        # Departed room per-player prior3_rb_share: mechanically reuse
        # compute_role_weights_and_hhi a second time with the pre-transition
        # room in the active_room argument slot (Section 15 point 4) -- no
        # formula change, just exposing the same helper's own output for a
        # different room.
        active_keys = set(active["player_clean_key"])
        departed = pre.loc[~pre["player_clean_key"].isin(active_keys)]
        if len(departed):
            departed_enriched, _ = compute_role_weights_and_hhi(departed, pre, player_logs)
            departed_shares = pd.to_numeric(departed_enriched["raw_w"], errors="coerce").fillna(0.0)
            departed_sum = float(departed_shares.sum())
            departed_max = float(departed_shares.max())
        else:
            departed_sum = 0.0
            departed_max = 0.0

        mc_match = team_week_mc.loc[
            team_week_mc["season"].eq(e.season) & team_week_mc["week"].eq(e.week) & team_week_mc["team"].eq(e.team)
        ]
        if len(mc_match) == 0 or mc_match[
            ["mc_projected_plays", "mc_dropback_rate", "historical_rb_room_rush_share"]
        ].isna().to_numpy().any():
            failing.append({**key, "reason": "missing_team_week_mc_or_history_share"})
            continue
        mc_row = mc_match.iloc[0]

        active_rush = active[["season", "week", "team", "player_clean_key"]].merge(
            rush_att_rows, on=CONSTRUCTIBILITY_IDENTITY_KEYS, how="inner"
        )
        if len(active_rush) == 0:
            failing.append({**key, "reason": "zero_active_players_matched_to_build_a_rush_att"})
            continue
        active_rush_sorted = active_rush.sort_values("mc_proj", ascending=False).reset_index(drop=True)
        top_rush = float(active_rush_sorted.iloc[0]["mc_proj"])
        second_rush = float(active_rush_sorted.iloc[1]["mc_proj"]) if len(active_rush_sorted) > 1 else 0.0

        row = {
            **key,
            "active_top_prior3_rb_share": top_share,
            "active_second_prior3_rb_share": second_share,
            "pre_transition_backfield_hhi": hhi,
            "departed_room_prior3_share_sum": departed_sum,
            "departed_room_max_prior3_share": departed_max,
            "active_rb_room_size": int(len(active)),
            "prior_rb_room_size": int(len(pre)),
            "mc_projected_plays": float(mc_row["mc_projected_plays"]),
            "mc_dropback_rate": float(mc_row["mc_dropback_rate"]),
            "raw_mc_team_rush_volume": float(mc_row["raw_mc_team_rush_volume"]),
            "historical_rb_room_rush_share": float(mc_row["historical_rb_room_rush_share"]),
            "raw_mc_top_active_rush_att": top_rush,
            "raw_mc_second_active_rush_att": second_rush,
        }
        if not all(np.isfinite(row[c]) for c in FEATURE_COLUMNS):
            failing.append({**key, "reason": "nonfinite_feature_value"})
            continue
        feature_rows.append(row)

    if failing:
        return {
            "disposition": WHOLE_EXPERIMENT_FAIL_CLOSED,
            "events_checked": int(len(scored_events)),
            "events_failing": len(failing),
            "failing_events": failing,
            "feature_rows": pd.DataFrame(),
        }
    return {
        "disposition": FEATURES_CONSTRUCTIBLE,
        "events_checked": int(len(scored_events)),
        "events_failing": 0,
        "failing_events": [],
        "feature_rows": pd.DataFrame(feature_rows, columns=["season", "week", "team"] + FEATURE_COLUMNS),
    }
