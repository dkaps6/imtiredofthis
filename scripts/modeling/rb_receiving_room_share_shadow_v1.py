"""No-fit RB receiving-room share redistribution shadow V1.

Preserves total RB/FB target entitlement and all non-RB entitlement. Missing
history players retain their exact current room share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.modeling.rb_receiving_identity_runtime_v1 import _snapshot_queries

TOL = 1e-10
RB_POS = {"RB", "FB", "HB", "TB"}


def _pos(v) -> str:
    p = str(v or "").upper().strip()
    if p in {"HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p.startswith("FB"):
        return "FB"
    return p


def attach_prior_room_share_raw(
    frame: pd.DataFrame,
    *,
    season: int,
    week: int,
    states: pd.DataFrame,
    prev: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()
    if "player_clean_key" not in out.columns or "team" not in out.columns:
        raise RuntimeError("RB receiving-room shadow requires player_clean_key and team")
    q = out[["player_clean_key", "team"]].copy()
    q["season"] = int(season)
    q["week"] = int(week)
    feat = _snapshot_queries(q, states, prev)
    keep = ["player_clean_key", "team", "season", "week"]
    for c in ("prior_rb_room_share", "prior_games"):
        if c in feat.columns:
            keep.append(c)
    feat = feat[keep].copy()
    if "prior_rb_room_share" not in feat.columns:
        feat["prior_rb_room_share"] = np.nan
    if "prior_games" not in feat.columns:
        feat["prior_games"] = np.nan
    feat["prior_rb_room_share"] = pd.to_numeric(feat["prior_rb_room_share"], errors="coerce")
    feat["prior_games"] = pd.to_numeric(feat["prior_games"], errors="coerce")
    return out.merge(
        feat[["player_clean_key", "team", "prior_rb_room_share", "prior_games"]],
        on=["player_clean_key", "team"],
        how="left",
        validate="many_to_one",
    )


def apply_rb_receiving_room_share_shadow(
    frame: pd.DataFrame,
    *,
    season: int,
    week: int,
    states: pd.DataFrame,
    prev: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Redistribute only within the existing RB/FB target-entitlement pool."""
    if frame is None or frame.empty:
        raise RuntimeError("RB receiving-room shadow requires non-empty frame")
    required = {"event_id", "team", "player_clean_key", "position", "entitlement_tgt_share"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"RB receiving-room shadow missing columns: {sorted(missing)}")

    out = attach_prior_room_share_raw(
        frame, season=season, week=week, states=states, prev=prev
    )
    out["position_family"] = out["position"].map(_pos)
    out["entitlement_tgt_share"] = pd.to_numeric(
        out["entitlement_tgt_share"], errors="coerce"
    )
    if out["entitlement_tgt_share"].isna().any() or out["entitlement_tgt_share"].lt(0).any():
        raise RuntimeError("invalid entitlement target share before RB receiving-room shadow")

    out["rb_room_current_share"] = np.nan
    out["rb_room_candidate_share"] = np.nan
    out["rb_receiving_room_shadow_applied"] = False
    out["rb_receiving_room_history_available"] = False
    out["rb_receiving_room_fallback_reason"] = ""

    audit_rows = []
    before_all = out["entitlement_tgt_share"].copy()

    for (event_id, team), idx in out.groupby(["event_id", "team"], dropna=False).groups.items():
        g = out.loc[idx]
        rb_idx = g.index[g["position_family"].isin({"RB", "FB"})]
        if len(rb_idx) == 0:
            continue

        ent = out.loc[rb_idx, "entitlement_tgt_share"].astype(float)
        total = float(ent.sum())
        if total <= 0:
            out.loc[rb_idx, "rb_receiving_room_fallback_reason"] = "ZERO_RB_ENTITLEMENT_MASS"
            continue

        current = ent / total
        out.loc[rb_idx, "rb_room_current_share"] = current

        hist = pd.to_numeric(out.loc[rb_idx, "prior_rb_room_share"], errors="coerce")
        available = hist.notna() & np.isfinite(hist.to_numpy(float)) & hist.ge(0)
        out.loc[rb_idx, "rb_receiving_room_history_available"] = available.to_numpy(bool)

        missing_idx = rb_idx[~available.to_numpy(bool)]
        avail_idx = rb_idx[available.to_numpy(bool)]
        candidate = current.copy()

        if len(avail_idx) < 2:
            reason = "LT2_HISTORY_PLAYERS"
        else:
            h = hist.loc[avail_idx]
            hsum = float(h.sum())
            if not np.isfinite(hsum) or hsum <= 0:
                reason = "NONPOSITIVE_HISTORY_MASS"
            else:
                fixed_missing = float(current.loc[missing_idx].sum()) if len(missing_idx) else 0.0
                remaining = max(0.0, 1.0 - fixed_missing)
                candidate.loc[missing_idx] = current.loc[missing_idx]
                candidate.loc[avail_idx] = remaining * h / hsum
                reason = "APPLIED"
                out.loc[rb_idx, "rb_receiving_room_shadow_applied"] = True

        out.loc[rb_idx, "rb_room_candidate_share"] = candidate
        out.loc[rb_idx, "rb_receiving_room_fallback_reason"] = reason
        out.loc[rb_idx, "entitlement_tgt_share"] = total * candidate

        audit_rows.append({
            "season": int(season),
            "week": int(week),
            "event_id": str(event_id),
            "team": str(team),
            "rb_players": int(len(rb_idx)),
            "history_players": int(len(avail_idx)),
            "missing_history_players": int(len(missing_idx)),
            "rb_entitlement_mass_before": total,
            "rb_entitlement_mass_after": float(out.loc[rb_idx, "entitlement_tgt_share"].sum()),
            "shadow_applied": bool(reason == "APPLIED"),
            "fallback_reason": reason,
            "max_player_room_share_change": float((candidate - current).abs().max()),
        })

    audit = pd.DataFrame(audit_rows)

    # Strong invariants.
    before_team = (
        frame.groupby(["event_id", "team"], dropna=False)["entitlement_tgt_share"]
        .sum()
        .sort_index()
    )
    after_team = (
        out.groupby(["event_id", "team"], dropna=False)["entitlement_tgt_share"]
        .sum()
        .sort_index()
    )
    max_team_gap = float((after_team - before_team).abs().max()) if len(before_team) else 0.0

    rb_before = (
        frame.assign(position_family=frame["position"].map(_pos))
        .loc[lambda d: d["position_family"].isin({"RB", "FB"})]
        .groupby(["event_id", "team"], dropna=False)["entitlement_tgt_share"]
        .sum()
        .sort_index()
    )
    rb_after = (
        out.loc[out["position_family"].isin({"RB", "FB"})]
        .groupby(["event_id", "team"], dropna=False)["entitlement_tgt_share"]
        .sum()
        .sort_index()
    )
    max_rb_gap = float((rb_after - rb_before).abs().max()) if len(rb_before) else 0.0

    non_rb = ~out["position_family"].isin({"RB", "FB"})
    max_non_rb_gap = float(
        (
            out.loc[non_rb, "entitlement_tgt_share"].to_numpy(float)
            - before_all.loc[non_rb].to_numpy(float)
        ).max(initial=0.0)
    )
    max_non_rb_abs_gap = float(
        np.max(np.abs(
            out.loc[non_rb, "entitlement_tgt_share"].to_numpy(float)
            - before_all.loc[non_rb].to_numpy(float)
        )) if non_rb.any() else 0.0
    )

    if max_team_gap > TOL:
        raise RuntimeError(f"RB receiving-room shadow changed team entitlement mass: {max_team_gap}")
    if max_rb_gap > TOL:
        raise RuntimeError(f"RB receiving-room shadow changed RB entitlement mass: {max_rb_gap}")
    if max_non_rb_abs_gap > TOL:
        raise RuntimeError(f"RB receiving-room shadow changed non-RB entitlement: {max_non_rb_abs_gap}")

    summary = {
        "version": "RB_RECEIVING_ROOM_SHARE_SHADOW_V1",
        "season": int(season),
        "week": int(week),
        "rooms": int(len(audit)),
        "applied_rooms": int(audit["shadow_applied"].sum()) if len(audit) else 0,
        "max_team_entitlement_gap": max_team_gap,
        "max_rb_entitlement_gap": max_rb_gap,
        "max_non_rb_entitlement_gap": max_non_rb_abs_gap,
        "parameters_fit": 0,
        "sportsbook_inputs_used": False,
    }
    return out, audit, summary
