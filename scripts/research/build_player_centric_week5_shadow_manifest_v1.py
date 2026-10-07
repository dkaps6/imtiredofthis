#!/usr/bin/env python3
"""Build immutable Week-5 all-player player-centric shadow manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.player_identity_v3 import player_name_key

SEASON = 2026
WEEK = 5
SUPPORTED = {"QB", "RB", "FB", "HB", "TB", "WR", "LWR", "RWR", "SWR", "TE"}

RB_CARRY_RUN = 37560824479
RB_CARRY_DIGEST = "sha256:66a428f0c19ee1dc356db117fa8e39092204896356461358667224826826e90a"
WRTE_RUN = 37654382316
WRTE_DIGEST = "sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3"
RB_TARGET_RUN = 37696979325
RB_TARGET_DIGEST = "sha256:23ed4317a9e5acb12207a1f6cb67947f71a6fed50d915c711b08b08a1276d94e"

FORBIDDEN = (
    "actual", "result", "target_game", "week5_outcome",
    "sportsbook", "bookmaker", "prop_line", "market_line", "odds",
)


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    bad = [c for c in x.columns if any(token in c for token in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"{label} contains forbidden outcome/sportsbook columns: {bad}")
    return x


def _pos(v) -> str:
    p = str(v or "").upper().strip()
    if p in {"RB", "HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p == "FB" or p.startswith("FB"):
        return "FB"
    if p.startswith("QB"):
        return "QB"
    if p in {"WR", "LWR", "RWR", "SWR"} or p.startswith("WR"):
        return "WR"
    if p.startswith("TE"):
        return "TE"
    return p


def _event(team, opponent) -> str:
    a = canon_team(team)
    b = canon_team(opponent)
    return "|".join(sorted([a, b]))


def _clean_key(v) -> str:
    _, key = canonicalize_player_name_safe(v)
    return str(key or "")


def _base_key(v) -> str:
    return player_name_key(v, strip_suffix=True)


def _digest_rows(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.12f", lineterminator="\n").encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_manifest(
    universe: pd.DataFrame,
    wrte: pd.DataFrame,
    rb_target: pd.DataFrame,
    rb_carry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    u = universe.copy()
    u.columns = [str(c).strip().lower() for c in u.columns]
    required = {"player", "team", "opponent", "position"}
    missing = required - set(u.columns)
    if missing:
        raise RuntimeError(f"Week-5 universe missing columns: {sorted(missing)}")

    u["season"] = SEASON
    u["week"] = WEEK
    u["team"] = u["team"].map(canon_team)
    u["opponent"] = u["opponent"].map(canon_team)
    u["position_raw"] = u["position"].astype(str).str.upper().str.strip()
    u = u.loc[u["position_raw"].isin(SUPPORTED)].copy()
    u["position_family"] = u["position_raw"].map(_pos)
    u["player_clean_key"] = u["player"].map(_clean_key)
    u["player_base_key"] = u["player"].map(_base_key)
    u["event_id"] = [_event(t, o) for t, o in zip(u["team"], u["opponent"])]
    if u[["team", "opponent", "player_clean_key", "event_id"]].eq("").any().any():
        raise RuntimeError("Week-5 universe contains unresolved identity")
    if u.duplicated(["team", "player_clean_key"]).any():
        bad = u.loc[u.duplicated(["team", "player_clean_key"], keep=False), ["team", "player"]]
        raise RuntimeError(f"duplicate Week-5 universe player identity: {bad.head(20).to_dict('records')}")

    target_cols = [
        "season", "week", "event_id", "team", "player_clean_key",
        "baseline_entitlement_tgt_share", "shadow_entitlement_tgt_share",
        "entitlement_delta", "trajectory_available", "trajectory_route",
        "trajectory_delta",
    ]

    for label, x, families in (
        ("WRTE", wrte, {"WR", "TE"}),
        ("RB_TARGET", rb_target, {"RB"}),
    ):
        miss = set(target_cols) - set(x.columns)
        if miss:
            raise RuntimeError(f"{label} target lock missing columns: {sorted(miss)}")
        if x.duplicated(["season", "week", "event_id", "team", "player_clean_key"]).any():
            raise RuntimeError(f"{label} target lock duplicate identity")
        fam = x["position_family"].astype(str).str.upper()
        if not set(fam.unique()).issubset(families):
            raise RuntimeError(f"{label} target lock family contamination: {sorted(set(fam.unique()) - families)}")
        for c in ("baseline_entitlement_tgt_share", "shadow_entitlement_tgt_share", "entitlement_delta"):
            vals = pd.to_numeric(x[c], errors="coerce")
            if vals.isna().any():
                raise RuntimeError(f"{label} has non-finite {c}")
        if (pd.to_numeric(x["baseline_entitlement_tgt_share"]) < 0).any() or (
            pd.to_numeric(x["shadow_entitlement_tgt_share"]) < 0
        ).any():
            raise RuntimeError(f"{label} contains negative target entitlement")

    wrte2 = wrte[target_cols].copy().rename(columns={
        "baseline_entitlement_tgt_share": "target_baseline_share",
        "shadow_entitlement_tgt_share": "target_shadow_share",
        "entitlement_delta": "target_shadow_delta",
        "trajectory_available": "target_shadow_feature_available",
        "trajectory_route": "target_shadow_route",
        "trajectory_delta": "target_trajectory_delta",
    })
    wrte2["target_shadow_family"] = "WRTE_TARGET_SHARE_TRAJECTORY_V1"
    wrte2["target_shadow_source_run"] = WRTE_RUN
    wrte2["target_shadow_source_digest"] = WRTE_DIGEST

    rb2 = rb_target[target_cols].copy().rename(columns={
        "baseline_entitlement_tgt_share": "target_baseline_share",
        "shadow_entitlement_tgt_share": "target_shadow_share",
        "entitlement_delta": "target_shadow_delta",
        "trajectory_available": "target_shadow_feature_available",
        "trajectory_route": "target_shadow_route",
        "trajectory_delta": "target_trajectory_delta",
    })
    rb2["target_shadow_family"] = "RB_TARGET_SHARE_TRAJECTORY_SHADOW_V1"
    rb2["target_shadow_source_run"] = RB_TARGET_RUN
    rb2["target_shadow_source_digest"] = RB_TARGET_DIGEST

    target = pd.concat([wrte2, rb2], ignore_index=True, sort=False)
    if target.duplicated(["season", "week", "event_id", "team", "player_clean_key"]).any():
        raise RuntimeError("target shadow locks overlap the same player identity")

    target_keys = ["season", "week", "event_id", "team", "player_clean_key"]
    universe_keys = set(map(tuple, u[target_keys].astype(str).to_numpy()))
    locked_target_keys = set(map(tuple, target[target_keys].astype(str).to_numpy()))
    missing_target = locked_target_keys - universe_keys
    if missing_target:
        raise RuntimeError(f"target lock rows missing from Week-5 universe: {sorted(missing_target)[:20]}")

    out = u.merge(target, on=target_keys, how="left", validate="one_to_one")
    out["target_shadow_available"] = out["target_shadow_family"].notna()
    out["target_shadow_feature_available"] = out["target_shadow_feature_available"].fillna(False).astype(bool)
    out["target_shadow_family"] = out["target_shadow_family"].fillna("")
    out["target_shadow_route"] = out["target_shadow_route"].fillna("")
    out["target_shadow_source_digest"] = out["target_shadow_source_digest"].fillna("")
    out["target_shadow_source_run"] = pd.to_numeric(out["target_shadow_source_run"], errors="coerce")

    carry_required = {
        "season", "week", "team", "player",
        "control_recent_carry_share", "shadow_player_state_share",
    }
    miss = carry_required - set(rb_carry.columns)
    if miss:
        raise RuntimeError(f"RB carry lock missing columns: {sorted(miss)}")
    c = rb_carry.copy()
    c["team"] = c["team"].map(canon_team)
    c["player_base_key"] = c["player"].map(_base_key)
    if c.duplicated(["season", "week", "team", "player_base_key"]).any():
        raise RuntimeError("RB carry lock duplicate canonical identity")
    for col in ("control_recent_carry_share", "shadow_player_state_share"):
        vals = pd.to_numeric(c[col], errors="coerce")
        if vals.isna().any() or (~vals.between(0, 1)).any():
            raise RuntimeError(f"RB carry lock invalid {col}")
    c["rb_carry_shadow_delta"] = (
        pd.to_numeric(c["shadow_player_state_share"])
        - pd.to_numeric(c["control_recent_carry_share"])
    )
    c2 = c[
        [
            "season", "week", "team", "player_base_key",
            "control_recent_carry_share", "shadow_player_state_share",
            "rb_carry_shadow_delta",
        ]
    ].rename(columns={
        "control_recent_carry_share": "rb_carry_control_share",
        "shadow_player_state_share": "rb_carry_shadow_share",
    })

    carry_universe = u.loc[u["position_family"].isin({"RB", "FB"})].copy()
    carry_join_cols = ["season", "week", "team", "player_base_key"]
    if carry_universe.duplicated(carry_join_cols).any():
        bad = carry_universe.loc[
            carry_universe.duplicated(carry_join_cols, keep=False),
            carry_join_cols + ["player"],
        ]
        raise RuntimeError(
            f"ambiguous public-universe RB carry identity: {bad.head(20).to_dict('records')}"
        )
    carry_keys = set(map(tuple, carry_universe[carry_join_cols].astype(str).to_numpy()))
    locked_carry_keys = set(map(tuple, c2[carry_join_cols].astype(str).to_numpy()))
    missing_carry = locked_carry_keys - carry_keys

    # The RB rushing shadow was frozen from a separately captured target-roster
    # authority. Its contract permits a lock identity to have zero matches in
    # another roster authority; it forbids adding or force-matching that player
    # after the fact. Preserve such source-universe differences explicitly.
    out = out.merge(
        c2,
        on=carry_join_cols,
        how="left",
        validate="one_to_one",
    )
    out["rb_carry_shadow_available"] = out["rb_carry_shadow_share"].notna()
    out["rb_carry_shadow_source_run"] = np.where(
        out["rb_carry_shadow_available"], RB_CARRY_RUN, np.nan
    )
    out["rb_carry_shadow_source_digest"] = np.where(
        out["rb_carry_shadow_available"], RB_CARRY_DIGEST, ""
    )

    # Position-family conflict gates.
    qb = out["position_family"].eq("QB")
    if out.loc[qb, "target_shadow_available"].any() or out.loc[qb, "rb_carry_shadow_available"].any():
        raise RuntimeError("QB incorrectly received a player shadow")
    wrte_mask = out["position_family"].isin({"WR", "TE"})
    if out.loc[wrte_mask, "rb_carry_shadow_available"].any():
        raise RuntimeError("WR/TE incorrectly received RB carry shadow")
    non_rb_target = out["target_shadow_family"].eq("RB_TARGET_SHARE_TRAJECTORY_SHADOW_V1") & ~out["position_family"].eq("RB")
    if non_rb_target.any():
        raise RuntimeError("RB target shadow mapped to non-RB")
    wrte_target = out["target_shadow_family"].eq("WRTE_TARGET_SHARE_TRAJECTORY_V1") & ~out["position_family"].isin({"WR", "TE"})
    if wrte_target.any():
        raise RuntimeError("WR/TE target shadow mapped to wrong family")

    # Carry lock conservation is certified on the immutable source cohort,
    # not on the subset that happens to overlap this public roster universe.
    carry_source_team = c2.groupby("team").agg(
        control_sum=("rb_carry_control_share", "sum"),
        shadow_sum=("rb_carry_shadow_share", "sum"),
        locked_players=("player_base_key", "size"),
    ).reset_index()
    carry_source_team["control_gap"] = (carry_source_team["control_sum"] - 1.0).abs()
    carry_source_team["shadow_gap"] = (carry_source_team["shadow_sum"] - 1.0).abs()
    if len(carry_source_team) and (
        float(carry_source_team["control_gap"].max()) > 1e-12
        or float(carry_source_team["shadow_gap"].max()) > 1e-12
    ):
        raise RuntimeError("RB carry source lock lost room conservation")

    carry_rows = out.loc[out["rb_carry_shadow_available"]].copy()
    mapped_carry_keys = set(
        map(
            tuple,
            carry_rows[["season", "week", "team", "player_base_key"]]
            .astype(str)
            .to_numpy(),
        )
    )
    if len(mapped_carry_keys) != len(carry_rows):
        raise RuntimeError("RB carry source mapped more than once into public universe")

    out["player_state_route"] = "BASELINE_NO_NEW_PLAYER_SHADOW"
    out.loc[qb, "player_state_route"] = "PROTECTED_PRODUCTION_QB_NO_NEW_PLAYER_SHADOW"
    out.loc[out["target_shadow_available"], "player_state_route"] = "TARGET_SHARE_SHADOW"
    both_rb = out["target_shadow_available"] & out["rb_carry_shadow_available"]
    out.loc[both_rb, "player_state_route"] = "RB_RUSH_AND_TARGET_SHARE_SHADOWS"
    carry_only = out["rb_carry_shadow_available"] & ~out["target_shadow_available"]
    out.loc[carry_only, "player_state_route"] = "RB_RUSH_SHADOW_ONLY"
    out["production_changed"] = False
    out["parameters_fit"] = 0
    out["sportsbook_inputs_used"] = 0
    out["week5_outcomes_read"] = 0

    conflict_rows = [
        {
            "check": "wrte_target_lock_all_rows_mapped",
            "passed": len(missing_target) == 0,
            "value": int(len(wrte)),
            "detail": WRTE_DIGEST,
        },
        {
            "check": "rb_target_lock_all_rows_mapped",
            "passed": len(missing_target) == 0,
            "value": int(len(rb_target)),
            "detail": RB_TARGET_DIGEST,
        },
        {
            "check": "rb_carry_source_lock_conserved",
            "passed": (
                float(carry_source_team["control_gap"].max()) <= 1e-12
                and float(carry_source_team["shadow_gap"].max()) <= 1e-12
            ) if len(carry_source_team) else False,
            "value": int(len(rb_carry)),
            "detail": RB_CARRY_DIGEST,
        },
        {
            "check": "rb_carry_unmapped_source_identities_preserved",
            "passed": True,
            "value": int(len(missing_carry)),
            "detail": ";".join(
                f"{key[2]}:{key[3]}" for key in sorted(missing_carry)
            ),
        },
        {
            "check": "target_lock_identity_overlap",
            "passed": True,
            "value": 0,
            "detail": "WR/TE and RB target families are disjoint",
        },
        {
            "check": "qb_shadow_contamination",
            "passed": True,
            "value": 0,
            "detail": "QB remains protected production baseline",
        },
        {
            "check": "rb_carry_max_control_gap",
            "passed": float(carry_source_team["control_gap"].max()) <= 1e-12 if len(carry_source_team) else False,
            "value": float(carry_source_team["control_gap"].max()) if len(carry_source_team) else np.nan,
            "detail": "immutable source locked cohort",
        },
        {
            "check": "rb_carry_max_shadow_gap",
            "passed": float(carry_source_team["shadow_gap"].max()) <= 1e-12 if len(carry_source_team) else False,
            "value": float(carry_source_team["shadow_gap"].max()) if len(carry_source_team) else np.nan,
            "detail": "immutable source locked cohort",
        },
    ]
    conflict = pd.DataFrame(conflict_rows)
    if not conflict["passed"].all():
        raise RuntimeError(f"manifest conflict audit failed: {conflict.loc[~conflict['passed']].to_dict('records')}")

    cols = [
        "season", "week", "event_id", "team", "opponent", "player",
        "player_clean_key", "player_base_key", "position_raw", "position_family",
        "player_state_route",
        "target_shadow_available", "target_shadow_feature_available",
        "target_shadow_family", "target_shadow_route",
        "target_baseline_share", "target_shadow_share", "target_shadow_delta",
        "target_trajectory_delta", "target_shadow_source_run",
        "target_shadow_source_digest",
        "rb_carry_shadow_available", "rb_carry_control_share",
        "rb_carry_shadow_share", "rb_carry_shadow_delta",
        "rb_carry_shadow_source_run", "rb_carry_shadow_source_digest",
        "parameters_fit", "sportsbook_inputs_used", "week5_outcomes_read",
        "production_changed",
    ]
    out = out[cols].sort_values(
        ["event_id", "team", "position_family", "player_clean_key"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = {
        "version": "PLAYER_CENTRIC_WEEK5_SHADOW_MANIFEST_V1",
        "status": "WEEK5_PLAYER_CENTRIC_MANIFEST_FROZEN",
        "season": SEASON,
        "week": WEEK,
        "players": int(len(out)),
        "player_weeks": int(len(out)),
        "position_counts": out.groupby("position_family").size().astype(int).to_dict(),
        "target_shadow_rows": int(out["target_shadow_available"].sum()),
        "target_feature_available_rows": int(out["target_shadow_feature_available"].sum()),
        "rb_carry_source_lock_rows": int(len(rb_carry)),
        "rb_carry_shadow_rows": int(out["rb_carry_shadow_available"].sum()),
        "rb_carry_unmapped_source_rows": int(len(missing_carry)),
        "rb_carry_unmapped_source_identities": [
            {"team": key[2], "player_base_key": key[3]}
            for key in sorted(missing_carry)
        ],
        "rb_rows_with_both_shadows": int(both_rb.sum()),
        "qb_shadow_rows": int(
            (
                out["position_family"].eq("QB")
                & (out["target_shadow_available"] | out["rb_carry_shadow_available"])
            ).sum()
        ),
        "wrte_target_source_run": WRTE_RUN,
        "wrte_target_row_digest": WRTE_DIGEST,
        "rb_target_source_run": RB_TARGET_RUN,
        "rb_target_row_digest": RB_TARGET_DIGEST,
        "rb_carry_source_run": RB_CARRY_RUN,
        "rb_carry_row_digest": RB_CARRY_DIGEST,
        "manifest_row_digest": _digest_rows(out),
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "week5_outcomes_read": 0,
        "production_changed": False,
        "target_depth_distribution_included": False,
        "automatic_promotion": False,
    }
    return out, conflict, summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=Path, required=True)
    p.add_argument("--wrte-target-lock", type=Path, required=True)
    p.add_argument("--rb-target-lock", type=Path, required=True)
    p.add_argument("--rb-carry-lock", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    out, conflict, summary = build_manifest(
        _read(a.universe, "Week-5 pregame universe"),
        _read(a.wrte_target_lock, "WR/TE target lock"),
        _read(a.rb_target_lock, "RB target lock"),
        _read(a.rb_carry_lock, "RB carry lock"),
    )
    out.to_csv(a.out_dir / "player_centric_week5_shadow_manifest.csv", index=False, float_format="%.12f")
    conflict.to_csv(a.out_dir / "player_centric_week5_shadow_conflict_audit.csv", index=False)
    (a.out_dir / "player_centric_week5_shadow_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
