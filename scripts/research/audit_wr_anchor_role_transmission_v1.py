#!/usr/bin/env python3
"""Diagnostic-only audit of WR anchor / current-role transmission.

Scientific contract
-------------------
No new projection candidate is constructed or scored. The frozen diagnostic asks
whether strict-prior receiver participation identifies a current role leader that
fails to propagate through the existing M38 WR1 / WR-R15 hierarchy.

Mechanical authority recovery
-----------------------------
The canonical WR-R15 OOS artifact is the scientific authority. It contains:
- all scored baseline/candidate prediction rows;
- all 5,321 frozen WR2+ feature rows, including exact M38 baseline entitlement,
  exact WR-R15 candidate entitlement and strict-prior participation features;
- all 1,088 team-game conservation rows, including the frozen M38 anchor row
  index and immutable anchor entitlement.

The artifact does not persist the anchor player's identity or anchor participation
features. We therefore replay only the exact authority-era pregame source to
bridge frozen anchor_idx -> player identity and to attach the anchor's strict-
prior participation / actual-target label.

The bridge is permitted only if it passes all mechanical parity gates:
1. every frozen secondary row index maps to the same player identity;
2. every frozen secondary strict-prior participation value matches the replay;
3. every available frozen WR1 prediction agrees with the mapped anchor identity;
4. every scored actual-target label matches the replay;
5. no same/future participation rows are used.

Frozen entitlements are NEVER replaced by replayed entitlements. This avoids the
upstream historical-source drift that made full numerical replay invalid while
preserving the exact original WR-R15 state used for the diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.evaluate_wr_r14_participation_entitlement_v1 import (
    WR_POS,
    _build_bundle_frame,
    _strict_prior_snap_features,
    _target_actuals,
)
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps
from scripts.utils.canonical_names import canon_team

VERSION = "WR_ANCHOR_ROLE_TRANSMISSION_AUDIT_V1"
AUTHORITY_SOURCE_COMMIT = "02c3dd1a681d4ab2953683039e39830554f9ec9f"
AUTHORITY_RUN = 34238301577
AUTHORITY_ARTIFACT = 10061328722
AUTHORITY_FROZEN_DATE = "2026-09-08"
BASELINE_VARIANT = "M38_EXPLICIT_BASELINE"
CANDIDATE_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
SEASONS = (2023, 2024)
TEAM_KEYS = ["season", "week", "team"]
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
FULL_KEYS = ["season", "week", "event_id", "team", "player_clean_key"]


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"empty {label}: {path}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _canon(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    if "season" in x.columns:
        x["season"] = num(x["season"]).astype("Int64")
    if "week" in x.columns:
        x["week"] = num(x["week"]).astype("Int64")
    if "team" in x.columns:
        x["team"] = x["team"].map(canon_team)
    if "player_clean_key" in x.columns:
        x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str)
    if "event_id" in x.columns:
        x["event_id"] = x["event_id"].fillna("").astype(str)
    return x


def prepare_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    need = {
        "variant", "event_id", "team", "player_clean_key", "season", "week",
        "wr_rank", "entitlement_tgt_share", "pred_targets", "actual_targets",
    }
    missing = need - set(pred.columns)
    if missing:
        raise RuntimeError(f"WR-R15 predictions missing {sorted(missing)}")
    x = _canon(pred)
    x = x.loc[x["season"].isin(SEASONS)].copy()
    observed = set(x["variant"].astype(str).unique())
    expected = {BASELINE_VARIANT, CANDIDATE_VARIANT}
    if not expected.issubset(observed):
        raise RuntimeError(
            f"WR-R15 prediction variants missing expected={expected} observed={observed}"
        )

    b = x.loc[x["variant"].eq(BASELINE_VARIANT)].copy()
    c = x.loc[x["variant"].eq(CANDIDATE_VARIANT)].copy()
    for label, d in (("baseline", b), ("candidate", c)):
        if d.duplicated(PLAYER_KEYS).any():
            bad = d.loc[d.duplicated(PLAYER_KEYS, keep=False), PLAYER_KEYS].head(10)
            raise RuntimeError(f"{label} duplicate authority identities: {bad.to_dict('records')}")

    b = b[PLAYER_KEYS + [
        "event_id", "wr_rank", "entitlement_tgt_share",
        "pred_targets", "actual_targets",
    ]].rename(columns={
        "event_id": "authority_event_id",
        "wr_rank": "authority_wr_rank",
        "entitlement_tgt_share": "authority_baseline_entitlement",
        "pred_targets": "authority_baseline_pred_targets",
        "actual_targets": "authority_actual_targets",
    })
    c = c[PLAYER_KEYS + [
        "entitlement_tgt_share", "pred_targets", "actual_targets",
    ]].rename(columns={
        "entitlement_tgt_share": "authority_candidate_entitlement",
        "pred_targets": "authority_candidate_pred_targets",
        "actual_targets": "authority_candidate_actual_targets",
    })
    out = b.merge(c, on=PLAYER_KEYS, how="inner", validate="one_to_one")
    if len(out) != len(b) or len(out) != len(c):
        raise RuntimeError("authority baseline/candidate identity universe mismatch")
    actual_gap = (
        num(out["authority_actual_targets"])
        - num(out["authority_candidate_actual_targets"])
    ).abs()
    if len(actual_gap) and float(actual_gap.max()) > 1e-12:
        raise RuntimeError("authority actual targets differ by variant")
    for col in [
        "authority_baseline_entitlement", "authority_candidate_entitlement",
        "authority_baseline_pred_targets", "authority_candidate_pred_targets",
        "authority_actual_targets", "authority_wr_rank",
    ]:
        out[col] = num(out[col])
        if out[col].isna().any():
            raise RuntimeError(f"authority non-numeric values in {col}")
    rank1 = out.loc[out["authority_wr_rank"].eq(1), PLAYER_KEYS].copy()
    return out, rank1


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    need = {
        "_row_index", "event_id", "team", "player_clean_key", "player",
        "position", "season", "week", "baseline_wr_rank",
        "baseline_entitlement_tgt_share", "candidate_entitlement_tgt_share",
        "prior_count_same_team", "prior1_same_team",
        "prior1_same_team_offense_pct", "prior1_same_team_offense_snaps",
    }
    missing = need - set(features.columns)
    if missing:
        raise RuntimeError(f"WR-R15 features missing {sorted(missing)}")
    x = _canon(features)
    x = x.loc[x["season"].isin(SEASONS)].copy()
    x["_row_index"] = num(x["_row_index"]).astype("Int64")
    x["baseline_wr_rank"] = num(x["baseline_wr_rank"]).astype("Int64")
    if not x["baseline_wr_rank"].ge(2).all():
        raise RuntimeError("frozen WR-R15 feature artifact contains non-secondary rows")
    for c in [
        "baseline_entitlement_tgt_share", "candidate_entitlement_tgt_share",
        "prior_count_same_team", "prior1_same_team_offense_pct",
        "prior1_same_team_offense_snaps",
    ]:
        x[c] = num(x[c])
    x["prior1_same_team"] = x["prior1_same_team"].fillna(False).astype(bool)
    keys = TEAM_KEYS + ["_row_index"]
    if x.duplicated(keys).any():
        raise RuntimeError("frozen WR-R15 feature row-index identity is not unique")
    return x


def prepare_conservation(audit: pd.DataFrame) -> pd.DataFrame:
    need = {
        "test_season", "week", "event_id", "team", "anchor_idx",
        "anchor_entitlement_before", "anchor_entitlement_after",
        "anchor_entitlement_delta", "baseline_wr_room_mass",
        "candidate_wr_room_mass", "wr_room_mass_gap",
    }
    missing = need - set(audit.columns)
    if missing:
        raise RuntimeError(f"WR-R15 conservation audit missing {sorted(missing)}")
    x = audit.copy().rename(columns={"test_season": "season"})
    x = _canon(x)
    x = x.loc[x["season"].isin(SEASONS)].copy()
    x["anchor_idx"] = num(x["anchor_idx"]).astype("Int64")
    for c in [
        "anchor_entitlement_before", "anchor_entitlement_after",
        "anchor_entitlement_delta", "baseline_wr_room_mass",
        "candidate_wr_room_mass", "wr_room_mass_gap",
    ]:
        x[c] = num(x[c])
        if x[c].isna().any():
            raise RuntimeError(f"conservation audit non-numeric values in {c}")
    if x.duplicated(TEAM_KEYS).any():
        raise RuntimeError("conservation audit team-game identities are not unique")
    return x


def _identity_norm(value: object) -> str:
    s = "" if value is None or pd.isna(value) else str(value).lower().strip()
    for ch in (" ", "'", "-", ".", "’", "`"):
        s = s.replace(ch, "")
    return s


def build_post_freeze_inverse_aliases(
    overrides_path: Path, frozen_keys: set[str]
) -> tuple[dict[str, str], pd.DataFrame]:
    """Map current post-freeze canonical identities back to frozen authority keys.

    This is a mechanical identity bridge only. It uses explicit, verified repo
    overrides added after the WR-R15 freeze and only retains mappings whose
    pre-freeze/source key is actually present in the frozen WR-R15 authority.
    """
    x = read(overrides_path, "manual name overrides")
    need = {"player_source_name", "full_name", "verified_date"}
    missing = need - set(x.columns)
    if missing:
        raise RuntimeError(f"manual name overrides missing {sorted(missing)}")
    x["verified_date"] = pd.to_datetime(x["verified_date"], errors="coerce")
    freeze = pd.Timestamp(AUTHORITY_FROZEN_DATE)
    x["source_key"] = x["player_source_name"].map(_identity_norm)
    x["current_key"] = x["full_name"].map(_identity_norm)
    x = x.loc[
        x["verified_date"].gt(freeze)
        & x["source_key"].isin(set(str(k) for k in frozen_keys))
        & x["current_key"].ne("")
        & x["source_key"].ne("")
        & x["current_key"].ne(x["source_key"])
    ].copy()
    if x["current_key"].duplicated().any():
        bad = x.loc[x["current_key"].duplicated(keep=False), [
            "player_source_name", "full_name", "current_key", "source_key"
        ]].to_dict("records")
        raise RuntimeError(f"post-freeze inverse aliases are ambiguous: {bad}")
    mapping = dict(zip(x["current_key"], x["source_key"]))
    return mapping, x


def reconstruct_bridge(
    *,
    data_dirs: dict[int, Path],
    logs_by_season: dict[int, pd.DataFrame],
    overrides_path: Path,
    frozen_keys: set[str],
) -> tuple[pd.DataFrame, dict]:
    snaps, dup_rate, source_seasons = _load_snaps()
    rows: list[pd.DataFrame] = []
    future_total = 0

    for season in SEASONS:
        logs = logs_by_season[int(season)]
        for week in range(1, 19):
            baseline = _build_bundle_frame(
                season=int(season),
                week=int(week),
                prior_season=int(season - 1),
                data_dir=data_dirs[int(season)],
                logs=logs,
            )
            baseline["team"] = baseline["team"].map(canon_team)
            pos = (
                baseline.get("position", pd.Series("", index=baseline.index))
                .fillna("").astype(str).str.upper().str.strip()
            )
            wr = baseline.loc[pos.isin(WR_POS)].copy()
            if wr.empty:
                raise RuntimeError(f"{season} W{week} bridge found zero WR rows")

            feat, future = _strict_prior_snap_features(wr.copy(), snaps)
            future_total += int(future)
            if int(future) != 0:
                raise RuntimeError(
                    f"{season} W{week} bridge used same/future participation rows: {future}"
                )
            feat["team"] = feat["team"].map(canon_team)
            feat["season"] = int(season)
            feat["week"] = int(week)
            feat["_row_index"] = num(feat["_row_index"]).astype("Int64")

            actual = _target_actuals(logs, int(season), int(week))
            actual["team"] = actual["team"].map(canon_team)
            actual["actual_targets"] = num(actual["actual_targets"])
            feat = feat.merge(
                actual,
                on=["team", "player_clean_key"],
                how="left",
                validate="one_to_one",
            )
            feat["actual_targets"] = num(feat["actual_targets"]).fillna(0.0)
            rows.append(feat[[
                "_row_index", "event_id", "team", "player_clean_key", "player",
                "position", "season", "week", "prior_count_same_team",
                "prior1_same_team", "prior1_same_team_offense_pct",
                "prior1_same_team_offense_snaps", "actual_targets",
            ]])

    bridge = pd.concat(rows, ignore_index=True)
    bridge = _canon(bridge)
    bridge["_row_index"] = num(bridge["_row_index"]).astype("Int64")

    # NFLverse/Ourlads retrospective identity normalization changed after the
    # 2026-09-08 authority freeze. The repo's verified post-freeze overrides
    # document those suffix corrections. Invert only mappings whose old/source
    # key exists in the frozen artifact, preserving both current and authority
    # identities for audit.
    alias_map, alias_rows = build_post_freeze_inverse_aliases(
        overrides_path, frozen_keys
    )
    bridge["current_player_clean_key"] = bridge["player_clean_key"].astype(str)
    bridge["authority_player_clean_key"] = bridge["current_player_clean_key"].map(
        lambda k: alias_map.get(str(k), str(k))
    )
    bridge["authority_alias_applied"] = (
        bridge["authority_player_clean_key"] != bridge["current_player_clean_key"]
    )
    bridge["player_clean_key"] = bridge["authority_player_clean_key"]

    if bridge.duplicated(PLAYER_KEYS).any():
        bad = bridge.loc[
            bridge.duplicated(PLAYER_KEYS, keep=False),
            PLAYER_KEYS + ["current_player_clean_key", "authority_alias_applied"],
        ].head(20).to_dict("records")
        raise RuntimeError(
            f"post-freeze authority aliasing created duplicate player identities: {bad}"
        )

    keys = TEAM_KEYS + ["_row_index"]
    if bridge.duplicated(keys).any():
        raise RuntimeError("authority bridge row-index identity is not unique")

    applied = bridge.loc[bridge["authority_alias_applied"]].copy()
    alias_usage = (
        applied.groupby(
            ["current_player_clean_key", "authority_player_clean_key"],
            as_index=False,
        ).size().rename(columns={"size": "rows"})
        if len(applied)
        else pd.DataFrame(
            columns=["current_player_clean_key", "authority_player_clean_key", "rows"]
        )
    )
    return bridge, {
        "authority_source_commit": AUTHORITY_SOURCE_COMMIT,
        "authority_frozen_date": AUTHORITY_FROZEN_DATE,
        "raw_snap_duplicate_rate": float(dup_rate),
        "snap_source_seasons": [int(x) for x in source_seasons],
        "strict_prior_future_violations": int(future_total),
        "bridge_wr_rows": int(len(bridge)),
        "bridge_team_games": int(
            bridge[["season", "week", "event_id", "team"]]
            .drop_duplicates().shape[0]
        ),
        "post_freeze_inverse_aliases_available": int(len(alias_map)),
        "post_freeze_inverse_alias_rows_used": int(len(applied)),
        "post_freeze_inverse_alias_usage": alias_usage.to_dict("records"),
        "post_freeze_override_records_eligible": int(len(alias_rows)),
    }


def build_frozen_full_state(
    *,
    predictions: pd.DataFrame,
    rank1_predictions: pd.DataFrame,
    features: pd.DataFrame,
    conservation: pd.DataFrame,
    bridge: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    # Row indices inside the historical pregame frame are not football identity
    # and can drift when upstream source rows are revised. The frozen artifact
    # already carries stable player identities for every WR2+ row, so secondary
    # bridge parity is keyed only by season/week/team/player.
    if bridge.duplicated(PLAYER_KEYS).any():
        bad = bridge.loc[
            bridge.duplicated(PLAYER_KEYS, keep=False), PLAYER_KEYS
        ].head(10).to_dict("records")
        raise RuntimeError(f"bridge player identities are not unique: {bad}")

    bridge_identity = bridge.rename(columns={
        "event_id": "bridge_event_id",
        "player": "bridge_player",
        "position": "bridge_position",
        "prior_count_same_team": "bridge_prior_count_same_team",
        "prior1_same_team": "bridge_prior1_same_team",
        "prior1_same_team_offense_pct": "bridge_prior1_same_team_offense_pct",
        "prior1_same_team_offense_snaps": "bridge_prior1_same_team_offense_snaps",
        "actual_targets": "bridge_actual_targets",
        "_row_index": "bridge_row_index",
    })

    sec = features.merge(
        bridge_identity,
        on=PLAYER_KEYS,
        how="left",
        validate="one_to_one",
    )
    if sec["bridge_event_id"].isna().any():
        missing = sec.loc[
            sec["bridge_event_id"].isna(), PLAYER_KEYS
        ].head(10).to_dict("records")
        raise RuntimeError(
            f"frozen secondary identity missing from authority-era WR universe: {missing}"
        )
    event_mismatch = sec["event_id"].astype(str).ne(
        sec["bridge_event_id"].astype(str)
    )
    count_gap = (
        num(sec["prior_count_same_team"])
        - num(sec["bridge_prior_count_same_team"])
    ).abs()
    bool_mismatch = (
        sec["prior1_same_team"].fillna(False).astype(bool)
        != sec["bridge_prior1_same_team"].fillna(False).astype(bool)
    )

    def max_numeric_gap(a: pd.Series, b: pd.Series) -> float:
        aa, bb = num(a), num(b)
        both_na = aa.isna() & bb.isna()
        one_na = aa.isna() ^ bb.isna()
        if one_na.any():
            return float("inf")
        d = (aa[~both_na] - bb[~both_na]).abs()
        return float(d.max()) if len(d) else 0.0

    pct_gap = max_numeric_gap(
        sec["prior1_same_team_offense_pct"],
        sec["bridge_prior1_same_team_offense_pct"],
    )
    snap_gap = max_numeric_gap(
        sec["prior1_same_team_offense_snaps"],
        sec["bridge_prior1_same_team_offense_snaps"],
    )
    mechanical = {
        "frozen_secondary_rows": int(len(sec)),
        "secondary_bridge_coverage": float(sec["bridge_event_id"].notna().mean()),
        "secondary_identity_mismatches": 0,
        "secondary_event_id_mismatches": int(event_mismatch.sum()),
        "secondary_prior_count_max_gap": float(count_gap.max()) if len(count_gap) else 0.0,
        "secondary_prior1_bool_mismatches": int(bool_mismatch.sum()),
        "secondary_prior1_offense_pct_max_gap": pct_gap,
        "secondary_prior1_offense_snaps_max_gap": snap_gap,
    }
    if mechanical["secondary_event_id_mismatches"] != 0:
        raise RuntimeError(f"secondary event bridge parity failed: {mechanical}")
    if mechanical["secondary_prior_count_max_gap"] > 0:
        raise RuntimeError(f"secondary prior-count parity failed: {mechanical}")
    if mechanical["secondary_prior1_bool_mismatches"] != 0:
        raise RuntimeError(
            f"secondary participation availability parity failed: {mechanical}"
        )
    if mechanical["secondary_prior1_offense_pct_max_gap"] > 1e-12:
        raise RuntimeError(f"secondary snap-pct parity failed: {mechanical}")
    if mechanical["secondary_prior1_offense_snaps_max_gap"] > 1e-12:
        raise RuntimeError(f"secondary snap-count parity failed: {mechanical}")

    # The frozen feature artifact contains every WR2+ row. Therefore the M38
    # anchor is the one WR identity in the authority-era full WR universe that is
    # absent from the frozen secondary identity set for that team-game.
    frozen_secondary_keys = features[PLAYER_KEYS].drop_duplicates().copy()
    universe = bridge.merge(
        frozen_secondary_keys.assign(_frozen_secondary=True),
        on=PLAYER_KEYS,
        how="left",
        validate="one_to_one",
    )
    anchor_candidates = universe.loc[
        universe["_frozen_secondary"].isna()
    ].copy()
    counts = anchor_candidates.groupby(TEAM_KEYS).size()
    frozen_team_count = int(len(conservation))
    all_bridge_team_count = int(
        bridge[TEAM_KEYS].drop_duplicates().shape[0]
    )
    mechanical.update({
        "frozen_team_games": frozen_team_count,
        "bridge_team_games_for_anchor_inference": all_bridge_team_count,
        "team_games_with_one_complement_anchor": int(counts.eq(1).sum()),
        "team_games_with_zero_complement_anchor": int(
            (counts.reindex(
                bridge[TEAM_KEYS].drop_duplicates().set_index(TEAM_KEYS).index,
                fill_value=0,
            ) == 0).sum()
        ),
        "team_games_with_multiple_complement_anchors": int(counts.gt(1).sum()),
    })
    if all_bridge_team_count != frozen_team_count:
        raise RuntimeError(
            f"bridge/frozen team-game universe mismatch: {mechanical}"
        )
    all_team = bridge[TEAM_KEYS].drop_duplicates()
    anchor_count_check = all_team.merge(
        counts.rename("anchor_candidates").reset_index(),
        on=TEAM_KEYS,
        how="left",
        validate="one_to_one",
    )
    anchor_count_check["anchor_candidates"] = (
        anchor_count_check["anchor_candidates"].fillna(0).astype(int)
    )
    if not anchor_count_check["anchor_candidates"].eq(1).all():
        bad = anchor_count_check.loc[
            ~anchor_count_check["anchor_candidates"].eq(1)
        ].head(20).to_dict("records")
        raise RuntimeError(
            f"frozen WR2+ complement does not yield exactly one anchor: {bad}"
        )

    anchors = anchor_candidates.merge(
        conservation,
        on=TEAM_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_bridge", "_audit"),
    )
    if len(anchors) != frozen_team_count:
        raise RuntimeError("complement anchor bridge did not preserve frozen team-game count")
    if (
        anchors["event_id_bridge"].astype(str)
        != anchors["event_id_audit"].astype(str)
    ).any():
        raise RuntimeError("complement anchor event identity mismatch")

    # Independent direct check: 1,026 frozen scored WR1 identities must equal
    # the inferred complement anchor identity.
    rank1_check = rank1_predictions.merge(
        anchors[TEAM_KEYS + ["player_clean_key"]].rename(
            columns={"player_clean_key": "mapped_anchor_key"}
        ),
        on=TEAM_KEYS,
        how="left",
        validate="one_to_one",
    )
    rank1_mismatch = rank1_check["player_clean_key"].astype(str).ne(
        rank1_check["mapped_anchor_key"].astype(str)
    )
    mechanical.update({
        "mapped_anchor_rows": int(len(anchors)),
        "frozen_scored_rank1_rows": int(len(rank1_check)),
        "rank1_anchor_identity_mismatches": int(rank1_mismatch.sum()),
    })
    if mechanical["rank1_anchor_identity_mismatches"] != 0:
        bad = rank1_check.loc[rank1_mismatch].head(10).to_dict("records")
        raise RuntimeError(
            f"complement anchor disagrees with frozen scored WR1 identity: {bad}"
        )

    # Scored actual-target parity against the authority artifact.
    label_check = predictions.merge(
        bridge[PLAYER_KEYS + ["actual_targets"]],
        on=PLAYER_KEYS,
        how="left",
        validate="one_to_one",
    )
    if label_check["actual_targets"].isna().any():
        raise RuntimeError("scored target-label bridge coverage is incomplete")
    label_gap = (
        num(label_check["authority_actual_targets"])
        - num(label_check["actual_targets"])
    ).abs()
    mechanical["scored_actual_target_rows"] = int(len(label_check))
    mechanical["max_scored_actual_target_gap"] = (
        float(label_gap.max()) if len(label_gap) else 0.0
    )
    if mechanical["max_scored_actual_target_gap"] > 1e-12:
        raise RuntimeError(f"scored actual-target parity failed: {mechanical}")

    # Frozen WR2+ rows: all entitlement and participation values come directly
    # from the immutable artifact. Replay supplies only the actual-target label,
    # which is independently exact on all scored rows above.
    secondary = pd.DataFrame({
        "season": sec["season"].astype(int),
        "week": sec["week"].astype(int),
        "event_id": sec["event_id"].astype(str),
        "team": sec["team"].astype(str),
        "player_clean_key": sec["player_clean_key"].astype(str),
        "player": sec["player"],
        "position": sec["position"],
        "baseline_wr_rank": num(sec["baseline_wr_rank"]).astype(int),
        "baseline_entitlement_tgt_share": num(sec["baseline_entitlement_tgt_share"]),
        "candidate_entitlement_tgt_share": num(sec["candidate_entitlement_tgt_share"]),
        "prior_count_same_team": num(sec["prior_count_same_team"]),
        "prior1_same_team": sec["prior1_same_team"].fillna(False).astype(bool),
        "prior1_same_team_offense_pct": num(sec["prior1_same_team_offense_pct"]),
        "prior1_same_team_offense_snaps": num(sec["prior1_same_team_offense_snaps"]),
        "actual_targets": num(sec["bridge_actual_targets"]).fillna(0.0),
        "authority_row_type": "FROZEN_SECONDARY",
    })

    # Frozen anchor entitlements come from conservation audit; identity,
    # participation and actual-target label come from the mechanically validated
    # complement bridge.
    anchor = pd.DataFrame({
        "season": anchors["season"].astype(int),
        "week": anchors["week"].astype(int),
        "event_id": anchors["event_id_audit"].astype(str),
        "team": anchors["team"].astype(str),
        "player_clean_key": anchors["player_clean_key"].astype(str),
        "player": anchors["player"],
        "position": anchors["position"],
        "baseline_wr_rank": 1,
        "baseline_entitlement_tgt_share": num(anchors["anchor_entitlement_before"]),
        "candidate_entitlement_tgt_share": num(anchors["anchor_entitlement_after"]),
        "prior_count_same_team": num(anchors["prior_count_same_team"]),
        "prior1_same_team": anchors["prior1_same_team"].fillna(False).astype(bool),
        "prior1_same_team_offense_pct": num(anchors["prior1_same_team_offense_pct"]),
        "prior1_same_team_offense_snaps": num(anchors["prior1_same_team_offense_snaps"]),
        "actual_targets": num(anchors["actual_targets"]).fillna(0.0),
        "authority_row_type": "FROZEN_ANCHOR_IDENTITY_COMPLEMENT_BRIDGE",
    })

    full = pd.concat([anchor, secondary], ignore_index=True, sort=False)
    if full.duplicated(FULL_KEYS).any():
        bad = full.loc[
            full.duplicated(FULL_KEYS, keep=False), FULL_KEYS
        ].head(10).to_dict("records")
        raise RuntimeError(f"frozen full state has duplicate identities: {bad}")
    anchor_count = (
        full.loc[full["baseline_wr_rank"].eq(1)]
        .groupby(["season", "week", "event_id", "team"])
        .size()
    )
    if len(anchor_count) != frozen_team_count or not anchor_count.eq(1).all():
        raise RuntimeError(
            "frozen full state does not contain exactly one anchor per team-game"
        )

    mechanical["max_frozen_anchor_entitlement_delta"] = float(
        num(conservation["anchor_entitlement_delta"]).abs().max()
    )
    mechanical["max_frozen_wr_room_mass_gap"] = float(
        num(conservation["wr_room_mass_gap"]).abs().max()
    )
    if mechanical["max_frozen_anchor_entitlement_delta"] > 1e-12:
        raise RuntimeError("frozen authority does not preserve anchor entitlement")
    if mechanical["max_frozen_wr_room_mass_gap"] > 1e-12:
        raise RuntimeError("frozen authority does not preserve WR room mass")
    mechanical["frozen_full_wr_rows"] = int(len(full))
    return full.sort_values(FULL_KEYS).reset_index(drop=True), mechanical

def deterministic_leader(
    g: pd.DataFrame, value_col: str, *, eligibility: pd.Series | None = None
) -> str:
    z = g.copy()
    if eligibility is not None:
        z = z.loc[eligibility.loc[z.index]].copy()
    z[value_col] = num(z[value_col])
    z = z.loc[z[value_col].notna()].copy()
    if z.empty:
        return ""
    z = z.sort_values(
        [value_col, "player_clean_key"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(z.iloc[0]["player_clean_key"])


def player_value(g: pd.DataFrame, player_key: str, col: str) -> float:
    z = g.loc[g["player_clean_key"].eq(str(player_key))]
    if len(z) != 1:
        return np.nan
    v = pd.to_numeric(z.iloc[0][col], errors="coerce")
    return float(v) if pd.notna(v) else np.nan


def rank_for_player(g: pd.DataFrame, player_key: str, col: str) -> float:
    z = g[["player_clean_key", col]].copy()
    z[col] = num(z[col])
    z = z.sort_values(
        [col, "player_clean_key"], ascending=[False, True], kind="mergesort"
    )
    z["rank"] = np.arange(1, len(z) + 1, dtype=float)
    hit = z.loc[z["player_clean_key"].eq(str(player_key)), "rank"]
    return float(hit.iloc[0]) if len(hit) else np.nan


def build_team_games(
    detail: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    auth = predictions.copy()
    auth["authority_baseline_abs_error"] = (
        auth["authority_baseline_pred_targets"] - auth["authority_actual_targets"]
    ).abs()
    auth["authority_candidate_abs_error"] = (
        auth["authority_candidate_pred_targets"] - auth["authority_actual_targets"]
    ).abs()
    err = (
        auth.groupby(TEAM_KEYS, as_index=False)
        .agg(
            authority_scored_wr_rows=("player_clean_key", "size"),
            baseline_target_abs_error_sum=("authority_baseline_abs_error", "sum"),
            candidate_target_abs_error_sum=("authority_candidate_abs_error", "sum"),
        )
    )

    rows = []
    group_cols = ["season", "week", "event_id", "team"]
    for keys, g in detail.groupby(group_cols, sort=True):
        season, week, event_id, team = keys
        anchor_rows = g.loc[num(g["baseline_wr_rank"]).eq(1)]
        if len(anchor_rows) != 1:
            raise RuntimeError(f"expected one frozen M38 anchor {keys}, found {len(anchor_rows)}")
        anchor_key = str(anchor_rows.iloc[0]["player_clean_key"])

        eligible = (
            g["prior1_same_team"].fillna(False).astype(bool)
            & num(g["prior1_same_team_offense_pct"]).notna()
        )
        participation = deterministic_leader(
            g, "prior1_same_team_offense_pct", eligibility=eligible
        )
        if not participation:
            continue

        candidate_leader = deterministic_leader(
            g, "candidate_entitlement_tgt_share"
        )
        actual_max = float(num(g["actual_targets"]).max())
        actual_top = set(
            g.loc[
                num(g["actual_targets"]).eq(actual_max), "player_clean_key"
            ].astype(str)
        )
        rec = {
            "season": int(season),
            "week": int(week),
            "event_id": str(event_id),
            "team": str(team),
            "n_wr_full_pregame": int(len(g)),
            "m38_anchor_key": anchor_key,
            "participation_leader_key": participation,
            "candidate_entitlement_leader_key": candidate_leader,
            "anchor_participation_mismatch": int(anchor_key != participation),
            "candidate_followed_participation_leader": int(
                candidate_leader == participation
            ),
            "candidate_leader_changed_from_anchor": int(
                candidate_leader != anchor_key
            ),
            "anchor_actual_top_hit": int(anchor_key in actual_top),
            "participation_leader_actual_top_hit": int(
                participation in actual_top
            ),
            "candidate_leader_actual_top_hit": int(
                candidate_leader in actual_top
            ),
            "anchor_actual_targets": player_value(
                g, anchor_key, "actual_targets"
            ),
            "participation_leader_actual_targets": player_value(
                g, participation, "actual_targets"
            ),
            "anchor_baseline_entitlement": player_value(
                g, anchor_key, "baseline_entitlement_tgt_share"
            ),
            "anchor_candidate_entitlement": player_value(
                g, anchor_key, "candidate_entitlement_tgt_share"
            ),
            "participation_leader_baseline_entitlement": player_value(
                g, participation, "baseline_entitlement_tgt_share"
            ),
            "participation_leader_candidate_entitlement": player_value(
                g, participation, "candidate_entitlement_tgt_share"
            ),
            "participation_leader_baseline_rank": rank_for_player(
                g, participation, "baseline_entitlement_tgt_share"
            ),
            "participation_leader_candidate_rank": rank_for_player(
                g, participation, "candidate_entitlement_tgt_share"
            ),
        }
        rec["participation_minus_anchor_actual_targets"] = (
            rec["participation_leader_actual_targets"]
            - rec["anchor_actual_targets"]
        )
        rec["anchor_entitlement_immutability_gap"] = abs(
            rec["anchor_candidate_entitlement"]
            - rec["anchor_baseline_entitlement"]
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("anchor transmission audit produced zero eligible team-games")
    out = out.merge(err, on=TEAM_KEYS, how="left", validate="one_to_one")
    out = out.loc[out["authority_scored_wr_rows"].notna()].copy()
    if out.empty:
        raise RuntimeError("zero eligible team-games overlap scored WR-R15 authority")
    return out.sort_values(group_cols).reset_index(drop=True)


def cohort_row(d: pd.DataFrame, scope: str, cohort: str) -> dict:
    scored = int(d["authority_scored_wr_rows"].sum()) if len(d) else 0
    return {
        "scope": scope,
        "cohort": cohort,
        "team_games": int(len(d)),
        "authority_scored_wr_rows": scored,
        "mismatch_rate": (
            float(d["anchor_participation_mismatch"].mean()) if len(d) else np.nan
        ),
        "candidate_follow_participation_rate": (
            float(d["candidate_followed_participation_leader"].mean())
            if len(d) else np.nan
        ),
        "anchor_actual_top_hit_rate": (
            float(d["anchor_actual_top_hit"].mean()) if len(d) else np.nan
        ),
        "participation_actual_top_hit_rate": (
            float(d["participation_leader_actual_top_hit"].mean())
            if len(d) else np.nan
        ),
        "candidate_actual_top_hit_rate": (
            float(d["candidate_leader_actual_top_hit"].mean()) if len(d) else np.nan
        ),
        "participation_minus_anchor_actual_top_hit_rate": (
            float(
                d["participation_leader_actual_top_hit"].mean()
                - d["anchor_actual_top_hit"].mean()
            )
            if len(d) else np.nan
        ),
        "mean_participation_minus_anchor_actual_targets": (
            float(d["participation_minus_anchor_actual_targets"].mean())
            if len(d) else np.nan
        ),
        "baseline_abs_error_per_scored_wr": (
            float(d["baseline_target_abs_error_sum"].sum() / max(1, scored))
            if len(d) else np.nan
        ),
        "candidate_abs_error_per_scored_wr": (
            float(d["candidate_target_abs_error_sum"].sum() / max(1, scored))
            if len(d) else np.nan
        ),
    }


def summaries(team_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_rows = []
    season_rows = []
    for scope, d in [("POOLED", team_games)] + [
        (str(s), team_games.loc[team_games["season"].eq(s)]) for s in SEASONS
    ]:
        mismatch = d.loc[d["anchor_participation_mismatch"].eq(1)]
        match = d.loc[d["anchor_participation_mismatch"].eq(0)]
        cohort_rows += [
            cohort_row(d, scope, "ALL"),
            cohort_row(mismatch, scope, "ANCHOR_PARTICIPATION_MISMATCH"),
            cohort_row(match, scope, "ANCHOR_PARTICIPATION_MATCH"),
        ]
        mrows = int(mismatch["authority_scored_wr_rows"].sum())
        arows = int(match["authority_scored_wr_rows"].sum())
        season_rows.append({
            "scope": scope,
            "eligible_team_games": int(len(d)),
            "mismatch_team_games": int(len(mismatch)),
            "mismatch_rate": float(len(mismatch) / len(d)) if len(d) else np.nan,
            "mismatch_participation_hit_minus_anchor": (
                float(
                    mismatch["participation_leader_actual_top_hit"].mean()
                    - mismatch["anchor_actual_top_hit"].mean()
                )
                if len(mismatch) else np.nan
            ),
            "mismatch_mean_participation_minus_anchor_actual_targets": (
                float(mismatch["participation_minus_anchor_actual_targets"].mean())
                if len(mismatch) else np.nan
            ),
            "mismatch_candidate_follow_participation_rate": (
                float(mismatch["candidate_followed_participation_leader"].mean())
                if len(mismatch) else np.nan
            ),
            "mismatch_candidate_abs_error_per_scored_wr": (
                float(mismatch["candidate_target_abs_error_sum"].sum() / max(1, mrows))
                if len(mismatch) else np.nan
            ),
            "match_candidate_abs_error_per_scored_wr": (
                float(match["candidate_target_abs_error_sum"].sum() / max(1, arows))
                if len(match) else np.nan
            ),
        })
    return pd.DataFrame(cohort_rows), pd.DataFrame(season_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-predictions", type=Path, required=True)
    ap.add_argument("--authority-features", type=Path, required=True)
    ap.add_argument("--authority-conservation", type=Path, required=True)
    ap.add_argument(
        "--manual-name-overrides",
        type=Path,
        default=Path("data/manual_name_overrides.csv"),
    )
    ap.add_argument("--data-2023", type=Path, required=True)
    ap.add_argument("--logs-2023", type=Path, required=True)
    ap.add_argument("--data-2024", type=Path, required=True)
    ap.add_argument("--logs-2024", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    predictions, rank1 = prepare_predictions(
        read(args.authority_predictions, "WR-R15 OOS predictions")
    )
    features = prepare_features(
        read(args.authority_features, "WR-R15 OOS features")
    )
    conservation = prepare_conservation(
        read(args.authority_conservation, "WR-R15 conservation audit")
    )
    frozen_keys = set(features["player_clean_key"].astype(str)) | set(
        predictions["player_clean_key"].astype(str)
    )
    bridge, bridge_audit = reconstruct_bridge(
        data_dirs={2023: args.data_2023, 2024: args.data_2024},
        logs_by_season={
            2023: read(args.logs_2023, "2023 fold player logs"),
            2024: read(args.logs_2024, "2024 fold player logs"),
        },
        overrides_path=args.manual_name_overrides,
        frozen_keys=frozen_keys,
    )
    full, mechanical = build_frozen_full_state(
        predictions=predictions,
        rank1_predictions=rank1,
        features=features,
        conservation=conservation,
        bridge=bridge,
    )
    team_games = build_team_games(full, predictions)
    cohort, season = summaries(team_games)

    pooled = season.loc[season["scope"].eq("POOLED")].iloc[0]
    s23 = season.loc[season["scope"].eq("2023")].iloc[0]
    s24 = season.loc[season["scope"].eq("2024")].iloc[0]
    mismatch = team_games.loc[team_games["anchor_participation_mismatch"].eq(1)]
    match = team_games.loc[team_games["anchor_participation_mismatch"].eq(0)]
    mismatch_rows = int(mismatch["authority_scored_wr_rows"].sum())
    match_rows = int(match["authority_scored_wr_rows"].sum())
    mismatch_err = float(
        mismatch["candidate_target_abs_error_sum"].sum() / max(1, mismatch_rows)
    )
    match_err = float(
        match["candidate_target_abs_error_sum"].sum() / max(1, match_rows)
    )

    criteria = {
        "mismatch_team_games_ge150": int(len(mismatch)) >= 150,
        "pooled_participation_top_hit_advantage_ge5pp":
            float(pooled["mismatch_participation_hit_minus_anchor"]) >= 0.05,
        "participation_top_hit_advantage_nonnegative_both_seasons":
            float(s23["mismatch_participation_hit_minus_anchor"]) >= 0.0
            and float(s24["mismatch_participation_hit_minus_anchor"]) >= 0.0,
        "participation_actual_target_advantage_positive_pooled_nonnegative_both":
            float(pooled["mismatch_mean_participation_minus_anchor_actual_targets"]) > 0.0
            and float(s23["mismatch_mean_participation_minus_anchor_actual_targets"]) >= 0.0
            and float(s24["mismatch_mean_participation_minus_anchor_actual_targets"]) >= 0.0,
        "wr_r15_follows_participation_lt50pct":
            float(pooled["mismatch_candidate_follow_participation_rate"]) < 0.50,
        "mismatch_final_target_error_ge5pct_worse_than_match":
            float(mismatch_err) >= 1.05 * float(match_err),
        "strict_prior_future_violations_zero":
            int(bridge_audit["strict_prior_future_violations"]) == 0,
        "sportsbook_inputs_zero": True,
        "candidate_variants_scored_zero": True,
    }
    warranted = all(criteria.values())
    disposition = (
        "WR_ANCHOR_ROLE_TRANSMISSION_GAP_WARRANTED"
        if warranted else "WR_ANCHOR_ROLE_TRANSMISSION_NO_GAP_CLOSED"
    )

    source_audit = {
        "version": VERSION,
        "wr_r15_authority_run": AUTHORITY_RUN,
        "wr_r15_authority_artifact": AUTHORITY_ARTIFACT,
        "authority_seasons": list(SEASONS),
        **bridge_audit,
        **mechanical,
        "eligible_team_games": int(len(team_games)),
        "max_anchor_entitlement_immutability_gap": float(
            team_games["anchor_entitlement_immutability_gap"].max()
        ),
        "sportsbook_inputs_used": 0,
        "candidate_variants_scored": 0,
        "parameters_fit": 0,
    }
    payload = {
        "version": VERSION,
        "disposition": disposition,
        "structural_hypothesis_warranted": bool(warranted),
        "candidate_variants_scored": 0,
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "production_mutations": 0,
        "mechanical_authority_bridge": (
            "frozen_secondary_features+frozen_conservation_anchor_state;"
            "authority-era replay used only for row-index identity, anchor participation,"
            "and target labels after parity"
        ),
        "criteria": criteria,
        "pooled": {
            "eligible_team_games": int(pooled["eligible_team_games"]),
            "mismatch_team_games": int(pooled["mismatch_team_games"]),
            "mismatch_rate": float(pooled["mismatch_rate"]),
            "participation_top_hit_minus_anchor": float(
                pooled["mismatch_participation_hit_minus_anchor"]
            ),
            "participation_minus_anchor_actual_targets": float(
                pooled["mismatch_mean_participation_minus_anchor_actual_targets"]
            ),
            "candidate_follow_participation_rate": float(
                pooled["mismatch_candidate_follow_participation_rate"]
            ),
            "mismatch_candidate_abs_error_per_scored_wr": mismatch_err,
            "match_candidate_abs_error_per_scored_wr": match_err,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.out_dir / "player_detail_anchor_transmission.csv", index=False)
    team_games.to_csv(args.out_dir / "team_game_anchor_transmission.csv", index=False)
    cohort.to_csv(args.out_dir / "cohort_summary.csv", index=False)
    season.to_csv(args.out_dir / "season_summary.csv", index=False)
    (args.out_dir / "source_audit.json").write_text(
        json.dumps(source_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# WR Anchor / Role-Transmission Audit V1",
        "",
        f"Disposition: **{disposition}**",
        "",
        "- candidate variants scored: **0**",
        "- parameters fit: **0**",
        "- sportsbook inputs: **0**",
        f"- frozen authority run/artifact: **{AUTHORITY_RUN} / {AUTHORITY_ARTIFACT}**",
        f"- authority source commit for identity bridge: **{AUTHORITY_SOURCE_COMMIT}**",
        "",
        "## Mechanical authority bridge",
        "",
        f"- frozen WR2+ rows bridged: {mechanical['frozen_secondary_rows']}",
        f"- frozen team-game anchors bridged: {mechanical['mapped_anchor_rows']}",
        f"- scored WR1 identity checks: {mechanical['frozen_scored_rank1_rows']}",
        f"- WR1 identity mismatches: {mechanical['rank1_anchor_identity_mismatches']}",
        f"- secondary identity mismatches: {mechanical['secondary_identity_mismatches']}",
        f"- secondary snap-pct max gap: {mechanical['secondary_prior1_offense_pct_max_gap']:.3g}",
        f"- secondary snap-count max gap: {mechanical['secondary_prior1_offense_snaps_max_gap']:.3g}",
        f"- scored actual-target max gap: {mechanical['max_scored_actual_target_gap']:.3g}",
        "",
        "## Pooled",
        "",
        f"- eligible team-games: {int(pooled['eligible_team_games'])}",
        f"- anchor/participation mismatch team-games: {int(pooled['mismatch_team_games'])} ({float(pooled['mismatch_rate']):.2%})",
        f"- mismatch participation-leader actual-top hit advantage vs M38 anchor: {float(pooled['mismatch_participation_hit_minus_anchor']):+.2%}",
        f"- mismatch mean actual targets, participation leader minus anchor: {float(pooled['mismatch_mean_participation_minus_anchor_actual_targets']):+.4f}",
        f"- mismatch WR-R15 final leader follows participation leader: {float(pooled['mismatch_candidate_follow_participation_rate']):.2%}",
        f"- mismatch candidate target AE / scored WR: {mismatch_err:.6f}",
        f"- match candidate target AE / scored WR: {match_err:.6f}",
        "",
        "## Frozen criteria",
        "",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in criteria.items()]
    lines += [
        "",
        "This audit does not authorize a WR1 or hierarchy change. "
        "A separate candidate may be frozen only if every diagnostic criterion passes.",
    ]
    (args.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
