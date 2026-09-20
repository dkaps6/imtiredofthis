#!/usr/bin/env python3
"""Outcome-free qualification for exact BDB2023 blocker-defender assignment support.

Consumes the certified BDB2023 protection-interaction materialization. The source
relationship is PFF pff_nflIdBlockedPlayer, i.e. an exact realized blocker->blocked
defender interaction for the target play. This qualifier does NOT read pressure,
hit, hurry, sack, or QB outcomes. Its only purpose is to decide whether the public
2021 Weeks 1-8 slice has enough identity and strict-prior support to justify a
separately frozen, explicitly hindsight/nondeployable value-of-information test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_HASH = "1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182"
MIN_STABLE_ID_COVERAGE = 0.99
MIN_TOTAL_EDGES = 1000
MIN_WEEKS = 8
MIN_LATE_EDGES = 500
MIN_PRIOR_PLAYER_EDGES = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _norm_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        v = float(text)
        if np.isfinite(v) and v.is_integer():
            return str(int(v))
    except (TypeError, ValueError):
        pass
    return text


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    low = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in low:
            return low[name.lower()]
    return None


def build_direct_identity_bridge(
    ids: set[str], players: pd.DataFrame
) -> tuple[dict[str, str], dict[str, object]]:
    p = players.copy()
    nfl_col = _first_col(p, ["nfl_id", "nfl_player_id", "nflid", "nfl_player_id_number"])
    gsis_col = _first_col(p, ["gsis_id", "player_id", "player_gsis_id"])
    if not nfl_col or not gsis_col:
        raise RuntimeError(
            "nflverse player crosswalk lacks direct nfl_id/gsis_id columns; "
            f"columns={sorted(p.columns.tolist())}"
        )
    q = p[[nfl_col, gsis_col]].copy()
    q["nfl_id_norm"] = q[nfl_col].map(_norm_id)
    q["gsis_id_norm"] = q[gsis_col].map(_norm_id)
    q = q.loc[q["nfl_id_norm"].ne("") & q["gsis_id_norm"].ne("")].drop_duplicates()

    ambiguity = q.groupby("nfl_id_norm")["gsis_id_norm"].nunique()
    ambiguous = set(ambiguity[ambiguity.gt(1)].index.astype(str))
    safe = q.loc[~q["nfl_id_norm"].isin(ambiguous)].drop_duplicates("nfl_id_norm")
    mapping = dict(zip(safe["nfl_id_norm"].astype(str), safe["gsis_id_norm"].astype(str)))

    mapped = sum(i in mapping for i in ids)
    coverage = float(mapped / len(ids)) if ids else 0.0
    return mapping, {
        "unique_bdb_assignment_ids": int(len(ids)),
        "mapped_to_stable_gsis": int(mapped),
        "stable_id_coverage": coverage,
        "ambiguous_bdb_nfl_ids": int(len(ambiguous & ids)),
        "name_fallback_used": False,
    }


def strict_prior_support(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    required = {"week", "blocker_nfl_id", "blocked_nfl_id"}
    missing = required - set(edges.columns)
    if missing:
        raise RuntimeError(f"interaction table missing columns: {sorted(missing)}")
    e = edges.copy()
    e["week"] = pd.to_numeric(e["week"], errors="coerce").astype("Int64")
    e["blocker_nfl_id"] = e["blocker_nfl_id"].map(_norm_id)
    e["blocked_nfl_id"] = e["blocked_nfl_id"].map(_norm_id)
    e = e.loc[
        e["week"].notna()
        & e["blocker_nfl_id"].ne("")
        & e["blocked_nfl_id"].ne("")
    ].copy()

    parts: list[pd.DataFrame] = []
    prior = e.iloc[0:0].copy()
    for week in sorted(int(x) for x in e["week"].dropna().unique()):
        cur = e.loc[e["week"].eq(week)].copy()
        hist = prior.loc[pd.to_numeric(prior["week"], errors="coerce").lt(week)]
        blocker_counts = hist.groupby("blocker_nfl_id").size()
        defender_counts = hist.groupby("blocked_nfl_id").size()
        pair_counts = hist.groupby(["blocker_nfl_id", "blocked_nfl_id"]).size()
        cur["prior_blocker_edges"] = (
            cur["blocker_nfl_id"].map(blocker_counts).fillna(0).astype(int)
        )
        cur["prior_defender_edges"] = (
            cur["blocked_nfl_id"].map(defender_counts).fillna(0).astype(int)
        )
        cur["prior_pair_edges"] = [
            int(pair_counts.get((b, d), 0))
            for b, d in zip(cur["blocker_nfl_id"], cur["blocked_nfl_id"])
        ]
        parts.append(cur)
        prior = pd.concat([prior, cur], ignore_index=True)
    return pd.concat(parts, ignore_index=True)


def run(
    materialized_dir: Path,
    players_path: Path,
    out_dir: Path,
    *,
    active_sha: str,
    materializer_sha: str,
) -> dict[str, object]:
    private = materialized_dir / "private"
    sanitized = materialized_dir / "sanitized"
    interactions_path = private / "bdb2023_protection_interactions_v1.csv"
    qa_path = sanitized / "bdb2023_protection_geometry_advanced_feature_qa_v1.json"
    if not qa_path.exists():
        matches = list(sanitized.glob("*qa*.json"))
        if len(matches) != 1:
            raise RuntimeError(f"unable to resolve certified BDB2023 QA file: {matches}")
        qa_path = matches[0]

    for p in [interactions_path, players_path, qa_path]:
        if not p.exists():
            raise RuntimeError(f"required input missing: {p}")

    qa = json.loads(qa_path.read_text())
    observed_hash = str(qa.get("source_hash_observed", ""))
    if observed_hash != SOURCE_HASH or not qa.get("source_hash_verified", False):
        raise RuntimeError(
            f"certified BDB2023 source hash mismatch: {observed_hash} != {SOURCE_HASH}"
        )

    edges = pd.read_csv(interactions_path, low_memory=False)
    players = pd.read_csv(players_path, low_memory=False)
    required = {
        "source_season", "week", "game_id", "play_id",
        "blocker_nfl_id", "blocked_nfl_id",
        "block_interaction_role", "blocked_defender_source_role",
    }
    missing = required - set(edges.columns)
    if missing:
        raise RuntimeError(f"certified interaction table missing columns: {sorted(missing)}")

    if not pd.to_numeric(edges["source_season"], errors="coerce").dropna().eq(2021).all():
        raise RuntimeError("BDB2023 exact assignment source unexpectedly contains non-2021 season")
    duplicate_key_rows = int(
        edges.duplicated(
            ["week", "game_id", "play_id", "blocker_nfl_id", "blocked_nfl_id"],
            keep=False,
        ).sum()
    )

    supported = strict_prior_support(edges)
    all_ids = set(supported["blocker_nfl_id"].astype(str))
    all_ids |= set(supported["blocked_nfl_id"].astype(str))
    all_ids.discard("")
    _, identity = build_direct_identity_bridge(all_ids, players)

    weeks = sorted(int(x) for x in supported["week"].dropna().unique())
    late = supported.loc[pd.to_numeric(supported["week"], errors="coerce").ge(5)].copy()
    late["both_players_prior10"] = (
        late["prior_blocker_edges"].ge(MIN_PRIOR_PLAYER_EDGES)
        & late["prior_defender_edges"].ge(MIN_PRIOR_PLAYER_EDGES)
    )
    late["prior_same_pair"] = late["prior_pair_edges"].ge(1)

    by_week = (
        supported.groupby("week")
        .agg(
            assignment_edges=("blocked_nfl_id", "size"),
            unique_blockers=("blocker_nfl_id", "nunique"),
            unique_defenders=("blocked_nfl_id", "nunique"),
            median_prior_blocker_edges=("prior_blocker_edges", "median"),
            median_prior_defender_edges=("prior_defender_edges", "median"),
            prior_pair_edge_rate=("prior_pair_edges", lambda x: float(pd.Series(x).ge(1).mean())),
        )
        .reset_index()
    )

    interaction_roles = (
        supported["block_interaction_role"].fillna("UNKNOWN").astype(str).value_counts()
        .rename_axis("block_interaction_role").reset_index(name="edges")
    )
    defender_roles = (
        supported["blocked_defender_source_role"].fillna("UNKNOWN").astype(str).value_counts()
        .rename_axis("blocked_defender_source_role").reset_index(name="edges")
    )

    structural = qa.get("structural_audit", {})
    chronology = qa.get("temporal_audit", {})
    materializer_integrity_ok = (
        int(structural.get("duplicate_blocker_target_pairs", -1)) == 0
        and int(chronology.get("target_game_rows_used", -1)) == 0
    )
    ready = (
        int(len(supported)) >= MIN_TOTAL_EDGES
        and len(weeks) >= MIN_WEEKS
        and identity["stable_id_coverage"] >= MIN_STABLE_ID_COVERAGE
        and identity["ambiguous_bdb_nfl_ids"] == 0
        and duplicate_key_rows == 0
        and materializer_integrity_ok
        and int(late["both_players_prior10"].sum()) >= MIN_LATE_EDGES
    )
    disposition = (
        "VALUE_OF_INFORMATION_LAB_READY_SOURCE_SLICE"
        if ready
        else "SOURCE_SLICE_INSUFFICIENT_FOR_FROZEN_VOI_LAB"
    )

    report = {
        "audit": "BDB2023_EXACT_BLOCKER_RUSHER_ASSIGNMENT_QUALIFICATION_V1",
        "active_sha": active_sha,
        "certified_materializer_sha": materializer_sha,
        "certified_source_sha256": SOURCE_HASH,
        "materialized_interactions_sha256": sha256(interactions_path),
        "source_season": 2021,
        "weeks": weeks,
        "assignment_edges": int(len(supported)),
        "unique_blockers": int(supported["blocker_nfl_id"].nunique()),
        "unique_blocked_defenders": int(supported["blocked_nfl_id"].nunique()),
        "unique_pairs": int(
            supported[["blocker_nfl_id", "blocked_nfl_id"]].drop_duplicates().shape[0]
        ),
        "late_week_edges_w5plus": int(len(late)),
        "late_week_edges_with_both_players_prior10": int(late["both_players_prior10"].sum()),
        "late_week_both_players_prior10_coverage": (
            float(late["both_players_prior10"].mean()) if len(late) else 0.0
        ),
        "late_week_edges_with_prior_same_pair": int(late["prior_same_pair"].sum()),
        "late_week_prior_same_pair_coverage": (
            float(late["prior_same_pair"].mean()) if len(late) else 0.0
        ),
        "identity": identity,
        "duplicate_assignment_key_rows": duplicate_key_rows,
        "materializer_integrity_ok": bool(materializer_integrity_ok),
        "source_relationship": "PFF pff_nflIdBlockedPlayer exact realized blocker-to-blocked-player relationship",
        "target_game_assignment_is_realized_hindsight": True,
        "pressure_hit_hurry_sack_values_read": False,
        "qb_outcomes_read": False,
        "sportsbook_read": False,
        "deployable_live_source_contract": False,
        "production_change_authorized": False,
        "issue_535_touched": False,
        "disposition": disposition,
        "next_if_ready": (
            "Freeze one explicitly nondeployable target-game assignment value-of-information "
            "experiment. Compare strictly-prior blocker/rusher quality alone against the same "
            "quality weighted by realized exact target-game assignment exposure. Use the result "
            "only to decide whether acquiring a historical+live assignment source is worth it."
        ),
        "rb_research_note": (
            "RB remains unresolved outside its qualified scopes. RB-PD2 yard-difficulty "
            "MC-width is a positive research qualification awaiting separate forward/shadow "
            "confirmation; Weeks 2-18 RB production authority remains unresolved."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    by_week.to_csv(out_dir / "bdb2023_exact_assignment_support_by_week_v1.csv", index=False)
    interaction_roles.to_csv(out_dir / "bdb2023_blocker_role_support_v1.csv", index=False)
    defender_roles.to_csv(out_dir / "bdb2023_blocked_defender_role_support_v1.csv", index=False)
    (out_dir / "bdb2023_exact_assignment_qualification_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialized-dir", required=True, type=Path)
    ap.add_argument("--players", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--active-sha", required=True)
    ap.add_argument("--materializer-sha", required=True)
    args = ap.parse_args()
    print(
        json.dumps(
            run(
                args.materialized_dir,
                args.players,
                args.out_dir,
                active_sha=args.active_sha,
                materializer_sha=args.materializer_sha,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
