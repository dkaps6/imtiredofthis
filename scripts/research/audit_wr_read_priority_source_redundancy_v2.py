#!/usr/bin/env python3
"""Source-only WR read-priority V2 audit.

Uses the frozen semantic contract in WR_READ_PRIORITY_SEMANTIC_RESOLUTION_V2.md.
No WR receiving-yard outcomes are loaded or scored. 2024 WR-R15 projection/outcome
fields are never parsed or materialized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import audit_wr_read_priority_source_redundancy_v1 as base

MIN_CLASSIFIABLE_TARGETS = 16
MIN_COVERAGE = 0.60
MAX_ABS_ENTITLEMENT_SPEARMAN = 0.75
MAX_REDUNDANCY_R2 = 0.60


def normalize_progression_v2(value, season: int) -> str:
    """Frozen maintainer-approved semantics, collapsed to the V2 process states."""
    if value is None or pd.isna(value):
        return "EARLY_NO_EXTENDED" if int(season) == 2022 else "MISSING"
    z = str(value).strip().upper()
    if z in {"0", "0.0"}:
        return "EARLY_NO_EXTENDED"
    if z == "DES":
        return "EARLY_NO_EXTENDED"
    if z in {"1", "1.0", "2", "2.0"}:
        return "EXTENDED_PROGRESS"
    if z == "CHK":
        return "CHECKDOWN"
    if z == "SD":
        return "SCRAMBLE_DRILL"
    return f"UNKNOWN:{z}"


def load_targets_v2(seasons):
    # base loader performs the already-validated exact FTN->PBP join and strict
    # receiver-target population construction. Only the semantic normalizer changes.
    original = base.normalize_read
    try:
        base.normalize_read = normalize_progression_v2
        targets, raw_meta = base.load_ftn_pbp_targets(seasons)
    finally:
        base.normalize_read = original

    targets = targets.copy()
    targets["progression_classifiable"] = targets["read_norm"].isin(
        ["EARLY_NO_EXTENDED", "EXTENDED_PROGRESS"]
    )
    targets["is_early_no_extended"] = targets["read_norm"].eq("EARLY_NO_EXTENDED").astype(float)

    meta = []
    by_season = {int(m["season"]): m for m in raw_meta}
    for season in sorted({int(s) for s in seasons}):
        t = targets.loc[targets["season"].eq(season)].copy()
        counts = t["read_norm"].value_counts(dropna=False).to_dict()
        unknown = int(t["read_norm"].astype(str).str.startswith("UNKNOWN:").sum())
        classifiable = int(t["progression_classifiable"].sum())
        early = int((t["read_norm"] == "EARLY_NO_EXTENDED").sum())
        ext = int((t["read_norm"] == "EXTENDED_PROGRESS").sum())
        inherited = by_season[season]
        meta.append({
            "season": season,
            "ftn_sha256": inherited["ftn_sha256"],
            "ftn_bytes": inherited["ftn_bytes"],
            "pbp_sha256": inherited["pbp_sha256"],
            "pbp_bytes": inherited["pbp_bytes"],
            "exact_join_rate": inherited["exact_join_rate"],
            "season_week_parity": inherited["season_week_parity"],
            "receiver_target_rows": int(len(t)),
            "normalized_category_counts": {str(k): int(v) for k, v in counts.items()},
            "unknown_code_rows": unknown,
            "classifiable_progression_rows": classifiable,
            "classifiable_progression_rate": float(classifiable / len(t)) if len(t) else 0.0,
            "early_no_extended_rows": early,
            "extended_progress_rows": ext,
            "early_no_extended_share_classifiable": float(early / classifiable) if classifiable else None,
        })
    return targets, meta


def regression_r2(frame: pd.DataFrame) -> float:
    cols = ["early_no_extended_share8", "entitlement_tgt_share", "pred_targets", "wr1_indicator"]
    x = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(x) < 10 or x["early_no_extended_share8"].nunique() < 2:
        return float("nan")
    y = x["early_no_extended_share8"].to_numpy(dtype=float)
    X = np.column_stack([
        np.ones(len(x)),
        x["entitlement_tgt_share"].to_numpy(dtype=float),
        x["pred_targets"].to_numpy(dtype=float),
        x["wr1_indicator"].to_numpy(dtype=float),
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def audit_v2(authority: pd.DataFrame, targets: pd.DataFrame, rosters: pd.DataFrame):
    rows = []
    identity_counts: dict[str, int] = {}
    for r in authority.itertuples(index=False):
        hist, ident = base.resolve_prior_receiver_history(
            targets, rosters, r.player_clean_key, r.team, r.season, r.week
        )
        identity_counts[ident["mode"]] = identity_counts.get(ident["mode"], 0) + 1
        selected = base._last_games(hist, base.PRIOR_GAMES)
        n_games = int(selected[["season", "week", "game_id"]].drop_duplicates().shape[0]) if len(selected) else 0
        n_targets = int(len(selected))
        c = selected.loc[selected["progression_classifiable"]].copy() if len(selected) else selected.copy()
        n_classifiable = int(len(c))
        supported = (
            ident["mode"] == "id"
            and n_games >= base.MIN_PRIOR_TARGET_GAMES
            and n_classifiable >= MIN_CLASSIFIABLE_TARGETS
        )
        share = float(c["is_early_no_extended"].mean()) if supported and n_classifiable else float("nan")
        rows.append({
            "season": int(r.season), "week": int(r.week), "team": r.team,
            "player_clean_key": r.player_clean_key, "player": r.player,
            "wr_rank": int(r.wr_rank), "wr1_indicator": int(int(r.wr_rank) == 1),
            "pred_targets": float(r.pred_targets), "entitlement_tgt_share": float(r.entitlement_tgt_share),
            "identity_mode": ident["mode"], "identity_source": ident.get("identity_source", ""),
            "resolved_receiver_id": ident.get("player_id", ""),
            "prior_target_games8": n_games, "prior_targets8": n_targets,
            "prior_classifiable_progression_targets8": n_classifiable,
            "early_no_extended_share8": share,
            "supported": bool(supported),
        })

    panel = pd.DataFrame(rows)
    supported = panel.loc[panel["supported"] & panel["early_no_extended_share8"].notna()].copy()
    coverage = float(len(supported) / len(panel)) if len(panel) else 0.0
    ent_p = base.pearson(supported["early_no_extended_share8"], supported["entitlement_tgt_share"])
    ent_s = base.spearman(supported["early_no_extended_share8"], supported["entitlement_tgt_share"])
    tgt_p = base.pearson(supported["early_no_extended_share8"], supported["pred_targets"])
    tgt_s = base.spearman(supported["early_no_extended_share8"], supported["pred_targets"])
    r2 = regression_r2(supported)

    quartiles = []
    if len(supported):
        q = pd.qcut(supported["entitlement_tgt_share"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        for label, g in supported.assign(entitlement_quartile=q).groupby("entitlement_quartile", observed=True):
            s = g["early_no_extended_share8"]
            quartiles.append({
                "quartile": str(label), "n": int(len(g)), "mean": float(s.mean()),
                "sd": float(s.std(ddof=0)), "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            })

    return panel, {
        "authority_rows_2023": int(len(panel)),
        "supported_rows": int(len(supported)),
        "coverage": coverage,
        "identity_mode_counts": identity_counts,
        "early_vs_entitlement_pearson": ent_p,
        "early_vs_entitlement_spearman": ent_s,
        "early_vs_pred_targets_pearson": tgt_p,
        "early_vs_pred_targets_spearman": tgt_s,
        "early_explained_by_entitlement_predtargets_wr1_r2": r2,
        "entitlement_quartile_dispersion": quartiles,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    authority, authority_meta = base.load_authority_opportunity_only(args.authority)
    targets, source_meta = load_targets_v2([2022, 2023, 2024])
    rosters = base.load_roster_identity([2022, 2023])
    panel, redundancy = audit_v2(
        authority,
        targets.loc[targets["season"].isin([2022, 2023])].copy(),
        rosters,
    )

    source_ok = all(
        float(x["exact_join_rate"]) >= base.MIN_JOIN_RATE and int(x["unknown_code_rows"]) == 0
        for x in source_meta
    )
    coverage_ok = float(redundancy["coverage"]) >= MIN_COVERAGE
    ent_s = redundancy["early_vs_entitlement_spearman"]
    r2 = redundancy["early_explained_by_entitlement_predtargets_wr1_r2"]
    nonredundant = bool(
        np.isfinite(ent_s) and abs(float(ent_s)) < MAX_ABS_ENTITLEMENT_SPEARMAN
        and np.isfinite(r2) and float(r2) < MAX_REDUNDANCY_R2
    )
    disposition = (
        "READ_PRIORITY_R20_PLAN_ELIGIBLE_V2"
        if source_ok and coverage_ok and nonredundant
        else "READ_PRIORITY_SOURCE_REDUNDANCY_BLOCKED_V2"
    )

    result = {
        "specification": "WR_POST_R19_READ_PRIORITY_SOURCE_REDUNDANCY_AUDIT_V2",
        "feature": "EARLY_NO_EXTENDED_SHARE8",
        "disposition": disposition,
        "source_ok": source_ok,
        "coverage_ok": coverage_ok,
        "nonredundant_under_frozen_screen": nonredundant,
        "semantic_contract": {
            "2022_early_no_extended": ["NA", "DES"],
            "2023plus_early_no_extended": ["0", "DES"],
            "extended_progression_all_seasons": ["1", "2"],
            "excluded_from_binary_denominator": ["CHK", "SD", "UNKNOWN"],
            "last_prior_target_bearing_games": 8,
            "min_prior_target_bearing_games": 4,
            "min_classifiable_progression_targets": MIN_CLASSIFIABLE_TARGETS,
        },
        "thresholds": {
            "min_join_rate": base.MIN_JOIN_RATE,
            "min_supported_coverage": MIN_COVERAGE,
            "max_abs_entitlement_spearman": MAX_ABS_ENTITLEMENT_SPEARMAN,
            "max_redundancy_r2": MAX_REDUNDANCY_R2,
        },
        "authority_meta": authority_meta,
        "source_meta": source_meta,
        "redundancy": redundancy,
        "wr_outcomes_loaded": False,
        "sportsbook_inputs": 0,
        "production_change": False,
    }

    panel.to_csv(args.out_dir / "wr_read_priority_v2_source_feature_panel_2023.csv", index=False)
    (args.out_dir / "wr_read_priority_v2_source_redundancy_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "wr_read_priority_v2_source_meta.json").write_text(
        json.dumps(source_meta, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "disposition": disposition,
        "coverage": redundancy["coverage"],
        "early_vs_entitlement_spearman": ent_s,
        "early_vs_pred_targets_spearman": redundancy["early_vs_pred_targets_spearman"],
        "redundancy_r2": r2,
        "wr_outcomes_loaded": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
